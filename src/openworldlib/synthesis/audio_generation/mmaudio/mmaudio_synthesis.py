# python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
import logging

import torch
import os
from loguru import logger
from huggingface_hub import snapshot_download

from .mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, make_video)
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio, get_my_mmaudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
from ...base_synthesis import BaseSynthesis


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

def _maybe_resolve_hf_cache_snapshot_dir(
    maybe_dir: str | os.PathLike,
    *,
    expected_files: tuple[str, ...],
) -> str:
    """
    将 HuggingFace 的本地缓存目录（例如 models--org--repo/ 或 snapshots/）解析到真实 snapshot 目录。

    支持以下输入：
    - 直接指向 snapshot 的目录（包含 expected_files）→ 原样返回
    - 指向 models--org--repo/ → 读取 refs/main 选取 snapshots/<hash>；否则取第一个 snapshot
    - 指向 .../snapshots/ → 取第一个子目录
    """
    p = Path(maybe_dir).expanduser()
    if not p.exists():
        return str(maybe_dir)

    if p.is_file():
        p = p.parent

    def _has_expected_files(d: Path) -> bool:
        return all((d / f).exists() for f in expected_files)

    if _has_expected_files(p):
        return str(p)

    # If user points to ".../snapshots"
    if p.name == "snapshots" and p.is_dir():
        children = [c for c in p.iterdir() if c.is_dir()]
        children.sort()
        for c in children:
            if _has_expected_files(c):
                return str(c)
        return str(children[0]) if children else str(p)

    # If user points to "models--org--repo" root
    snapshots_dir = p / "snapshots"
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        # Prefer refs/main -> snapshot hash
        ref_main = p / "refs" / "main"
        if ref_main.exists():
            try:
                rev = ref_main.read_text(encoding="utf-8").strip()
                candidate = snapshots_dir / rev
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
            except Exception:
                pass

        children = [c for c in snapshots_dir.iterdir() if c.is_dir()]
        children.sort()
        for c in children:
            if _has_expected_files(c):
                return str(c)
        return str(children[0]) if children else str(p)

    return str(p)


def _normalize_required_components_paths(required_components: Dict[str, str]) -> Dict[str, str]:
    """
    仿照 Cosmos 的实现方式：
    - 如果 Path(x).exists()：按“本地文件/目录”处理
    - 否则：按“远端 HuggingFace repo_id / schema”处理（保留原样，走原下载/缓存逻辑）

    同时适配 MMAudio 的两个组件：
    - CLIP(open_clip): 需要 open_clip_config.json + open_clip_pytorch_model.bin
    - BigVGANv2(44k): 需要 config.json + bigvgan_generator.pt
    """
    out = dict(required_components or {})

    # --- CLIP (open_clip) ---
    clip_model_path = out.get("clip_model_path")
    if isinstance(clip_model_path, str) and clip_model_path:
        # open_clip 已支持 schema：hf-hub:... / local-dir:...
        if not clip_model_path.startswith(("hf-hub:", "local-dir:")):
            p = Path(clip_model_path).expanduser()
            if p.exists():
                clip_dir = _maybe_resolve_hf_cache_snapshot_dir(
                    str(p),
                    expected_files=("open_clip_config.json", "open_clip_pytorch_model.bin"),
                )
                clip_dir_p = Path(clip_dir)
                if (clip_dir_p / "open_clip_config.json").exists() and (clip_dir_p / "open_clip_pytorch_model.bin").exists():
                    # open_clip 要求本地目录使用 local-dir: schema
                    out["clip_model_path"] = f"local-dir:{clip_dir_p.resolve()}"

    # --- BigVGANv2 (44k) ---
    vocoder_44k = out.get("vocoder_ckpt_path_44k")
    if isinstance(vocoder_44k, str) and vocoder_44k:
        p = Path(vocoder_44k).expanduser()
        if p.exists():
            vocoder_dir = _maybe_resolve_hf_cache_snapshot_dir(
                str(p),
                expected_files=("config.json", "bigvgan_generator.pt"),
            )
            vocoder_dir_p = Path(vocoder_dir)
            if (vocoder_dir_p / "config.json").exists() and (vocoder_dir_p / "bigvgan_generator.pt").exists():
                # BigVGANv2.from_pretrained 支持直接传入本地目录
                out["vocoder_ckpt_path_44k"] = str(vocoder_dir_p.resolve())

    return out


def load_models(
    variant: str,
    full_precision: bool,
    num_steps: int,
    model_path,
    device,
    logger_obj,
    required_components,
):
    """
    加载 MMAudio 模型
    
    Args:
        args: 配置参数，包含 variant, full_precision, num_steps 等
        device: 设备 (cuda/cpu/mps)
        logger_obj: 日志记录器
        
    Returns:
        net: MMAudio 主网络
        feature_utils: 特征工具
        fm: FlowMatching 实例
        seq_cfg: 序列配置
        model: ModelConfig 实例
    """
    if logger_obj:
        logger_obj.info(f"Loading MMAudio model variant: {variant}")
    

    if os.path.isdir(model_path):
        model_root = Path(model_path)
    else:
        repo_id = model_path
        folder_name = repo_id.split("/")[-1]
        download_dir = Path(os.getcwd()) / folder_name
        model_root = Path(
            snapshot_download(
                repo_id,
                local_dir=str(download_dir),
                local_dir_use_symlinks=False,
            )
        )
    
    # 基于 model_root 解析出当前使用的 ModelConfig，使所有权重路径都在 model_root 下
    base_model: ModelConfig = all_model_cfg[variant]
    model: ModelConfig = base_model.with_root(model_root)
    
    # 序列配置只依赖于模式，不依赖具体路径
    seq_cfg = model.seq_cfg

    dtype = torch.float32 if full_precision else torch.bfloat16
    
    if logger_obj:
        logger_obj.info(f"Loading network: {model.model_name} on {device} with dtype {dtype}")
    
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    if logger_obj:
        logger_obj.info("Loading feature utils...")

    required_components = _normalize_required_components_paths(required_components)
    
    # vocoder_ckpt_path 支持两种形式：
    # - 16k: 使用 modelconfig 中的本地 BigVGAN 权重路径
    # - 44k: 使用 BigVGANv2 的 HuggingFace repo_id（默认 nvidia/bigvgan_v2_44khz_128band_512x）
    if getattr(model, "mode", None) == "44k":
        vocoder_ckpt_path = required_components.get("vocoder_ckpt_path_44k", "nvidia/bigvgan_v2_44khz_128band_512x")
    else:
        vocoder_ckpt_path = model.bigvgan_16k_path

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        clip_model_path=required_components["clip_model_path"],
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=vocoder_ckpt_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    if logger_obj:
        logger_obj.info(f"Successfully loaded MMAudio model from {model.model_path}")

    return net, feature_utils, fm, seq_cfg, model
    

class MMAudioSynthesis(BaseSynthesis):
    """
    MMAudio 生成合成类，提供统一的接口用于音频生成
    """
    def __init__(self, variant: str, full_precision: bool, num_steps: int,
                 net, feature_utils, fm, seq_cfg, model_config, device, logger_obj, required_components):
        """
        初始化 MMAudioSynthesis
        
        Args:
            variant: 模型变体名称
            full_precision: 是否使用全精度（float32）
            num_steps: FlowMatching 推理步数
            net: MMAudio 主网络
            feature_utils: 特征工具
            fm: FlowMatching 实例
            seq_cfg: 序列配置
            model_config: 模型配置
            device: 设备
            logger_obj: 日志记录器
        """
        self.variant = variant
        self.full_precision = full_precision
        self.num_steps = num_steps
        self.device = device
        self.logger = logger_obj
        self.net = net
        self.feature_utils = feature_utils
        self.fm = fm
        self.seq_cfg = seq_cfg
        self.model_config = model_config
        self.required_components = required_components
        # 初始化随机数生成器
        self.rng = torch.Generator(device=device)
        
        if self.logger:
            self.logger.info("MMAudioSynthesis initialized successfully")
        
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        required_components: Dict[str, str],
        variant: str = "large_44k_v2",
        full_precision: bool = False,
        num_steps: int = 25,
        device=None,
        logger_obj=None,
        **kwargs,
    ):
        """
        从预训练模型路径加载 MMAudioSynthesis
        
        Args:
            model_path: 预训练模型路径，可以是本地路径或者hugid路径
            required_components: 所需的组件，包含 tod_vae_ckpt, clip_model_path
            variant: 模型变体名称
            full_precision: 是否使用全精度（float32）
            num_steps: FlowMatching 推理步数
            device: 设备，默认为 None（自动检测）
            logger_obj: 日志记录器，默认为 None
            **kwargs: 额外参数
            
        Returns:
            MMAudioSynthesis 实例
        """
        
        logger_inst = logger_obj
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
                if logger_inst:
                    logger_inst.warning('CUDA/MPS are not available, running on CPU')
        
        torch.set_grad_enabled(False)
        
        # 加载模型组件
        net, feature_utils, fm, seq_cfg, model_config = load_models(
            variant=variant,
            full_precision=full_precision,
            num_steps=num_steps,
            model_path=model_path,
            device=device,
            logger_obj=logger_inst,
            required_components=required_components,
        )
        
        return cls(
            variant=variant,
            full_precision=full_precision,
            num_steps=num_steps,
            net=net,
            feature_utils=feature_utils,
            fm=fm,
            seq_cfg=seq_cfg,
            model_config=model_config,
            device=device,
            logger_obj=logger_inst,
            required_components=required_components,
        )

    @torch.no_grad()
    def predict(
        self, 
        processed_data: Dict[str, any], 
        seed: Optional[Union[int, List[int]]] = None, 
        cfg_strength: float = 4.5, 
        **kwargs
    ) -> Dict:
        """
        生成音频预测结果
        
        Args:
            processed_data: 从 operator 处理后的数据，包含：
                - clip_frames: CLIP 视频帧 [B, C, T, H, W] 或 None
                - sync_frames: Sync 视频帧 [B, C, T, H, W] 或 None
                - prompt: 文本提示列表
                - negative_prompt: 负面提示列表
                - duration: 音频时长（秒）
                - video_info: VideoInfo 对象（可选，用于后续视频合成）
            seed: 随机种子
            cfg_strength: Classifier-free guidance 强度
            num_steps: 生成步数，如果为 None 则使用 args 中的值
            **kwargs: 额外参数
            
        Returns:
            Dict 包含生成的结果：
                - audio: 生成的音频 tensor [seq_len]
                - sampling_rate: 采样率
                - duration: 实际音频时长
                - prompt: 使用的 prompt
                - video_info: VideoInfo 对象（如果有）
        """
        
        # 设置随机种子
        self.rng.manual_seed(seed)
        if self.logger:
            self.logger.info(f"Using seed: {seed}")
        
        duration = processed_data["duration"]
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len, 
            self.seq_cfg.clip_seq_len, 
            self.seq_cfg.sync_seq_len
        )
        
        if self.logger:
            self.logger.info(f"Generating audio for duration: {duration}s with cfg_strength: {cfg_strength}")
        
        clip_frames = processed_data["clip_frames"]
        sync_frames = processed_data["sync_frames"]
        prompt = processed_data["prompt"]
        negative_prompt = processed_data["negative_prompt"]
        
        audios = generate(
            clip_frames, 
            sync_frames, 
            prompt, 
            negative_text=negative_prompt, 
            feature_utils=self.feature_utils, 
            net=self.net, 
            fm=self.fm, 
            rng=self.rng, 
            cfg_strength=cfg_strength
        )
        
        audio = audios.float().cpu()[0]
        
        
        result = {
            "audio": audio,
            "sampling_rate": self.seq_cfg.sampling_rate,
            "duration": duration,
            "prompt": prompt[0] if isinstance(prompt, list) and len(prompt) > 0 else prompt,
        }
        
        if "video_info" in processed_data and processed_data["video_info"] is not None:
            result["video_info"] = processed_data["video_info"]
        
        return result
