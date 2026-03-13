from typing import Literal, Optional
import json
import os
from pathlib import Path
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict
from ThinkSound.models.utils import copy_state_dict
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import T5EncoderModel, AutoTokenizer
import logging
from data_utils.ext.synchformer import Synchformer

log = logging.getLogger()

# 统一控制 HuggingFace 模型的下载路径：
# 会下载到「当前运行目录」下的 huggingface_cache/ 里
THINKSOUND_HF_CACHE_DIR = os.path.join(os.getcwd(), "huggingface_cache")
os.makedirs(THINKSOUND_HF_CACHE_DIR, exist_ok=True)


def _resolve_local_or_repo(path_or_repo: str) -> str:
    """
    参考 MMAudio / Cosmos 的风格：
    - 如果字符串对应的本地路径存在（文件或目录），优先当作本地路径使用；
    - 否则当作 HuggingFace repo_id / model_id 交给 transformers.from_pretrained 处理。
    """
    p = Path(path_or_repo).expanduser()
    if not p.exists():
        return path_or_repo

    if p.is_file():
        return str(p)

    # 兼容 HuggingFace Hub 的本地缓存结构：
    # models--org--repo/{blobs,refs,snapshots} 或者直接指向 snapshots/ 目录。
    # transformers.from_pretrained 需要的是 snapshots/<hash>/ 这种“包含真实文件”的目录，
    # 直接传 models--org--repo 根目录会找不到 pytorch_model.bin / model.safetensors。
    try:
        # if user points to ".../snapshots"
        if p.name == "snapshots" and p.is_dir():
            children = [c for c in p.iterdir() if c.is_dir()]
            children.sort()
            return str(children[0]) if children else str(p)

        snapshots_dir = p / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            ref_main = p / "refs" / "main"
            if ref_main.exists():
                rev = ref_main.read_text(encoding="utf-8").strip()
                candidate = snapshots_dir / rev
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
            # fallback: first snapshot dir
            children = [c for c in snapshots_dir.iterdir() if c.is_dir()]
            children.sort()
            return str(children[0]) if children else str(p)
    except Exception:
        # 解析失败就退回原路径（至少不影响普通本地目录）
        pass

    return str(p)

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_get_text_features(self, input_ids=None, attention_mask=None, position_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        last_hidden_state = text_outputs[0]
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features, last_hidden_state

    clip_model.get_text_features = new_get_text_features.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):
 
    def __init__(
        self,
        *, 
        vae_ckpt: Optional[str] = None,
        vae_config: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        clip_backbone_id: str = "facebook/metaclip-h14-fullcc2.5b",
        t5_model_id: str = "google/t5-v1_1-xl",
        clip_processor_id: str = "openai/clip-vit-large-patch14",
        enable_conditions: bool = True,
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            # 允许将默认的 repo_id 替换成本地路径（与 MMAudio 的 required_components 处理方式一致）：
            # - 若调用方通过 required_components / CLI 传入自定义路径或 repo_id，则优先使用该值
            # - 若保持默认，则使用内置的模型 ID，并自动下载到 THINKSOUND_HF_CACHE_DIR。
            clip_backbone_id = _resolve_local_or_repo(clip_backbone_id)
            t5_id = _resolve_local_or_repo(t5_model_id)
            clip_processor_id = _resolve_local_or_repo(clip_processor_id)

            self.clip_model = AutoModel.from_pretrained(
                clip_backbone_id,
                cache_dir=THINKSOUND_HF_CACHE_DIR,
            )
            self.clip_model = patch_clip(self.clip_model)
            self.t5_tokenizer = AutoTokenizer.from_pretrained(
                t5_id,
                cache_dir=THINKSOUND_HF_CACHE_DIR,
                use_fast=False,
            )
            self.t5_model = T5EncoderModel.from_pretrained(
                t5_id,
                cache_dir=THINKSOUND_HF_CACHE_DIR,
            )
            self.clip_processor = AutoProcessor.from_pretrained(
                clip_processor_id,
                cache_dir=THINKSOUND_HF_CACHE_DIR,
            )
            # self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
            #                                  std=[0.26862954, 0.26130258, 0.27577711])
            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))

            # self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
        else:
            self.clip_model = None
            self.synchformer = None
            self.tokenizer = None

        if vae_ckpt is not None:
            with open(vae_config) as f:
                vae_config = json.load(f)
            self.vae = create_model_from_config(vae_config)
            print(f"Loading model checkpoint from {vae_ckpt}")
            # Load checkpoint
            copy_state_dict(self.vae, load_ckpt_state_dict(vae_ckpt,prefix='autoencoder.'))#,prefix='autoencoder.'
        else:
            self.tod = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)


    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        
        assert c == 3 and h == 224 and w == 224
        # x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.get_image_features(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        # import ipdb
        # ipdb.set_trace()
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.clip_processor(text=text, truncation=True, max_length=77, padding="max_length",return_tensors="pt").to(self.device)
        return self.clip_model.get_text_features(**tokens)

    @torch.inference_mode()
    def encode_t5_text(self, text: list[str]) -> torch.Tensor:
        assert self.t5_model is not None, 'T5 model is not loaded'
        assert self.t5_tokenizer is not None, 'T5 Tokenizer is not loaded'
        # x: (B, L)
        inputs = self.t5_tokenizer(text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt").to(self.device)
        return self.t5_model(**inputs).last_hidden_state

    @torch.inference_mode()
    def encode_audio(self, x) -> torch.Tensor:
        x = self.vae.encode(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
