import os

os.environ.setdefault("DISABLE_XFORMERS", "1")

import gc
import logging
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from termcolor import colored
from torchvision import transforms as T
from tqdm import tqdm

# ── Vendor path ──────────────────────────────────────────────────────
# ``sana_wm_diffusion/`` is a vendored subtree in the same directory as
# ``synthesis.py``.  We add it to ``sys.path`` so that its internal imports
# (``import sana_wm_diffusion.xxx``) resolve correctly.
_VENDOR_DIR = str(Path(__file__).resolve().parent)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# Import all Sana-WM model nets to register them with MODELS registry.
import sana_wm_diffusion.model.nets  # noqa: F401
from sana_wm_diffusion import FlowEuler, LTXFlowEuler, DPMS
from sana_wm_diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    vae_decode,
    vae_encode,
)
from openworldlib.base_models.diffusion_model.video.ltx2_vae import get_vae
from sana_wm_diffusion.model.utils import get_weight_dtype
from openworldlib.base_models.diffusion_model.video.ltx2_refiner import (
    DiffusersLTX2Refiner,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from sana_wm_diffusion.utils.cam_utils import compute_raymap, get_pose_inverse
from sana_wm_diffusion.utils.camctrl_config import (
    ModelVideoCamCtrlConfig,
    model_video_camctrl_init_config,
)
from sana_wm_diffusion.utils.chunk_utils import get_chunk_index_from_config
from sana_wm_diffusion.utils.config import AEConfig, SchedulerConfig, TextEncoderConfig
from sana_wm_diffusion.utils.logger import get_root_logger

from ....base_synthesis import BaseSynthesis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_HEIGHT = 704
TARGET_WIDTH = 1280

# Sana-WM's default HuggingFace repo.
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"

DEFAULT_CONFIG_YAML = "config.yaml"
DEFAULT_DIT_WEIGHT = "dit/sana_wm_1600m_720p.safetensors"

# ---------------------------------------------------------------------------
# Helpers (moved from inference_sana_wm.py CLI / config)
# ---------------------------------------------------------------------------


def _resolve_flow_shift(scheduler_cfg: SchedulerConfig, override: float | None) -> float:
    if override is not None:
        return override
    return (
        scheduler_cfg.inference_flow_shift
        if scheduler_cfg.inference_flow_shift is not None
        else scheduler_cfg.flow_shift
    )


# Restore upstream-style upper_bound to avoid snapping beyond trajectory length.
def _snap_num_frames(num_frames: int, vae_time_stride: int = 8, upper_bound: int | None = None) -> int:
    """Snap num_frames to ``8k + 1`` required by LTX-2 VAE.

    Args:
        num_frames: Requested frame count.
        vae_time_stride: VAE temporal stride (default 8 for LTX-2).
        upper_bound: Maximum allowed value (e.g. trajectory length).

    Returns:
        Snapped frame count ``<= upper_bound`` if provided.
    """
    if num_frames < 1:
        return 1
    if (num_frames - 1) % vae_time_stride == 0:
        return num_frames
    floor_cand = num_frames - ((num_frames - 1) % vae_time_stride)
    ceil_cand = floor_cand + vae_time_stride
    snapped = floor_cand if (num_frames - floor_cand) < (ceil_cand - num_frames) else ceil_cand
    if upper_bound is not None and snapped > upper_bound:
        snapped = floor_cand
    return max(snapped, 1)


# ---------------------------------------------------------------------------
# Config shim
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Mini config that matches the structure expected by the Sana-WM pipeline.

    Created from the YAML config shipped with the model weights.
    """

    model: ModelVideoCamCtrlConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    work_dir: str = ""


def load_config_from_yaml(yaml_path: str | Path) -> InferenceConfig:
    """Load an ``InferenceConfig`` from the YAML file bundled with the weights."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    model = ModelVideoCamCtrlConfig(**raw.get("model", {}))
    vae = AEConfig(**raw.get("vae", {}))
    text_encoder = TextEncoderConfig(**raw.get("text_encoder", {}))
    scheduler = SchedulerConfig(**raw.get("scheduler", {}))
    return InferenceConfig(model=model, vae=vae, text_encoder=text_encoder, scheduler=scheduler)


# ---------------------------------------------------------------------------
# Camera condition tensors (from inference_sana_wm.py)
# ---------------------------------------------------------------------------


def _pack_camera_conditions(
    poses: torch.Tensor,
    intrinsics_latent: torch.Tensor,
    num_frames: int,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    vae_time_stride: int,
) -> dict[str, torch.Tensor]:
    """Build raymap + chunk_plucker tensors the model consumes."""
    time_indices = torch.arange(0, num_frames, vae_time_stride)
    if len(time_indices) > latent_frames:
        time_indices = time_indices[:latent_frames]

    raymap = torch.cat(
        [poses[time_indices].reshape(len(time_indices), -1), intrinsics_latent[time_indices]],
        dim=-1,
    )

    chunk_starts = time_indices - (vae_time_stride - 1)
    chunks = []
    for start in chunk_starts:
        s = max(0, int(start))
        e = s + vae_time_stride
        chunk_poses, chunk_intrs = poses[s:e], intrinsics_latent[s:e]
        if chunk_poses.shape[0] < vae_time_stride:
            pad = vae_time_stride - chunk_poses.shape[0]
            chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad, 1, 1)], dim=0)
            chunk_intrs = torch.cat([chunk_intrs, chunk_intrs[-1:].repeat(pad, 1)], dim=0)
        plucker = compute_raymap(chunk_intrs, chunk_poses, latent_h, latent_w, use_plucker=True)
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_h, latent_w))
    chunk_plucker = torch.stack(chunks).permute(1, 0, 2, 3)
    return {"raymap": raymap, "chunk_plucker": chunk_plucker}


def prepare_camera(
    poses_c2w: np.ndarray,
    intrinsics_vec4: np.ndarray,
    *,
    target_size: tuple[int, int],
    vae_stride: tuple[int, int, int] | list[int],
) -> dict[str, torch.Tensor]:
    """Relativise poses to frame 0 and build the model-input tensors."""
    num_frames = poses_c2w.shape[0]
    vae_time_stride, vae_spatial_stride = vae_stride[0], vae_stride[-1]
    H_pixel, W_pixel = target_size
    latent_h = H_pixel // vae_spatial_stride
    latent_w = W_pixel // vae_spatial_stride
    latent_frames = (num_frames - 1) // vae_time_stride + 1

    poses = torch.from_numpy(poses_c2w).float()
    first_inv = get_pose_inverse(poses[0:1]).squeeze(0)
    poses_rel = torch.matmul(first_inv, poses[1:])
    poses = torch.cat([torch.eye(4).unsqueeze(0), poses_rel], dim=0)

    intrinsics = torch.from_numpy(intrinsics_vec4).float()
    intrinsics_latent = intrinsics.clone()
    intrinsics_latent[:, [0, 2]] *= latent_w / float(W_pixel)
    intrinsics_latent[:, [1, 3]] *= latent_h / float(H_pixel)

    return _pack_camera_conditions(
        poses,
        intrinsics_latent,
        num_frames,
        latent_frames,
        latent_h,
        latent_w,
        vae_time_stride,
    )


# ---------------------------------------------------------------------------
# Sana-WM Synthesis
# ---------------------------------------------------------------------------


class SanaWMSynthesis(BaseSynthesis):
    """Sana-WM image-to-video synthesis with camera control.

    Follows the ``BaseSynthesis`` interface (``from_pretrained`` / ``predict``)
    and wraps the Sana-WM pipeline components.

    Typical usage::

        synth = SanaWMSynthesis.from_pretrained("Efficient-Large-Model/SANA-WM_bidirectional")
        video = synth.predict(
            image=my_pil_image,
            prompt="driving forward on a sunny road",
            c2ws=my_c2w_poses,              # (F, 4, 4)
            intrinsics_vec4=my_intrinsics,   # (F, 4)
        )
    """

    def __init__(
        self,
        config: InferenceConfig,
        vae,
        text_encoder: torch.nn.Module,
        tokenizer: Any,
        model: torch.nn.Module,
        device: str = "cuda",
        refiner: Optional[DiffusersLTX2Refiner] = None,
        offload_vae: bool = False,
        offload_refiner: bool = False,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.offload_vae = offload_vae
        self.offload_refiner = offload_refiner

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = model
        self.refiner = refiner

        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = torch.bfloat16  # LTX-2 VAE expects bf16
        self.logger = get_root_logger()

    # -------------------------------------------------------------------
    # from_pretrained
    # -------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = HF_REPO,
        device: str = "cuda",
        text_encoder_path: str | None = None,
        offload_vae: bool = False,
        offload_refiner: bool = False,
        enable_refiner: bool = True,
        refiner_sink_size: int = 1,
        **kwargs,
    ) -> "SanaWMSynthesis":
        """Load Sana-WM from a local directory or HuggingFace repo.

        Args:
            pretrained_model_path: Local path or HuggingFace repo ID
                (default ``Efficient-Large-Model/SANA-WM_bidirectional``).
            device: Target device.
            offload_vae: If True, VAE is kept on CPU and moved to GPU per call.
            offload_refiner: If True, refiner is kept on CPU and moved to GPU per call.
            enable_refiner: If False, skip the LTX-2 refiner (decode with VAE only).
            refiner_sink_size: Sink anchor frames for the refiner.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            Initialised ``SanaWMSynthesis`` instance.
        """
        # ── 1. Resolve model path ──────────────────────────────────────
        if not os.path.isdir(pretrained_model_path):
            print(f"[SanaWMSynthesis] Downloading from HuggingFace: {pretrained_model_path}")
            try:
                pretrained_model_path = snapshot_download(pretrained_model_path)
                print(f"[SanaWMSynthesis] Downloaded to: {pretrained_model_path}")
            except Exception as e:
                raise ValueError(
                    f"'{pretrained_model_path}' is neither a local directory "
                    f"nor a valid HuggingFace repo ID. Error: {e}"
                )

        root = Path(pretrained_model_path)
        device_obj = torch.device(device)

        # ── 2. Load config ─────────────────────────────────────────────
        config_path = root / DEFAULT_CONFIG_YAML
        if not config_path.is_file():
            # Fallback: search for any yaml in root
            yamls = list(root.glob("*.yaml"))
            if yamls:
                config_path = yamls[0]
            else:
                raise FileNotFoundError(f"No config YAML found in {root}")
        config = load_config_from_yaml(str(config_path))
        # Ensure the vae_pretrained path points to the local root
        config.vae.vae_pretrained = str(root)

        print("[SanaWMSynthesis] Loading VAE (LTX-2)...")
        vae = get_vae(
            model_path=str(root),
            device=device_obj,
            dtype=torch.bfloat16,
            tile_sample_stride_num_frames=getattr(config.vae, "tile_sample_stride_num_frames", 64),
            tile_sample_min_num_frames=getattr(config.vae, "tile_sample_min_num_frames", 96),
        )
        if offload_vae:
            vae.to("cpu")
        else:
            vae.to(device_obj)
        vae.eval()

        # ── 4. Build text encoder ──────────────────────────────────────
        print("[SanaWMSynthesis] Loading text encoder (gemma-2-2b-it)...")
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name,
            device=device_obj,
            model_path=text_encoder_path,
        )
        text_encoder.eval()

        # ── 5. Build DiT model ─────────────────────────────────────────
        print("[SanaWMSynthesis] Building DiT model...")
        latent_size = config.model.image_size // config.vae.vae_stride[-1]
        kwargs_model = model_video_camctrl_init_config(config, latent_size=latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False),
            **kwargs_model,
        ).to(device_obj)

        # ── 6. Load DiT state dict ──────────────────────────────────────
        dit_path = root / DEFAULT_DIT_WEIGHT
        if not dit_path.is_file():
            # Search for any .safetensors / .pth under dit/
            dit_candidates = list(root.rglob("*.safetensors")) + list(root.rglob("*.pth"))
            if dit_candidates:
                dit_path = dit_candidates[0]
                print(f"[SanaWMSynthesis] Found DiT weights at: {dit_path}")
            else:
                raise FileNotFoundError(
                    f"No DiT weight found under {root / 'dit'} or {root}"
                )

        if dit_path.suffix == ".safetensors":
            state = safetensors_load_file(str(dit_path), device="cpu")
        else:
            state = torch.load(str(dit_path), map_location="cpu")

        if "generator" in state:
            state = state["generator"]
        if "state_dict" not in state:
            stripped = {
                (k[len("model."):] if k.startswith("model.") else k): v
                for k, v in state.items()
            }
            state = {"state_dict": stripped}
        state["state_dict"].pop("pos_embed", None)
        missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
        if missing:
            print(f"[SanaWMSynthesis] Missing keys: {missing}")
        if unexpected:
            print(f"[SanaWMSynthesis] Unexpected keys: {unexpected}")
        model = model.eval().to(dtype=get_weight_dtype(config.model.mixed_precision))

        # ── 7. Build refiner (optional) ────────────────────────────────
        refiner = None
        if enable_refiner and config.vae.vae_type == "LTX2VAE_diffusers":
            refiner_root = root / "refiner"
            gemma_root = root / "refiner" / "text_encoder"
            if (refiner_root / "transformer" / "config.json").is_file():
                print("[SanaWMSynthesis] Building LTX-2 refiner...")
                from .....base_models.diffusion_model.video.ltx2_refiner import (
                    DiffusersLTX2Refiner,
                )

                refiner = DiffusersLTX2Refiner(
                    refiner_root=str(refiner_root),
                    gemma_root=str(gemma_root),
                    dtype=model.dtype,
                    device=device_obj,
                )
                if offload_refiner:
                    refiner.to("cpu")
            else:
                print("[SanaWMSynthesis] Refiner not found — running VAE-only.")

        return cls(
            config=config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            model=model,
            device=device,
            refiner=refiner,
            offload_vae=offload_vae,
            offload_refiner=offload_refiner,
        )

    # -------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        c2ws: np.ndarray,
        intrinsics_vec4: np.ndarray,
        num_frames: int = 161,
        fps: int = 16,
        step: int = 60,
        cfg_scale: float = 5.0,
        flow_shift: float | None = None,
        seed: int = 42,
        negative_prompt: str = "",
        sampling_algo: str = "flow_euler_ltx",
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a camera-controlled video.

        Args:
            image: Input PIL image (will be resized+cropped to 704×1280).
            prompt: Text prompt.
            c2ws: ``(F, 4, 4)`` camera-to-world matrices.
            intrinsics_vec4: ``(F, 4)`` ``[fx, fy, cx, cy]`` intrinsics.
            num_frames: Number of output frames (will be snapped to ``8k+1``).
            fps: Frames per second (for metadata / refiner).
            step: Number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            flow_shift: Flow-matching shift (default from config).
            seed: Random seed.
            negative_prompt: Negative text prompt.
            sampling_algo: One of ``"flow_euler_ltx"``, ``"flow_euler"``,
                ``"flow_dpm-solver"``.

        Returns:
            Dict with keys:
                - ``video``: ``(T, H, W, 3)`` uint8 numpy array
                - ``c2w``: ``(T', 4, 4)`` aligned c2w trajectory
                - ``latent``: raw Sana latent (CPU tensor)
        """
        # Snap frame count with upper_bound to stay within trajectory length.
        vae_stride = self.config.vae.vae_stride
        max_frames = min(len(c2ws), len(intrinsics_vec4))
        num_frames = _snap_num_frames(num_frames, vae_stride[0], upper_bound=max_frames)

        latent_T = (num_frames - 1) // vae_stride[0] + 1
        latent_h = TARGET_HEIGHT // vae_stride[-1]
        latent_w = TARGET_WIDTH // vae_stride[-1]

        # ── 1. Prepare camera ──────────────────────────────────────────
        camera = prepare_camera(
            c2ws[:num_frames],
            intrinsics_vec4[:num_frames],
            target_size=(TARGET_HEIGHT, TARGET_WIDTH),
            vae_stride=vae_stride,
        )

        # ── 2. Stage 1: Sana DiT sampling ──────────────────────────────
        sana_latent = self._sample_stage1(
            image, prompt, camera, negative_prompt,
            num_frames, latent_T, latent_h, latent_w,
            step, cfg_scale, flow_shift, seed, sampling_algo,
        )

        # ── 3. Decode ──────────────────────────────────────────────────
        if self.refiner is not None:
            video = self._refine(sana_latent, prompt, fps, seed)
            video_c2w = c2ws[1:num_frames]
        else:
            video = self._decode_with_vae(sana_latent)
            video_c2w = c2ws[:num_frames]

        return {"video": video, "c2w": video_c2w, "latent": sana_latent.cpu()}

    # -------------------------------------------------------------------
    # Internal: stage 1
    # -------------------------------------------------------------------

    def _encode_prompts(
        self, prompt: str, negative_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = self.config.text_encoder.model_max_length
        chi_prompt = "\n".join(self.config.text_encoder.chi_prompt or [])
        if chi_prompt:
            prompt = chi_prompt + prompt
            max_length_all = len(self.tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            max_length_all = max_length

        def encode(text: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            tok = self.tokenizer(
                [text], max_length=length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            return self.text_encoder(tok.input_ids, tok.attention_mask)[0], tok.attention_mask

        cond, cond_mask = encode(prompt, max_length_all)
        select = [0] + list(range(-max_length + 1, 0))
        cond = cond[:, None][:, :, select]
        cond_mask = cond_mask[:, select]
        neg, neg_mask = encode(negative_prompt, max_length)
        return cond, cond_mask, neg[:, None], neg_mask

    def _sample_stage1(
        self,
        image: Image.Image,
        prompt: str,
        camera: dict[str, torch.Tensor],
        negative_prompt: str,
        num_frames: int,
        latent_T: int,
        latent_h: int,
        latent_w: int,
        step: int,
        cfg_scale: float,
        flow_shift: float | None,
        seed: int,
        sampling_algo: str,
    ) -> torch.Tensor:
        if self.offload_vae:
            self.vae.to(self.device)

        # Encode first frame
        img = (T.ToTensor()(image) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2).to(self.device)
        first_latent = vae_encode(
            self.config.vae.vae_type,
            self.vae,
            img.to(dtype=self.vae_dtype),
            device=self.device,
        ).to(self.weight_dtype)
        if self.offload_vae:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

        # Encode text
        cond, cond_mask, neg, neg_mask = self._encode_prompts(prompt, negative_prompt)
        raymap = camera["raymap"].unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(self.device, dtype=self.weight_dtype)

        if cfg_scale > 1.0:
            mask_cfg = torch.cat([neg_mask, cond_mask], dim=0)
            raymap_cfg = torch.cat([raymap, raymap], dim=0)
            chunk_plucker_cfg = torch.cat([chunk_plucker, chunk_plucker], dim=0)
        else:
            mask_cfg, raymap_cfg, chunk_plucker_cfg = cond_mask, raymap, chunk_plucker

        # Init noise
        latent_channels = first_latent.shape[1]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(
            1, latent_channels, latent_T, latent_h, latent_w,
            dtype=self.weight_dtype, device=self.device, generator=generator,
        )
        z[:, :, :1] = first_latent

        # Model kwargs
        chunk_index = get_chunk_index_from_config(self.config, num_frames=latent_T)
        model_kwargs = dict(
            data_info={
                "img_hw": torch.tensor(
                    [[TARGET_HEIGHT, TARGET_WIDTH]], dtype=torch.float, device=self.device
                ),
                "condition_frame_info": {0: 0.0},
            },
            mask=mask_cfg,
            camera_conditions=raymap_cfg,
            chunk_plucker=chunk_plucker_cfg,
        )
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index

        flow_shift_val = _resolve_flow_shift(self.config.scheduler, flow_shift)
        samples = self._dispatch_solver(
            sampling_algo, z, cond, neg, cfg_scale, flow_shift_val, step, model_kwargs,
            chunk_index, generator,
        )
        torch.cuda.empty_cache()
        return samples.detach()

    def _dispatch_solver(
        self,
        algo: str,
        z: torch.Tensor,
        cond: torch.Tensor,
        neg: torch.Tensor,
        cfg_scale: float,
        flow_shift: float,
        steps: int,
        model_kwargs: dict,
        chunk_index: Any,
        generator: torch.Generator,
    ) -> torch.Tensor:
        base = dict(
            condition=cond,
            uncondition=neg,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            model_kwargs=model_kwargs,
        )
        if algo == "flow_euler_ltx":
            return LTXFlowEuler(self.model, **base).sample(z, steps=steps, generator=generator)
        if algo == "flow_euler":
            return FlowEuler(self.model, **base).sample(z, steps=steps)
        if algo == "flow_dpm-solver":
            from sana_wm_diffusion import DPMS

            return DPMS(
                self.model,
                condition=cond,
                uncondition=neg,
                cfg_scale=cfg_scale,
                model_type="flow",
                guidance_type="classifier-free",
                model_kwargs=model_kwargs,
                schedule="FLOW",
            ).sample(z, steps=steps, order=2, skip_type="time_uniform_flow", method="multistep",
                     flow_shift=flow_shift)
        raise ValueError(f"Unknown sampling_algo: {algo}")

    # -------------------------------------------------------------------
    # Internal: decode
    # -------------------------------------------------------------------

    def _decode_with_vae(self, sana_latent: torch.Tensor) -> np.ndarray:
        if self.offload_vae:
            self.vae.to(self.device)
        samples = sana_latent.to(device=self.device, dtype=self.vae_dtype)
        decoded = vae_decode(self.config.vae.vae_type, self.vae, samples)
        if isinstance(decoded, list):
            decoded = torch.stack(decoded, dim=0)
        video = (
            torch.clamp(127.5 * decoded + 127.5, 0, 255)
            .permute(0, 2, 3, 4, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()[0]
        )
        if self.offload_vae:
            self.vae.to("cpu")
        torch.cuda.empty_cache()
        return video

    def _refine(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        fps: int,
        seed: int,
    ) -> np.ndarray:
        if self.offload_refiner:
            self.model.to("cpu")
            self.text_encoder.to("cpu")
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            self.refiner.to(self.device)

        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        self.logger.info(f"[refiner] {len(sigmas) - 1}-step Euler")
        refined = self.refiner.refine_latents(
            sana_latent,
            prompt,
            fps=float(fps),
            sink_size=1,
            seed=seed,
            progress=True,
        )
        if self.offload_refiner:
            self.refiner.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        video = self._decode_with_vae(refined)
        # Drop sink anchor frame
        video = video[1:]
        del refined
        torch.cuda.empty_cache()
        gc.collect()
        return video