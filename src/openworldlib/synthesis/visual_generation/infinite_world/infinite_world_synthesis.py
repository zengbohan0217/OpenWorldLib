from __future__ import annotations

import datetime
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from ...base_synthesis import BaseSynthesis
from .infworld.context_parallel import context_parallel_util as cp_util
from .infworld.models import dit_model as dit_model_module
from .infworld.models.scheduler import RFlowScheduler
from .infworld.models.umt5 import T5EncoderModel
from .infworld.vae import WanVAEModelWrapper


DEFAULT_REPO_ID = "MeiGen-AI/Infinite-World"
DEFAULT_NEGATIVE_PROMPT = (
    "many cars, crowds, vivid hues, overexposed, static, blurry details, subtitles, "
    "style, work, artwork, image, still, overall grayish, worst quality, low quality, "
    "jpeg compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless "
    "image, cluttered background, three legs, crowded background, walking backwards."
)


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        device = "cuda"
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Infinite-World inference")
        torch.cuda.set_device(resolved.index if resolved.index is not None else 0)
    return resolved


def _configure_runtime(device: torch.device) -> Tuple[int, int]:
    if "RANK" in os.environ:
        if not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", device.index or 0))
            if device.type == "cuda":
                torch.cuda.set_device(local_rank)
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(hours=24),
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if cp_util.get_dp_size() is None or cp_util.get_cp_size() is None:
            from .infworld.context_parallel.context_parallel_util import init_context_parallel

            init_context_parallel(
                context_parallel_size=1,
                global_rank=rank,
                world_size=world_size,
            )
        return rank, world_size

    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    return 0, 1


def _load_state_dict(checkpoint_path: str):
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _find_existing_path(root: str, candidates: List[str]) -> Optional[str]:
    for rel_path in candidates:
        candidate = os.path.join(root, rel_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _resolve_model_root(pretrained_model_path: Optional[str]) -> Tuple[str, Optional[str]]:
    target = pretrained_model_path or DEFAULT_REPO_ID
    if os.path.isfile(target):
        return os.path.abspath(os.path.dirname(target)), os.path.abspath(target)
    if os.path.isdir(target):
        return os.path.abspath(target), None
    downloaded_root = snapshot_download(target)
    return os.path.abspath(downloaded_root), None


def _resolve_components(
    model_root: str,
    checkpoint_override: Optional[str],
    required_components: Optional[Dict[str, str]],
) -> Dict[str, str]:
    required_components = required_components or {}
    components = {
        "checkpoint_path": checkpoint_override
        or required_components.get("checkpoint_path")
        or _find_existing_path(
            model_root,
            [
                "checkpoints/infinite_world_model.ckpt",
                "infinite_world_model.ckpt",
                "checkpoints/diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.safetensors",
            ],
        ),
        "vae_pth": required_components.get("vae_pth")
        or required_components.get("vae_model_path")
        or _find_existing_path(
            model_root,
            [
                "checkpoints/models/Wan2.1_VAE.pth",
                "models/Wan2.1_VAE.pth",
                "Wan2.1_VAE.pth",
            ],
        ),
        "text_encoder_checkpoint_path": required_components.get("text_encoder_checkpoint_path")
        or required_components.get("text_encoder_model_path")
        or _find_existing_path(
            model_root,
            [
                "checkpoints/models/models_t5_umt5-xxl-enc-bf16.pth",
                "models/models_t5_umt5-xxl-enc-bf16.pth",
            ],
        ),
        "tokenizer_path": required_components.get("tokenizer_path")
        or _find_existing_path(
            model_root,
            [
                "checkpoints/models/google/umt5-xxl",
                "models/google/umt5-xxl",
                "google/umt5-xxl",
            ],
        ),
    }

    missing = [name for name, path in components.items() if not path]
    if missing:
        raise FileNotFoundError(
            f"Unable to resolve Infinite-World components: {missing}. "
            f"Checked under '{model_root}' and required_components={required_components}."
        )
    return components


class InfiniteWorldSynthesis(BaseSynthesis):
    def __init__(
        self,
        dit,
        vae,
        text_encoder,
        scheduler,
        device: torch.device,
        weight_dtype,
        rank: int,
        world_size: int,
        validation_num_frames: int = 81,
        bucket_config_name: str = "ASPECT_RATIO_627_F64",
        component_paths: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.device = device
        self.weight_dtype = weight_dtype
        self.rank = rank
        self.world_size = world_size
        self.validation_num_frames = validation_num_frames
        self.bucket_config_name = bucket_config_name
        self.component_paths = component_paths or {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        required_components: Optional[Dict[str, str]] = None,
        device=None,
        weight_dtype=torch.bfloat16,
        seed: int = 42,
        num_sampling_steps: int = 30,
        validation_num_frames: int = 81,
        bucket_config_name: str = "ASPECT_RATIO_627_F64",
        **kwargs,
    ):
        if not (
            getattr(dit_model_module, "FLASH_ATTN_2_AVAILABLE", False)
            or getattr(dit_model_module, "FLASH_ATTN_3_AVAILABLE", False)
        ):
            raise ImportError(
                "Infinite-World requires flash_attn for inference. "
                "Install a compatible flash-attn build before running this pipeline."
            )

        resolved_device = _resolve_device(device)
        rank, world_size = _configure_runtime(resolved_device)
        _seed_everything(seed + rank)

        model_root, checkpoint_override = _resolve_model_root(pretrained_model_path)
        component_paths = _resolve_components(
            model_root=model_root,
            checkpoint_override=checkpoint_override,
            required_components=required_components,
        )

        text_encoder = T5EncoderModel(
            model_max_length=512,
            dtype=weight_dtype,
            device=resolved_device,
            checkpoint_path=component_paths["text_encoder_checkpoint_path"],
            tokenizer_path=component_paths["tokenizer_path"],
        )
        vae = WanVAEModelWrapper(
            vae_pth=component_paths["vae_pth"],
            dtype=torch.float,
            device=str(resolved_device),
        )

        scheduler = RFlowScheduler(
            shift=7.0,
            use_reversed_velocity=True,
            use_timestep_transform=True,
            num_sampling_steps=num_sampling_steps,
            audio_cfg_scale=5.0,
            text_cfg_scale=5.0,
        )
        scheduler.num_sampling_steps = num_sampling_steps

        dit = dit_model_module.WanModel(
            out_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_context_parallel=False,
            model_type="t2v",
            dim=1536,
            in_channels=20,
            ffn_dim=8960,
            freq_dim=256,
            num_heads=12,
            num_layers=30,
        )
        dit = dit.to(device=resolved_device, dtype=weight_dtype)
        dit.eval()
        dit.requires_grad_(False)

        state_dict = _load_state_dict(component_paths["checkpoint_path"])
        state_dict.pop("pos_embed_temporal", None)
        state_dict.pop("pos_embed", None)
        dit.load_state_dict(state_dict, strict=False)

        return cls(
            dit=dit,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            device=resolved_device,
            weight_dtype=weight_dtype,
            rank=rank,
            world_size=world_size,
            validation_num_frames=validation_num_frames,
            bucket_config_name=bucket_config_name,
            component_paths=component_paths,
        )

    def _slice_action_window(self, action_ids: torch.Tensor, start_idx: int) -> torch.Tensor:
        if start_idx >= action_ids.shape[0]:
            return torch.zeros(self.validation_num_frames, dtype=torch.long)

        window = action_ids[start_idx : start_idx + self.validation_num_frames]
        if window.shape[0] < self.validation_num_frames:
            pad = torch.zeros(self.validation_num_frames - window.shape[0], dtype=torch.long)
            window = torch.cat([window, pad], dim=0)
        return window

    def _latent_output_frames(self, height: int, width: int) -> int:
        latent_size = self.vae.get_latent_size((self.validation_num_frames, height, width))
        return int(latent_size[0])

    def _tensor_to_frames(self, video: torch.Tensor) -> List[np.ndarray]:
        video = video.clamp(-1, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()
        video = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return [np.ascontiguousarray(frame) for frame in video]

    @torch.no_grad()
    def predict(
        self,
        cond_video: torch.Tensor,
        move_ids: torch.Tensor,
        view_ids: torch.Tensor,
        prompt: str = "",
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_output_frames: int = 80,
        guidance_scale: float = 5.0,
        num_sampling_steps: Optional[int] = None,
        shift: Optional[float] = None,
        seed: Optional[int] = None,
        progress: bool = True,
    ):
        if cond_video.ndim != 5:
            raise ValueError(f"cond_video must be [B, C, T, H, W], got {cond_video.shape}")
        if cond_video.shape[0] != 1:
            raise ValueError("Infinite-World synthesis currently supports batch size 1")
        if num_output_frames <= 0:
            return []

        if seed is not None:
            _seed_everything(seed + self.rank)
        if num_sampling_steps is not None:
            self.scheduler.num_sampling_steps = num_sampling_steps
        if shift is not None:
            self.scheduler.shift = shift

        video_buffer = cond_video.detach().cpu()
        move_ids = move_ids.detach().long().cpu()
        view_ids = view_ids.detach().long().cpu()

        generated_segments: List[torch.Tensor] = []
        generated_frames = 0

        while generated_frames < num_output_frames:
            current_cond = video_buffer.to(self.device)
            current_latent = self.vae.encode(current_cond)

            latent_size = list(current_latent.shape)
            latent_size[2] = self._latent_output_frames(
                height=current_cond.shape[-2],
                width=current_cond.shape[-1],
            )

            # Action ids describe only the current turn, so advance by emitted frames.
            start_idx = generated_frames
            move_window = self._slice_action_window(move_ids, start_idx)
            view_window = self._slice_action_window(view_ids, start_idx)

            samples = self.scheduler.sample(
                model=self.dit,
                text_encoder=self.text_encoder,
                null_embedder=self.dit.y_embedder,
                z_size=torch.Size(latent_size),
                prompts=[prompt],
                guidance_scale=guidance_scale,
                negative_prompts=[negative_prompt],
                device=self.device,
                additional_args={
                    "image_cond": current_latent,
                    "move": move_window.unsqueeze(0).to(self.device),
                    "view": view_window.unsqueeze(0).to(self.device),
                },
                progress=progress,
            )

            decoded_chunk = self.vae.decode(samples).cpu()
            new_frames = decoded_chunk[:, :, 1:, :, :]
            remaining = num_output_frames - generated_frames
            if new_frames.shape[2] > remaining:
                new_frames = new_frames[:, :, :remaining, :, :]

            video_buffer = torch.cat([video_buffer, new_frames], dim=2)
            generated_segments.append(new_frames)
            generated_frames += new_frames.shape[2]

        generated_video = torch.cat(generated_segments, dim=2)[0]
        return self._tensor_to_frames(generated_video)
