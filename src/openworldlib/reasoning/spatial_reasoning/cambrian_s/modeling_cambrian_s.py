from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from .constants import IMAGE_TOKEN_INDEX
from .mm_utils import (
    SigLipVisionTower,
    resize_patch_grid,
    unpad_image,
    validate_cambrian_s_environment,
)


def _build_vision_projector(config):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        modules: list[nn.Module] = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f"Unsupported Cambrian-S projector type: {projector_type}")


class CambrianSConfig(Qwen2Config):
    model_type = "cambrian_qwen"


class CambrianSModel(Qwen2Model):
    config_class = CambrianSConfig

    def __init__(self, config: CambrianSConfig):
        super().__init__(config)

        self.mm_projector = _build_vision_projector(config)
        embed_std = 1 / math.sqrt(config.hidden_size)
        self.image_newline = nn.Parameter(
            torch.randn(config.hidden_size, dtype=self.embed_tokens.weight.dtype) * embed_std
        )
        vision_tower_names = list(
            getattr(config, "mm_vision_tower_aux_list", None)
            or getattr(config, "vision_tower_aux_list", None)
            or []
        )
        self.vision_tower_aux_list = [
            SigLipVisionTower(vision_tower_name, delay_load=True)
            for vision_tower_name in vision_tower_names
        ]

    def get_vision_tower_aux_list(self) -> list[SigLipVisionTower]:
        return self.vision_tower_aux_list

    def load_vision_towers(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        for tower in self.vision_tower_aux_list:
            tower.load_model()
            tower.to(device=device, dtype=dtype or self.embed_tokens.weight.dtype)

    def _project_features(self, features: torch.Tensor) -> torch.Tensor:
        projector_dtype = next(self.mm_projector.parameters()).dtype
        projected = self.mm_projector(features.to(projector_dtype))
        return projected.to(self.embed_tokens.weight.dtype)

    def _use_image_newline_token(self) -> bool:
        return not hasattr(self.config, "mm_use_im_newline_token") or bool(
            self.config.mm_use_im_newline_token
        )

    def _append_newline_token(self, features: torch.Tensor) -> torch.Tensor:
        if not self._use_image_newline_token():
            return features
        hidden_size = features.shape[-1]
        newline = self.image_newline.to(features.dtype)[None, None, None, :].expand(
            features.shape[0],
            features.shape[1],
            1,
            hidden_size,
        )
        return torch.cat([features, newline], dim=2)

    def _format_image_features(
        self,
        image_features: torch.Tensor,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        target_side = int(getattr(self.config, "si_token_len", image_features.shape[1]) ** 0.5)
        image_features = resize_patch_grid(image_features, target_side)
        batch_size, _, hidden_size = image_features.shape
        grid = image_features.view(batch_size, target_side, target_side, hidden_size)
        grid = unpad_image(grid, original_size)
        return self._append_newline_token(grid).flatten(1, 2)

    def _format_anyres_image_features(
        self,
        image_features: torch.Tensor,
        image_size: tuple[int, int, int, int],
    ) -> torch.Tensor:
        target_side = int(getattr(self.config, "si_token_len", image_features.shape[1]) ** 0.5)
        image_features = resize_patch_grid(image_features, target_side)
        _, _, hidden_size = image_features.shape
        grid = image_features.view(image_features.shape[0], target_side, target_side, hidden_size)

        snapshot_features = grid[0].unsqueeze(0)
        patch_rows, patch_cols = image_size[2:]
        anyres_features = grid[1:].unflatten(0, (patch_rows, patch_cols))
        anyres_features = anyres_features.permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1).unsqueeze(0)

        original_size = image_size[:2]
        snapshot_features = unpad_image(snapshot_features, original_size)
        anyres_features = unpad_image(anyres_features, original_size)

        snapshot_features = self._append_newline_token(snapshot_features).flatten(1, 2)
        anyres_features = self._append_newline_token(anyres_features).flatten(1, 2)
        return torch.cat([snapshot_features, anyres_features], dim=1)

    def _format_video_features(
        self,
        video_features: torch.Tensor,
        video_size: tuple[int, int, int],
    ) -> torch.Tensor:
        target_side = int(max(getattr(self.config, "miv_token_len", video_features.shape[1]), 1) ** 0.5)
        video_features = resize_patch_grid(video_features, target_side)
        _, _, hidden_size = video_features.shape
        grid = video_features.view(video_features.shape[0], target_side, target_side, hidden_size)
        grid = unpad_image(grid, video_size[:2])
        return self._append_newline_token(grid).flatten(1, 2).flatten(0, 1).unsqueeze(0)

    def prepare_media_embeddings(
        self,
        images: list[torch.Tensor],
        image_sizes: list[tuple[int, ...]],
    ) -> list[torch.Tensor]:
        if not images:
            return []
        if not self.vision_tower_aux_list:
            raise RuntimeError("Cambrian-S model does not define a vision tower.")
        if len(images) != len(image_sizes):
            raise ValueError(
                "Cambrian-S expected image tensors and image_sizes to align, "
                f"but received {len(images)} tensors and {len(image_sizes)} size entries."
            )

        tower = self.vision_tower_aux_list[0]
        media_counts = [media.shape[0] for media in images]
        stacked_images = torch.cat(images, dim=0).to(device=tower.device, dtype=tower.dtype)
        vision_features = tower(stacked_images)
        projected_features = self._project_features(vision_features)
        split_features = list(torch.split(projected_features, media_counts, dim=0))

        media_embeddings: list[torch.Tensor] = []
        for features, media_size in zip(split_features, image_sizes):
            if len(media_size) == 2:
                media_embeddings.append(self._format_image_features(features, media_size))
            elif len(media_size) == 3:
                media_embeddings.append(self._format_video_features(features, media_size))
            elif len(media_size) == 4:
                media_embeddings.append(self._format_anyres_image_features(features, media_size))
            else:
                raise ValueError(f"Unsupported Cambrian-S media size: {media_size}")
        return media_embeddings


class CambrianSForCausalLM(Qwen2ForCausalLM):
    config_class = CambrianSConfig

    def __init__(self, config: CambrianSConfig):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        config.rope_scaling = None

        self.model = CambrianSModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self) -> CambrianSModel:
        return self.model

    def load_vision_towers(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.get_model().load_vision_towers(device=device, dtype=dtype)

    def prepare_inputs_labels_for_multimodal_for_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
        labels: torch.Tensor | None,
        images: list[torch.Tensor] | None,
        image_sizes: list[tuple[int, ...]] | None = None,
    ):
        if not images or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if input_ids.shape[0] != 1:
            raise ValueError("Cambrian-S v1 only supports single-sample generation.")

        media_embeddings = self.get_model().prepare_media_embeddings(images, image_sizes or [])

        input_ids_for_embed = torch.where(input_ids == IMAGE_TOKEN_INDEX, 0, input_ids)
        input_embeds = self.get_model().embed_tokens(input_ids_for_embed)
        image_positions = input_ids[0].eq(IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()

        if len(image_positions) != len(media_embeddings):
            raise ValueError(
                "Cambrian-S found a mismatch between visual placeholders and prepared media embeddings: "
                f"{len(image_positions)} placeholders vs {len(media_embeddings)} media items."
            )

        pieces: list[torch.Tensor] = []
        start = 0
        for image_position, media_embedding in zip(image_positions, media_embeddings):
            pieces.append(input_embeds[:, start:image_position])
            pieces.append(media_embedding.to(dtype=input_embeds.dtype, device=input_embeds.device))
            start = image_position + 1
        pieces.append(input_embeds[:, start:])

        new_input_embeds = torch.cat(pieces, dim=1)
        attention_mask = torch.ones(
            new_input_embeds.shape[:2],
            device=new_input_embeds.device,
            dtype=torch.bool,
        )
        return None, None, attention_mask, past_key_values, new_input_embeds, labels

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        images: list[torch.Tensor] | None = None,
        image_sizes: list[tuple[int, ...]] | None = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("Cambrian-S does not accept external inputs_embeds.")

        if images:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal_for_generation(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            if inputs is None:
                raise ValueError("Cambrian-S generate requires token inputs when no images are provided.")
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        prepared = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            prepared["images"] = images
        if image_sizes is not None:
            prepared["image_sizes"] = image_sizes
        return prepared


CambrianQwenConfig = CambrianSConfig
CambrianQwenForCausalLM = CambrianSForCausalLM


def register_cambrian_s_autoclasses() -> None:
    try:
        AutoConfig.register("cambrian_qwen", CambrianSConfig)
    except ValueError:
        pass

    try:
        AutoModelForCausalLM.register(CambrianSConfig, CambrianSForCausalLM)
    except ValueError:
        pass


register_cambrian_s_autoclasses()

__all__ = [
    "CambrianSConfig",
    "CambrianSForCausalLM",
    "CambrianSModel",
    "CambrianQwenConfig",
    "CambrianQwenForCausalLM",
    "register_cambrian_s_autoclasses",
    "validate_cambrian_s_environment",
]
