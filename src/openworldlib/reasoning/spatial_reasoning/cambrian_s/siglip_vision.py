from __future__ import annotations

from dataclasses import dataclass
from functools import partial, reduce
from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput


class SigLipImageProcessor:
    def __init__(
        self,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        size: tuple[int, int] = (384, 384),
        crop_size: Dict[str, int] | None = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        self.crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format

    def preprocess(self, images, return_tensors: str):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            images = [to_numpy_array(image) for image in images]

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]
        images = reduce(lambda value, fn: [*map(fn, value)], transforms, images)
        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size: int = 1152,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 384,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_mean = image_mean
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "SigLipVisionConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]
        return cls.from_dict(config_dict, **kwargs)


@dataclass
class SigLipVisionModelOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)


class SigLipAttention(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attention_weights = nn.functional.dropout(attention_weights, p=self.dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, value_states)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, query_len, self.embed_dim)
        attention_output = self.out_proj(attention_output)
        if not output_attentions:
            attention_weights = None
        return attention_output, attention_weights


class SigLipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights


class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states, attn_weights = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(value for value in [hidden_states, encoder_states, all_attentions] if value is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs[0])
        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipPreTrainedModel(PreTrainedModel):
    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        return None


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)
        self.vision_model = SigLipVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower_name: str, delay_load: bool = True):
        super().__init__()
        self.vision_tower_name = vision_tower_name.split("-interp")[0]
        self.image_processor = SigLipImageProcessor()
        self.config = SigLipVisionConfig()
        self.is_loaded = False
        self.vision_tower: SigLipVisionModel | None = None
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None) -> None:
        if self.is_loaded:
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
        )
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def to(self, device, dtype: torch.dtype | None = None):
        if not self.is_loaded:
            self.load_model()
        if self.vision_tower is None:
            raise RuntimeError("SigLip vision tower is not loaded.")

        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.vision_tower = self.vision_tower.to(**kwargs)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded or self.vision_tower is None:
            raise RuntimeError("SigLip vision tower is not loaded.")

        outputs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        image_features = outputs.hidden_states[-1].to(images.dtype)
        if image_features.shape[-2] != self.num_patches:
            raise RuntimeError(
                "Unexpected SigLIP token count: "
                f"expected {self.num_patches}, found {image_features.shape[-2]}."
            )
        return image_features

    @property
    def dtype(self) -> torch.dtype:
        if self.vision_tower is None:
            return torch.float32
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self) -> torch.device:
        if self.vision_tower is None:
            return torch.device("cpu")
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_patches(self) -> int:
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self) -> int:
        return self.config.image_size


__all__ = [
    "SigLipImageProcessor",
    "SigLipVisionConfig",
    "SigLipVisionModel",
    "SigLipVisionTower",
]
