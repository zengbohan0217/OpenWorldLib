# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention, CrossAttentionRope, MemEffCrossAttentionRope, FlashAttentionRope
from ......base_models.perception_core.general_perception.dinov2.layers.drop_path import DropPath
from ......base_models.perception_core.general_perception.dinov2.layers.layer_scale import LayerScale
from ......base_models.perception_core.general_perception.dinov2.layers.mlp import Mlp
from ...pi3.layers.block import (
    Block, drop_add_residual_stochastic_depth, get_branges_scales, add_residual, get_attn_bias_and_cat, drop_add_residual_stochastic_depth_list, NestedTensorBlock, CrossBlockRope, PoseInjectBlock, CrossOnlyBlockRope
)

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        # warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Block)")


attn_bias_cache: Dict[Tuple, Any] = {}

class BlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool=False,
        rope=None
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def compute_kv_cache(self, x: Tensor, xpos=None) -> tuple[Tensor, Tensor]:
        """Compute K, V for caching from input x.
        
        Args:
            x: Input tensor [B, N, C]
            xpos: Position info for RoPE
            
        Returns:
            (k, v): Cached K and V tensors, each [B, num_heads, N, head_dim]
        """
        x_normed = self.norm1(x)
        return self.attn.compute_kv(x_normed, xpos=xpos)
    
    def forward_with_kv_cache(
        self, 
        x: Tensor, 
        k_cache: Tensor, 
        v_cache: Tensor,
        xpos=None,
        attn_mask=None
    ) -> Tensor:
        """Forward with pre-computed KV cache for history tokens.
        
        Args:
            x: Current tokens [B, N_curr, C]
            k_cache: Cached K from history [B, num_heads, N_hist, head_dim]
            v_cache: Cached V from history [B, num_heads, N_hist, head_dim]
            xpos: Position info for current tokens
            attn_mask: Optional attention mask
            
        Returns:
            Output for current tokens [B, N_curr, C]
        """
        # Attention with KV cache
        x_normed = self.norm1(x)
        attn_out = self.attn.forward_with_kv_cache(
            x_normed, k_cache, v_cache, xpos=xpos, attn_mask=attn_mask
        )
        x = x + self.ls1(attn_out)
        
        # MLP (only on current tokens)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    def forward(self, x: Tensor, xpos=None, attn_mask=None) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), xpos=xpos, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x
