# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Video Autoencoder (VAE).

Wraps ``diffusers.models.autoencoders.AutoencoderKLLTX2Video`` with the
encode/decode conventions used by Sana-WM and other LTX-2 based models.

Usage::

    vae = get_vae(model_path="/path/to/SANA-WM_bidirectional", device="cuda")
    latent = vae_encode("LTX2VAE_diffusers", vae, images, device="cuda")
    video  = vae_decode("LTX2VAE_diffusers", vae, latent)
"""

from __future__ import annotations

import torch
from diffusers.models.autoencoders import AutoencoderKLLTX2Video
from termcolor import colored


def get_vae(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    subfolder: str = "vae",
    **kwargs,
) -> AutoencoderKLLTX2Video:
    """Load an LTX-2 VAE from a HuggingFace repo or local directory.

    Args:
        model_path: Root of the model (e.g. ``Efficient-Large-Model/SANA-WM_bidirectional``
            or a local ``.cache/`` directory).
        device: Target device.
        dtype: Data type to load the VAE in. Defaults to ``bfloat16``.
        subfolder: Subfolder within ``model_path`` that contains the VAE weights.
            Defaults to ``"vae"``.
        **kwargs: Additional arguments passed to ``AutoencoderKLLTX2Video.from_pretrained``.

    Returns:
        Initialised ``AutoencoderKLLTX2Video`` in eval mode on ``device``.
    """
    if dtype is None:
        dtype = torch.bfloat16

    print(colored(f"[LTX2VAE] Loading from {model_path}", attrs=["bold"]))

    vae = AutoencoderKLLTX2Video.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=dtype,
        **kwargs,
    ).to(device).eval()

    # Enable tiling for long-video decode.
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    # Enable framewise encoding/decoding (required for correct temporal behaviour).
    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = True
        vae.use_framewise_decoding = True
        vae.tile_sample_stride_num_frames = kwargs.get("tile_sample_stride_num_frames", 64)
        vae.tile_sample_min_num_frames = kwargs.get("tile_sample_min_num_frames", 96)

    return vae


@torch.no_grad()
def vae_encode(
    vae: AutoencoderKLLTX2Video,
    images: torch.Tensor,
    device: str = "cuda",
    scaling_factor: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Encode RGB images ``(B, C, T, H, W)`` → latent ``(B, C', T', H', W')``.

    Applies LTX-2's latent-space normalisation::

        z = (z_raw - latents_mean) * scaling_factor / latents_std

    Args:
        vae: Initialised ``AutoencoderKLLTX2Video``.
        images: ``(B, C, T, H, W)`` tensor in ``[-1, 1]``.
        device: Device to run on.
        scaling_factor: Override the VAE's ``scaling_factor`` config value.
            If ``None`` the value from ``vae.config.scaling_factor`` is used.
        **kwargs: Reserved for future extension.

    Returns:
        ``(B, C', T', H', W')`` latent tensor.
    """
    dtype = vae.dtype
    posterior = vae.encode(images.to(device=vae.device, dtype=vae.dtype)).latent_dist
    z = posterior.mode()

    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
    sf = scaling_factor if scaling_factor is not None else vae.config.scaling_factor
    z = (z - latents_mean) * sf / latents_std
    return z.to(dtype=dtype)


@torch.no_grad()
def vae_decode(
    vae: AutoencoderKLLTX2Video,
    latent: torch.Tensor,
    scaling_factor: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Decode latent ``(B, C', T', H', W')`` → RGB video ``(B, C, T, H, W)``.

    Applies the inverse of ``vae_encode`` normalisation::

        latent_denorm = latent * latents_std / scaling_factor + latents_mean

    Args:
        vae: Initialised ``AutoencoderKLLTX2Video``.
        latent: ``(B, C', T', H', W')`` latent tensor.
        scaling_factor: Override the VAE's ``scaling_factor`` config value.
            If ``None`` the value from ``vae.config.scaling_factor`` is used.
        **kwargs: Passed through to ``vae.decode``.

    Returns:
        ``(B, C, T, H, W)`` tensor in ``[-1, 1]``; clamp to ``[-1, 1]``.
    """
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
    sf = scaling_factor if scaling_factor is not None else vae.config.scaling_factor
    latent_denorm = latent * latents_std / sf + latents_mean
    latent_denorm = latent_denorm.to(vae.dtype)

    samples = vae.decode(latent_denorm, temb=None, return_dict=False, **kwargs)[0]
    # Ensure output stays in [-1, 1].
    samples = torch.clamp(samples, -1.0, 1.0)
    return samples
