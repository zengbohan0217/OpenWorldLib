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

import os

import torch
from mmcv import Registry
from termcolor import colored
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5Tokenizer,
)
from transformers import logging as transformers_logging

from .utils import set_fp32_attention, set_grad_checkpoint

MODELS = Registry("models")

transformers_logging.set_verbosity_error()


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(name="T5", device="cuda", model_path=None):
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
    }
    # If a local model_path is provided and is a directory, use it directly
    # instead of the HuggingFace repo name (for offline environments).
    if model_path is not None and os.path.isdir(model_path):
        pass  # model_path will be used below
    else:
        assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_dict[name])
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_dict[name], torch_dtype=torch.float16).to(device)
    elif "gemma" in name:
        load_path = model_path if (model_path is not None and os.path.isdir(model_path)) else text_encoder_dict[name]
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    else:
        raise ValueError(f"Unsupported text encoder: {name}")

    return tokenizer, text_encoder


def get_image_encoder(name, model_path, tokenizer_path=None, device="cuda", dtype=None, config=None):
    if name == "CLIP":
        raise NotImplementedError("CLIP image encoder not available in cropped vendor; use flux-siglip instead.")
    elif name == "flux-siglip":
        image_encoder = SiglipVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=dtype).to(
            device
        )
        image_processor = SiglipImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
        return image_encoder.eval().requires_grad_(False), image_processor
    else:
        raise ValueError(f"Unsupported image encoder: {name}")


@torch.no_grad()
def encode_image(name, image_encoder, images, device="cuda", image_processor=None, dtype=None):
    if image_encoder is None:
        return None
    if name == "flux-siglip":
        dtype = dtype or image_encoder.dtype
        images = (images + 1) / 2.0  # [-1, 1] -> [0, 1]
        images = image_processor(images=images.clamp(0, 1), return_tensors="pt", do_rescale=False).to(
            device=device, dtype=image_encoder.dtype
        )
        image_embeds = image_encoder(**images).last_hidden_state
        return image_embeds.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported image encoder: {name}")


def get_vae(name, model_path, device="cuda", dtype=None, config=None, **kwargs):
    if "LTX2VAE_diffusers" in name:
        from .......base_models.diffusion_model.video.ltx2_vae import get_vae as _get
        assert config is not None, "config.vae is required for LTX2VAE_diffusers"
        return _get(
            model_path=config.vae_pretrained,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported VAE type for Sana-WM cropped vendor: {name}")


def vae_encode(name, vae, images, device="cuda", **kwargs):
    if "LTX2VAE_diffusers" in name:
        from .......base_models.diffusion_model.video.ltx2_vae import vae_encode as _encode
        return _encode(vae, images, device=device, **kwargs)
    else:
        raise ValueError(f"Unsupported VAE type for Sana-WM cropped vendor: {name}")


def vae_decode(name, vae, latent, **kwargs):
    if "LTX2VAE_diffusers" in name:
        from .......base_models.diffusion_model.video.ltx2_vae import vae_decode as _decode
        return _decode(vae, latent, **kwargs)
    else:
        raise ValueError(f"Unsupported VAE type for Sana-WM cropped vendor: {name}")
