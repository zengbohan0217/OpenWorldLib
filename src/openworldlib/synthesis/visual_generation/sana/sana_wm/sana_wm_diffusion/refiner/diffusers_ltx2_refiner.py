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
Backward-compatibility shim for sana_wm_diffusion.refiner.diffusers_ltx2_refiner.

All symbols are re-exported from the canonical base_models implementation at:

    openworldlib.base_models.diffusion_model.video.ltx2_refiner

This file intentionally contains NO logic — it only forwards imports.
"""

from openworldlib.base_models.diffusion_model.video.ltx2_refiner import (
    DiffusersLTX2Refiner,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    _empty_cuda_cache,
    _forward_video_block,
    _pack_latents,
    _pack_text_embeds,
    _streaming_self_attention,
    _unpack_latents,
)

__all__ = [
    "DiffusersLTX2Refiner",
    "STAGE_2_DISTILLED_SIGMA_VALUES",
    "_empty_cuda_cache",
    "_forward_video_block",
    "_pack_latents",
    "_pack_text_embeds",
    "_streaming_self_attention",
    "_unpack_latents",
]
