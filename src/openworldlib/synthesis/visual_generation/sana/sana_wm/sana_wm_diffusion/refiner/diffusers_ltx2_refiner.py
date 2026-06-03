# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Diffusers-backed LTX-2 refiner used by Sana-WM inference.

The Sana-WM refiner checkpoint is a standard LTX-2 transformer plus text
connectors. Diffusers already owns those modules, but its public transformer
forward always runs the audio stream and does not expose the streaming
sink/current video self-attention mask that this refiner was trained with.

This wrapper keeps the custom surface narrow: load diffusers components, encode
the prompt through Gemma + ``LTX2TextConnectors``, and run a video-only forward
through the diffusers transformer blocks. The only local attention code is the
streaming sink/current split, implemented with diffusers attention modules
without materializing the full sequence-by-sequence mask.
"""

# codeflicker-refactor: ltx-base-5/2vvzxiatcoizx40gv1il
# Refiner now imports from base_models for cross-model reuse.
from openworldlib.base_models.diffusion_model.video.ltx2_refiner import DiffusersLTX2Refiner
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
    ) -> torch.Tensor:
        """Run the 3-step LTX-2 refiner and return refined VAE latents."""
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}.")

        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt)

        self.transformer.to(self.device)
        z = sana_latent.to(device=self.device, dtype=self.dtype)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas[0])

        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        eps = torch.randn(current.shape, generator=generator, device=self.device, dtype=self.dtype)
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        iterator = range(len(sigmas) - 1)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="refiner", unit="step")

        for step_index in iterator:
            sigma = sigmas[step_index]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
            )
            noisy_tokens = _pack_latents(
# Re-export all public symbols from base_models for backward compatibility.
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
