from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from PIL import Image

from ...operators.infinite_world_operator import InfiniteWorldOperator
from ...memories.visual_synthesis.infinite_world.infinite_world_memory import (
    InfiniteWorldMemory,
)
from ...synthesis.visual_generation.infinite_world.infinite_world_synthesis import (
    DEFAULT_NEGATIVE_PROMPT,
    InfiniteWorldSynthesis,
)


class InfiniteWorldPipeline:
    def __init__(
        self,
        operators: Optional[InfiniteWorldOperator] = None,
        synthesis_model: Optional[InfiniteWorldSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
    ):
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        bucket_config_name: str = "ASPECT_RATIO_627_F64",
        **kwargs,
    ) -> "InfiniteWorldPipeline":
        synthesis_model = InfiniteWorldSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            required_components=required_components,
            device=device,
            weight_dtype=weight_dtype,
            bucket_config_name=bucket_config_name,
            **kwargs,
        )
        operators = InfiniteWorldOperator(bucket_config_name=bucket_config_name)
        memory_module = InfiniteWorldMemory()

        return cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype,
        )

    def _resolve_num_frames(self, interactions: List[str], num_frames: Optional[int]) -> int:
        if num_frames is not None:
            return int(num_frames)
        return max(len(interactions) * 16, 16)

    def process(
        self,
        input_context,
        interactions: List[str],
        num_output_frames: int,
        size: Optional[Tuple[int, int]] = None,
    ):
        perception_dict = self.operators.process_perception(
            input_context,
            size=size,
        )
        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction(num_output_frames + 1)
        self.operators.delete_last_interaction()

        return {
            "visual_context": perception_dict,
            "operator_condition": operator_condition,
        }

    def __call__(
        self,
        images,
        prompt: str = "",
        interactions: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = 5.0,
        num_sampling_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress: bool = True,
        **kwargs,
    ):
        if not isinstance(images, Image.Image):
            raise ValueError("InfiniteWorldPipeline expects a PIL.Image as `images`.")

        interactions = interactions or ["forward"]
        num_output_frames = self._resolve_num_frames(interactions, num_frames)
        output_dict = self.process(
            input_context=images,
            interactions=interactions,
            num_output_frames=num_output_frames,
            size=size,
        )

        return self.synthesis_model.predict(
            cond_video=output_dict["visual_context"]["video"],
            move_ids=output_dict["operator_condition"]["move_ids"],
            view_ids=output_dict["operator_condition"]["view_ids"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_output_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            progress=progress,
            **kwargs,
        )

    def stream(
        self,
        images: Optional[Image.Image],
        interactions: List[str],
        prompt: str = "",
        num_frames: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = 5.0,
        num_sampling_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress: bool = True,
        **kwargs,
    ):
        if self.memory_module is None:
            raise ValueError("memory_module is None")

        interactions = interactions or ["forward"]
        num_output_frames = self._resolve_num_frames(interactions, num_frames)

        if images is not None:
            if self.memory_module.has_frames():
                self.memory_module.manage(action="reset")
            first_turn_context = self.operators.process_perception(images, size=size)
            self.memory_module.record(
                images,
                processed_frames=first_turn_context["frames"],
                target_size=first_turn_context["size"],
            )

        if not self.memory_module.has_frames():
            raise ValueError("No image history in memory. Provide `images` on the first stream() call.")

        if size is not None and self.memory_module.target_size is not None:
            if tuple(size) != tuple(self.memory_module.target_size):
                raise ValueError(
                    f"stream size mismatch: expected {self.memory_module.target_size}, got {size}"
                )

        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No current image in memory. Provide `images` on the first stream() call.")

        output_dict = self.process(
            input_context=current_image,
            interactions=interactions,
            num_output_frames=num_output_frames,
            size=self.memory_module.target_size,
        )

        generated_frames = self.synthesis_model.predict(
            cond_video=output_dict["visual_context"]["video"],
            move_ids=output_dict["operator_condition"]["move_ids"],
            view_ids=output_dict["operator_condition"]["view_ids"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_output_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            progress=progress,
            **kwargs,
        )
        self.memory_module.record(generated_frames)
        return generated_frames
