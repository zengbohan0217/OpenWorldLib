from pathlib import Path
from typing import Any

import torch

from .base_operator import BaseOperator
from ..reasoning.spatial_reasoning.cambrian_s.constants import DEFAULT_SYSTEM_PROMPT
from ..reasoning.spatial_reasoning.cambrian_s.conversation import (
    build_qwen_chat_prompt,
    extract_media_inputs,
)
from ..reasoning.spatial_reasoning.cambrian_s.mm_utils import (
    preprocess_single_image,
    preprocess_video_frames,
    tokenizer_image_token,
)


class CambrianSOperator(BaseOperator):
    """
    Lightweight operator placeholder for Cambrian-S.
    It tracks interactions and converts OpenWorldLib chat messages into
    Cambrian-S prompt tokens plus image/video tensors.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        super().__init__(operation_types=operation_types or ["reasoning"])
        self.interaction_template = interaction_template or []
        self.interaction_template_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "CambrianSOperator":
        return cls()

    def check_interaction(self, interaction):
        return True

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def process_interaction(self, *args, **kwargs):
        return self.current_interaction

    def process_perception(
        self,
        messages: list[dict[str, Any]],
        tokenizer,
        image_processors: list[Any],
        model_config: Any = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        video_max_frames: int | None = None,
    ) -> dict[str, Any]:
        prompt = build_qwen_chat_prompt(messages, system_prompt=system_prompt)
        media_inputs = extract_media_inputs(messages)

        if media_inputs and not image_processors:
            raise RuntimeError("Cambrian-S received visual inputs, but no image processor is available.")

        image_tensors = []
        image_sizes = []
        if media_inputs:
            processor = image_processors[0]
            image_count = sum(1 for media in media_inputs if media.get("type") == "image")
            image_aspect_ratio = getattr(model_config, "image_aspect_ratio", "pad")
            anyres_max_subimages = int(getattr(model_config, "anyres_max_subimages", 1))
            for media in media_inputs:
                media_type = media.get("type")
                if media_type == "image":
                    use_anyres = image_count == 1 and image_aspect_ratio == "anyres"
                    pixel_values, original_size = preprocess_single_image(
                        media["image"],
                        processor,
                        image_aspect_ratio="anyres" if use_anyres else "pad",
                        anyres_max_subimages=anyres_max_subimages,
                    )
                elif media_type == "video":
                    video_input = media["video"]
                    num_threads = -1
                    resolved_video_max_frames = (
                        video_max_frames
                        if video_max_frames is not None
                        else int(getattr(model_config, "video_max_frames", 32))
                    )
                    if isinstance(video_input, (str, Path)):
                        video_name = str(video_input)
                        if "Ego4D" in video_name or "video_mmmu" in video_name:
                            num_threads = 1
                    pixel_values, original_size = preprocess_video_frames(
                        video_input,
                        processor,
                        max_frames=resolved_video_max_frames,
                        model_config=model_config,
                        num_threads=num_threads,
                    )
                else:
                    continue
                image_tensors.append(pixel_values)
                image_sizes.append(original_size)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image_tensors,
            "image_sizes": image_sizes,
        }

    def delete_last_interaction(self):
        super().delete_last_interaction()
