from typing import List, Optional, Sequence, Union
from PIL import Image as PILImage

from ...reasoning.spatial_reasoning.spatial_ladder.spatial_ladder_reasoning import (
    SpatialLadderReasoning,
)
from ...operators.spatial_ladder_operator import SpatialLadderOperator


class SpatialLadderPipeline:
    """
    Pipeline that builds vision/text inputs and calls SpatialLadderReasoning directly.
    """

    def __init__(self, reasoning: SpatialLadderReasoning, operator: SpatialLadderOperator):
        self.reasoning = reasoning
        self.operator = operator
        self.processor = reasoning.processor

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "hongxingli/SpatialLadder-3B",
        device: Optional[Union[str, "torch.device"]] = None,
        weight_dtype: "torch.dtype" = None,
        **kwargs,
    ) -> "SpatialLadderPipeline":
        """
        Args:
            model_path: HuggingFace model ID or local path to the model.
            device: Target device to load all models onto (e.g. "cuda", "cpu").
            weight_dtype: Weight dtype for the model (e.g. torch.bfloat16, torch.float16).
        """
        import torch
        if weight_dtype is None:
            weight_dtype = torch.bfloat16
        reasoning = SpatialLadderReasoning.from_pretrained(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        operator = SpatialLadderOperator.from_pretrained()
        return cls(reasoning=reasoning, operator=operator)

    def _build_messages(
        self,
        images: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]],
        videos: Optional[Union[str, List[PILImage.Image], Sequence[Union[str, List[PILImage.Image]]]]],
        prompt: str,
    ):
        if images is None:
            images = []
        if videos is None:
            videos = []
        # Wrap a single image (str or PIL.Image) into a list
        if isinstance(images, (str, PILImage.Image)):
            images = [images]
        # Wrap a single video path (str) into a list;
        # a list[PIL.Image] is treated as the frame sequence of one video
        if isinstance(videos, str):
            videos = [videos]
        elif isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], PILImage.Image):
            videos = [videos]

        content = [{"type": "image", "image": img} for img in images]
        content += [{"type": "video", "video": vid} for vid in videos]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]] = None,
        videos: Optional[Union[str, List[PILImage.Image], Sequence[Union[str, List[PILImage.Image]]]]] = None,
        max_new_tokens: int = 2048,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        # Record interaction for interface consistency
        self.operator.get_interaction(prompt)
        self.operator.process_interaction()

        if messages is None:
            batched_messages = [
                self._build_messages(
                    images=images,
                    videos=videos,
                    prompt=prompt,
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batched_messages
        ]

        inputs = self.operator.process_perception(batched_messages, texts, processor=self.processor)

        outputs = self.reasoning.inference(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        self.operator.delete_last_interaction()
        return outputs
