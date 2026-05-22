from typing import List, Optional, Sequence, Union

from PIL import Image as PILImage

from ...operators.cambrian_s_operator import CambrianSOperator
from ...reasoning.spatial_reasoning.cambrian_s.cambrian_s_reasoning import CambrianSReasoning


class CambrianSPipeline:
    """
    Pipeline that builds Cambrian-S multimodal inputs and runs Cambrian-S reasoning.
    """

    def __init__(self, reasoning: CambrianSReasoning, operator: CambrianSOperator):
        self.reasoning = reasoning
        self.operator = operator

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "nyu-visionx/Cambrian-S-7B",
        device: Optional[Union[str, "torch.device"]] = None,
        weight_dtype: "torch.dtype" = None,
        **kwargs,
    ) -> "CambrianSPipeline":
        reasoning = CambrianSReasoning.from_pretrained(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        operator = CambrianSOperator.from_pretrained()
        return cls(reasoning=reasoning, operator=operator)

    def _build_messages(
        self,
        images: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]],
        videos: Optional[
            Union[
                str,
                list[PILImage.Image],
                Sequence[Union[str, list[PILImage.Image]]],
            ]
        ],
        prompt: str,
    ):
        if images is None:
            images = []
        if videos is None:
            videos = []

        if isinstance(images, (str, PILImage.Image)):
            images = [images]
        if isinstance(videos, str):
            videos = [videos]
        elif isinstance(videos, list) and videos and isinstance(videos[0], PILImage.Image):
            videos = [videos]

        content = [{"type": "image", "image": image} for image in images]
        content += [{"type": "video", "video": video} for video in videos]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[str, PILImage.Image, Sequence[Union[str, PILImage.Image]]]] = None,
        videos: Optional[
            Union[
                str,
                list[PILImage.Image],
                Sequence[Union[str, list[PILImage.Image]]],
            ]
        ] = None,
        max_new_tokens: int = 2048,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        self.operator.get_interaction(prompt)
        self.operator.process_interaction()

        if messages is None:
            batched_messages = [self._build_messages(images=images, videos=videos, prompt=prompt)]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        outputs: List[str] = []
        for sample_messages in batched_messages:
            model_config = getattr(getattr(self.reasoning, "model", None), "config", None)
            model_inputs = self.operator.process_perception(
                sample_messages,
                tokenizer=self.reasoning.tokenizer,
                image_processors=self.reasoning.image_processors,
                model_config=model_config,
            )
            outputs.extend(
                self.reasoning.inference(
                    inputs=model_inputs,
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                )
            )

        self.operator.delete_last_interaction()
        return outputs
