from typing import List, Optional, Union

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ...base_reasoning import BaseReasoning


class SpatialReasonerReasoning(BaseReasoning):
    """
    SpatialReasoner:https://arxiv.org/abs/2504.20024
    """
    
    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "ccvl/SpatialReasoner",
        device: Union[str, torch.device] = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> "SpatialReasonerReasoning":
        """
        Load SpatialReasoner model and processor.

        Args:
            model_path: HuggingFace model ID or local path to the model.
            device: Target device to load the model onto (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda".
            weight_dtype: Weight dtype for the model (e.g. torch.bfloat16, torch.float16). Defaults to torch.bfloat16.
            attn_implementation: Attention implementation override.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        return cls(model=model, processor=processor, device=device)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("API init is not supported for SpatialReasoner.")

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def inference(
        self,
        inputs,
        max_new_tokens: int = 2048,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        """
        Run SpatialReasoner generation on preprocessed model inputs.
        """
        inputs = inputs.to(self.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
