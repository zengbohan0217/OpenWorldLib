

from typing import Any, List, Optional, Union

import torch
from transformers import AutoTokenizer

from ...base_reasoning import BaseReasoning
from .constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from .mm_utils import validate_cambrian_s_environment
from .modeling_cambrian_s import CambrianSForCausalLM


class CambrianSReasoning(BaseReasoning):
    """
    Cambrian-S: https://arxiv.org/abs/2511.04670
    """

    def __init__(
        self,
        model: CambrianSForCausalLM,
        tokenizer: Any,
        image_processors: list[Any],
        device: Union[str, "torch.device"] = "cuda",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processors = image_processors
        self.processor = image_processors[0] if image_processors else None
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "nyu-visionx/Cambrian-S-7B",
        device: Optional[Union[str, "torch.device"]] = None,
        weight_dtype: "torch.dtype" = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> "CambrianSReasoning":
        validate_cambrian_s_environment(require_video=False)

        config_override_names = (
            "video_max_frames",
            "video_fps",
            "video_force_sample",
            "add_time_instruction",
            "miv_token_len",
            "si_token_len",
            "image_aspect_ratio",
            "anyres_max_subimages",
        )
        config_overrides = {
            attr_name: kwargs.pop(attr_name)
            for attr_name in config_override_names
            if attr_name in kwargs
        }

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if weight_dtype is None:
            weight_dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = CambrianSForCausalLM.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=kwargs.pop("low_cpu_mem_usage", True),
            attn_implementation=attn_implementation,
            **kwargs,
        )
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
        model.resize_token_embeddings(len(tokenizer))
        for attr_name, attr_value in config_overrides.items():
            setattr(model.config, attr_name, attr_value)
        model = model.to(device)
        model.load_vision_towers(device=device, dtype=weight_dtype)
        image_processors = [tower.image_processor for tower in model.get_model().get_vision_tower_aux_list()]
        return cls(model=model, tokenizer=tokenizer, image_processors=image_processors, device=device)

    def api_init(self, api_key, endpoint):
        raise NotImplementedError("API init is not supported for Cambrian-S.")

    def _get_default_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def inference(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 2048,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        generation_config = {"max_new_tokens": max_new_tokens}
        if generation_kwargs:
            generation_config.update(generation_kwargs)

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        images = inputs.get("images") or []
        model_dtype = next(self.model.parameters()).dtype
        images = [image.to(device=self.device, dtype=model_dtype) for image in images]
        image_sizes = inputs.get("image_sizes") or []

        generated_ids = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_sizes=image_sizes,
            **generation_config,
        )
        if hasattr(generated_ids, "sequences"):
            generated_ids = generated_ids.sequences

        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
