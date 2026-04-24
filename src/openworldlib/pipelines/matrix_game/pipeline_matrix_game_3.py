from pathlib import Path
from typing import Optional, List, Any, Union
import logging
import warnings

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from diffusers.utils import export_to_video

from ...operators.matrix_game_3_operator import MatrixGame3Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_3_synthesis import MatrixGame3Synthesis
from ...memories.visual_synthesis.matrix_game.matrix_game_3_memory import MatrixGame3Memory


class MatrixGame3Pipeline:
    """Matrix-Game-3 pipeline following OpenWorldLib (Operator + Synthesis + Memory)."""

    def __init__(
        self,
        operators: Optional[MatrixGame3Operator] = None,
        synthesis_model: Optional[MatrixGame3Synthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
    ):
        self.operators = operators
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module
        self.device = device
        self.current_image = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        required_components: Optional[dict] = None,
        code_dir: Optional[str] = None,
        visualize_warning: bool = False,
        **kwargs,
    ) -> "MatrixGame3Pipeline":
        if not model_path:
            raise ValueError("MatrixGame3Pipeline requires a local `model_path` as checkpoint_dir.")

        synthesis_model = MatrixGame3Synthesis.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            code_dir=code_dir,
            visualize_warning=visualize_warning,
        )
        operators = MatrixGame3Operator()
        memory_module = MatrixGame3Memory()
        return cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
        )

    def process(self, input_image: Image.Image, interactions: Optional[List[str]] = None) -> dict:
        # MG3 clip-level default: first clip has 57 frames.
        interaction_payload = self.operators.process_interaction(interactions or [], num_frames=57)
        return {
            "image": input_image,
            **interaction_payload,
        }

    def __call__(
        self,
        images: Image.Image,
        interactions: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_name: str = "matrix_game_3_demo",
        size: str = "704*1280",
        num_iterations: int = 12,
        num_inference_steps: int = 3,
        seed: int = 42,
        sample_shift: Optional[float] = None,
        sample_guide_scale: Optional[float] = None,
        fa_version: str = "0",
        use_int8: bool = False,
        verify_quant: bool = False,
        use_async_vae: bool = False,
        async_vae_warmup_iters: int = 0,
        compile_vae: bool = False,
        lightvae_pruning_rate: Optional[float] = None,
        vae_type: str = "mg_lightvae_v2",
        use_base_model: bool = False,
        save_video: bool = True,
        return_result: bool = False,
        video_save_path: Optional[Union[str, Path]] = None,
        visualize_warning: bool = False,
        **kwargs,
    ) -> Any:
        if not isinstance(images, Image.Image):
            raise ValueError("MatrixGame3Pipeline expects `images` to be a PIL.Image.")
        if self.synthesis_model is None:
            raise RuntimeError("MatrixGame3Pipeline.synthesis_model is not initialized.")
        if not visualize_warning:
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\. Please use `torch\.amp\.autocast\('cuda', args\.\.\.\)` instead\.",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torch\.load.*weights_only=False.*",
                category=FutureWarning,
            )
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)

        processed_inputs = self.process(images, interactions=interactions)

        prompt_text = prompt or "A first-person view interactive scene."
        need_payload = return_result or (video_save_path is not None)
        prediction: Any = self.synthesis_model.predict(
            image=processed_inputs["image"],
            prompt=prompt_text,
            interactions=interactions,
            operator_condition=processed_inputs,
            output_dir=output_dir,
            save_name=save_name,
            size=size,
            num_iterations=num_iterations,
            num_inference_steps=num_inference_steps,
            seed=seed,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            fa_version=fa_version,
            use_int8=use_int8,
            verify_quant=verify_quant,
            use_async_vae=use_async_vae,
            async_vae_warmup_iters=async_vae_warmup_iters,
            compile_vae=compile_vae,
            lightvae_pruning_rate=lightvae_pruning_rate,
            vae_type=vae_type,
            use_base_model=use_base_model,
            save_video=save_video,
            return_result=need_payload,
            visualize_warning=visualize_warning,
            **kwargs,
        )

        if video_save_path is not None:
            if not isinstance(prediction, dict):
                raise RuntimeError("Expected payload dict when `video_save_path` is set.")
            video_tensor = prediction.get("video_tensor")
            if video_tensor is None:
                raise RuntimeError("Pipeline did not return `video_tensor`; cannot save to custom path.")
            saved_path = self.save_video_tensor(video_tensor, video_save_path)
            prediction["video_path"] = saved_path

        if return_result:
            return prediction
        if video_save_path is not None:
            return str(Path(video_save_path))
        return prediction

    @staticmethod
    def save_video_tensor(video_tensor: torch.Tensor, save_path: Union[str, Path], fps: int = 17) -> str:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        video_np = np.ascontiguousarray(
            ((rearrange(video_tensor, "C T H W -> T H W C").float() + 1) * 127.5)
            .clip(0, 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        export_to_video([frame / 255.0 for frame in video_np], str(path), fps=fps)
        return str(path)


    def stream(
        self,
        images: Optional[Image.Image],
        interactions: List[str],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if self.memory_module is None:
            raise RuntimeError("MatrixGame3Pipeline.memory_module is not initialized.")
        if images is not None:
            self.memory_module.record(images)
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'images' first.")
        return self.__call__(images=current_image, interactions=interactions, prompt=prompt, **kwargs)
