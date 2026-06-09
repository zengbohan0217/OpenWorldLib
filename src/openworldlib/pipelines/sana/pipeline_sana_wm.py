import torch
import numpy as np
from PIL import Image
from typing import Any, Optional

from ...operators.sana_wm_operator import SanaWMOperator
from ...synthesis.visual_generation.sana.sana_wm.sana_wm_synthesis import SanaWMSynthesis
from ...memories.visual_synthesis.sana.sana_wm_memory import SanaWMMemory


class SanaWMPipeline:

    def __init__(
        self,
        operators: Optional[SanaWMOperator] = None,
        synthesis_model: Optional[SanaWMSynthesis] = None,
        memory_module: Optional[SanaWMMemory] = None,
        device: str = "cuda",
    ):
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device


    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "Efficient-Large-Model/SANA-WM_bidirectional",
        device: str = "cuda",
        text_encoder_path: str | None = None,
        **kwargs,
    ) -> "SanaWMPipeline":

        print(f"[SanaWMPipeline] Loading Sana-WM from {model_path}...")

        synthesis_model = SanaWMSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            text_encoder_path=text_encoder_path,
            **kwargs,
        )

        operators = SanaWMOperator()
        memory_module = SanaWMMemory()

        return cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
        )


    def process(
        self,
        images: Any = None,
        prompt: Optional[str] = None,
        interactions: Optional[list[str]] = None,
        c2ws: Optional[np.ndarray] = None,
        intrinsics_vec4: Optional[np.ndarray] = None,
        num_frames: int = 161,
    ) -> dict:
        """Run Operator perception + interaction pre-processing.

        Args:
            images: Input PIL image or path string.
            prompt: Text prompt.
            interactions: List of action DSL segments (e.g. ``["w-80", "none-10"]``).
            c2ws: Explicit ``(F, 4, 4)`` camera-to-world poses (alternative to ``interactions``).
            intrinsics_vec4: ``(F, 4)`` intrinsics ``[fx, fy, cx, cy]``.
            num_frames: Target number of frames.

        Returns:
            Dict with keys consumed by ``synthesis_model.predict``.
        """
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")

        # Use "pil_image" key (consistent with operator.process_perception return).
        perception = self.operators.process_perception(images)

        # Interaction: build c2ws + intrinsics
        interaction_signal = {
            "prompt": prompt if prompt is not None else "",
            "action_list": interactions if interactions is not None else None,
            "c2ws": c2ws,
            "intrinsics_vec4": intrinsics_vec4,
        }
        self.operators.get_interaction(interaction_signal)
        interaction_out = self.operators.process_interaction(
            perception=perception,
            num_frames=num_frames,
        )

        return {
            "pil_image": perception["pil_image"],
            "prompt": interaction_out["prompt"],
            "c2ws": interaction_out["c2ws"],
            "intrinsics_vec4": interaction_out["intrinsics_vec4"],
        }

    def __call__(
        self,
        images: Any = None,
        num_frames: int = 161,
        prompt: Optional[str] = None,
        interactions: Optional[list[str]] = None,
        c2ws: Optional[np.ndarray] = None,
        intrinsics_vec4: Optional[np.ndarray] = None,
        seed: int = 42,
        **kwds,
    ) -> Optional[np.ndarray]:
        """Single-shot I2V generation.

        Returns:
            ``(T, H, W, 3)`` uint8 numpy array, or ``None`` if generation fails.
        """
        processed = self.process(
            images=images,
            prompt=prompt,
            interactions=interactions,
            c2ws=c2ws,
            intrinsics_vec4=intrinsics_vec4,
            num_frames=num_frames,
        )

        result = self.synthesis_model.predict(
            image=processed["pil_image"],
            prompt=processed["prompt"],
            c2ws=processed["c2ws"],
            intrinsics_vec4=processed["intrinsics_vec4"],
            num_frames=num_frames,
            seed=seed,
            **kwds,
        )

        return result.get("video") if result else None

    # -------------------------------------------------------------------
    # stream: multi-turn generation with memory
    # -------------------------------------------------------------------

    def stream(
        self,
        prompt: Optional[str] = None,
        interactions: Optional[list[str]] = None,
        images: Any = None,
        num_frames: int = 161,
        seed: int = 42,
        **kwds,
    ) -> Optional[np.ndarray]:
        """Multi-turn streaming generation.

        The first call (with ``images``) resets memory. Subsequent calls
        use the last generated frame as the new start image.

        Args:
            prompt: Text prompt for this turn.
            interactions: Action DSL segments for this turn.
            images: Input PIL image or path (first turn only).
            num_frames: Frames to generate in this turn.
            seed: Random seed.
            **kwds: Passed through to ``Synthesis.predict``.

        Returns:
            ``(T, H, W, 3)`` uint8 numpy array.
        """
        # 1. Initialise memory
        if images is not None:
            print("[SanaWMPipeline] --- Stream started ---")
            self.memory_module.manage(action="reset")
            if isinstance(images, str):
                images = Image.open(images).convert("RGB")
            self.memory_module.record(images, type="image")

        # 2. Retrieve current context
        current_img = self.memory_module.select()
        if current_img is None:
            raise ValueError("No image in storage. Provide 'images' on the first call.")

        # 3. Generate
        video = self.__call__(
            images=current_img,
            num_frames=num_frames,
            prompt=prompt,
            interactions=interactions,
            seed=seed,
            **kwds,
        )

        # 4. Update memory
        if video is not None:
            self.memory_module.record(video, type="video_chunk")

        return video