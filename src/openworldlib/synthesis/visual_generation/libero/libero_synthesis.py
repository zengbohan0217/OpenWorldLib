import os
os.environ['MUJOCO_GL'] = 'osmesa'

from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from ....synthesis.base_synthesis import BaseSynthesis


class LiberoSynthesis(BaseSynthesis):
    """
    LIBERO simulation synthesis layer.

    Responsibilities:
    - Hold the initialised OffScreenRenderEnv and related benchmark objects.
    - Execute a processed action sequence step-by-step (predict).
    - Optionally record a video of the execution.
    """

    def __init__(
        self,
        *,
        env: OffScreenRenderEnv,
        bm: Any,
        task_id: int,
        init_state_idx: int = 0,
    ) -> None:
        super().__init__()
        self.env = env
        self.bm = bm
        self.task_id = task_id
        self.init_state_idx = init_state_idx

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        bddl_file: str,
        args: Optional[Any] = None,
        device: Optional[int] = None,
        *,
        benchmark_name: str = "libero_10",
        task_id: int = 0,
        camera_heights: int = 256,
        camera_widths: int = 256,
        camera_names: Optional[List[str]] = None,
        init_state_idx: int = 0,
        **kwargs,
    ) -> "LiberoSynthesis":
        """
        Initialise a LIBERO environment from a specific BDDL file path.

        Args:
            bddl_file:             Absolute path to the .bddl task file.
            benchmark_name:        LIBERO benchmark name (e.g. "libero_10"),
                                   used to load initial states.
            task_id:               Task index within the benchmark, used to
                                   retrieve initial states (default 0).
            camera_heights/widths: Render resolution.
            camera_names:          List of camera names to render.
            init_state_idx:        Index of the initial state to use.
        """
        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]

        render_gpu_device_id = device if device is not None else 0

        bm_obj = benchmark.get_benchmark(benchmark_name)()

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_names": camera_names,
            "render_gpu_device_id": render_gpu_device_id,
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        # Reset to the designated initial state
        env.reset()
        init_states = bm_obj.get_task_init_states(task_id)
        env.set_init_state(init_states[init_state_idx])
        env.reset()

        return cls(
            env=env,
            bm=bm_obj,
            task_id=task_id,
            init_state_idx=init_state_idx,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        *,
        processed_inputs: Dict[str, Any],
        video_path: Optional[str] = "libero_execution.mp4",
        fps: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the action sequence in the LIBERO environment.

        Args:
            processed_inputs: Dict with key "actions": list of np.ndarray (7-dim each).
            video_path:        If provided, record an mp4 video to this path.
            fps:               Frame rate of the output video.

        Returns:
            Dict with:
                - "success":    bool
                - "frames":     list of np.ndarray (H, W, 3)
                - "video_path": str or None
        """
        actions: List[np.ndarray] = processed_inputs["actions"]

        writer = None
        if video_path is not None:
            writer = imageio.get_writer(video_path, fps=fps)

        frames = []
        success = False

        print(f">>> Executing action sequence ({len(actions)} steps)...")

        for i, action in enumerate(actions):
            env_action = np.array(action)
            obs, _reward, _done, _info = self.env.step(env_action)

            frame = obs["agentview_image"][::-1]  # flip vertical if needed
            frames.append(frame)

            if writer is not None:
                writer.append_data(frame)

            if (i + 1) % 10 == 0:
                print(f"    Step {i + 1}/{len(actions)}")

            if self.env.check_success():
                success = True
                print(f"!!! Task succeeded at step {i + 1} !!!")

        if writer is not None:
            writer.close()

        print("-" * 40)
        if video_path:
            print(f"Visualization complete. Video saved to: {video_path}")
        print(f"Final result: {'SUCCESS' if success else 'FAIL'}")
        print("-" * 40)

        return {
            "success": success,
            "frames": frames,
            "video_path": video_path,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the LIBERO environment."""
        self.env.close()
