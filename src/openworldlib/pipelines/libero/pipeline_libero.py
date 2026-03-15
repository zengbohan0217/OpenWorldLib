from typing import Any, Dict, List, Optional

from ...operators.libero_operator import LiberoOperator
from ...synthesis.visual_generation.libero.libero_synthesis import LiberoSynthesis
from ...pipelines.pipeline_utils import PipelineABC


class LiberoPipeline(PipelineABC):
    """
    Pipeline for visualizing robot control signals in a LIBERO environment.

    Flow:
        raw actions  →  LiberoOperator (unnormalization)
                     →  LiberoSynthesis.predict (env execution + video recording)
    """

    def __init__(
        self,
        *,
        operator: LiberoOperator,
        synthesis_model: LiberoSynthesis,
        norm_stats: Optional[Dict[str, Any]] = None,
        action_dim: int = 7,
    ) -> None:
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.norm_stats = norm_stats
        self.action_dim = action_dim

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        bddl_file: str,
        *,
        benchmark_name: str = "libero_10",
        task_id: int = 0,
        norm_stats: Optional[Dict[str, Any]] = None,
        action_dim: int = 7,
        device: Optional[int] = None,
        camera_heights: int = 256,
        camera_widths: int = 256,
        camera_names: Optional[List[str]] = None,
        init_state_idx: int = 0,
        **kwargs,
    ) -> "LiberoPipeline":
        """
        Build the pipeline by initialising the LIBERO environment from a BDDL file.

        Args:
            bddl_file:             Absolute path to the .bddl task file.
            benchmark_name:        LIBERO benchmark name (default "libero_10"),
                                   used to load initial states.
            task_id:               Task index within the benchmark for initial
                                   states (default 0).
            norm_stats:            Optional unnormalization statistics:
                                   {"actions": {"mean": [...], "std": [...]}}.
            action_dim:            Number of action dimensions to keep (default 7).
            device:                GPU device ID for rendering (default 0).
            camera_heights/widths: Render resolution.
            camera_names:          List of camera names.
            init_state_idx:        Index of the initial state to use.
        """
        operator = LiberoOperator()
        synthesis_model = LiberoSynthesis.from_pretrained(
            bddl_file=bddl_file,
            benchmark_name=benchmark_name,
            task_id=task_id,
            device=device,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_names=camera_names,
            init_state_idx=init_state_idx,
        )
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            norm_stats=norm_stats,
            action_dim=action_dim,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        actions: List[Any],
    ) -> Dict[str, Any]:
        """
        Run the operator to produce processed (unnormalized) actions.

        Args:
            actions: Raw action sequence from a VLA pipeline.

        Returns:
            Dict with key "actions": list of np.ndarray.
        """
        self.operator.get_interaction(actions)
        processed = self.operator.process_interaction(
            norm_stats=self.norm_stats,
            action_dim=self.action_dim,
        )
        return processed

    def __call__(
        self,
        *,
        actions: List[Any],
        video_path: Optional[str] = "libero_execution.mp4",
        fps: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute and visualize the robot action sequence.

        Args:
            actions:    Raw action sequence (list of 7- or 8-dim arrays).
            video_path: Output video path (pass None to skip recording).
            fps:        Frame rate of the output video.

        Returns:
            Dict with "success", "frames", and "video_path".
        """
        processed = self.process(actions=actions)
        result = self.synthesis_model.predict(
            processed_inputs=processed,
            video_path=video_path,
            fps=fps,
        )
        return result

    def stream(self, **kwargs):
        raise NotImplementedError("LiberoPipeline does not support streaming.")
