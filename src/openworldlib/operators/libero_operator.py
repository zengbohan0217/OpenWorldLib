from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base_operator import BaseOperator


class LiberoOperator(BaseOperator):
    """
    LIBERO robot control Operator.

    - process_interaction: receives raw robot action signal(s) from a VLA pipeline
      and performs optional unnormalization.
    - process_perception: not used for robot control, kept for API consistency.
    """

    def __init__(self, operation_types: Optional[List[str]] = None) -> None:
        if operation_types is None:
            operation_types = ["action_instruction"]
        super(LiberoOperator, self).__init__(operation_types=operation_types)

        self.interaction_template = ["robot_action"]
        self.interaction_template_init()

    # ------------------------------------------------------------------
    # Interaction management
    # ------------------------------------------------------------------

    def get_interaction(self, interaction: Any) -> None:
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: Any) -> bool:
        if not isinstance(interaction, (list, np.ndarray)):
            raise TypeError(
                f"Interaction must be a list or np.ndarray of actions, got {type(interaction)}"
            )
        return True

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_interaction(
        self,
        *,
        norm_stats: Optional[Dict[str, Any]] = None,
        action_dim: int = 7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process the raw action sequence stored via get_interaction().

        Args:
            norm_stats: Optional dict with keys "actions" -> {"mean": [...], "std": [...]}.
                        If provided, each action is unnormalized before being returned.
            action_dim: Number of dimensions to retain after unnormalization (default 7).

        Returns:
            Dict with key "actions": list of np.ndarray, each of shape (action_dim,).
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process. Call get_interaction() first.")

        raw_actions = self.current_interaction[-1]
        self.interaction_history.append(raw_actions)

        processed_actions = []
        for raw_action in raw_actions:
            action = np.array(raw_action, dtype=np.float64)

            if norm_stats is not None:
                action = self._unnormalize(action, norm_stats, action_dim)
            else:
                action = action[:action_dim]

            processed_actions.append(action)

        return {"actions": processed_actions}

    def process_perception(self, **kwargs) -> Dict[str, Any]:
        """Not used for robot control; returns an empty dict."""
        return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unnormalize(
        self,
        action: np.ndarray,
        norm_stats: Dict[str, Any],
        action_dim: int,
    ) -> np.ndarray:
        """
        Unnormalize a single action:  env_action = (action * std) + mean

        The stats arrays are padded / truncated to match the full action
        dimensionality, and the result is clipped to `action_dim`.
        """
        stats = norm_stats["actions"]
        mean = np.array(stats["mean"], dtype=np.float64)
        std = np.array(stats["std"], dtype=np.float64)

        full_dim = max(len(action), len(mean), len(std))

        # Pad action, mean, std to full_dim
        def _pad(arr: np.ndarray, size: int) -> np.ndarray:
            if len(arr) < size:
                return np.pad(arr, (0, size - len(arr)), "constant")
            return arr[:size]

        action_full = _pad(action, full_dim)
        mean_full = _pad(mean, full_dim)
        std_full = _pad(std, full_dim)

        unnormalized = action_full * std_full + mean_full
        return unnormalized[:action_dim]
