from .base_operator import BaseOperator
import torch
from typing import List, Dict, Optional


class MatrixGame3Operator(BaseOperator):
    """
    Minimal operator wrapper for Matrix-Game-3.

    Matrix-Game-3 upstream currently generates action-conditioned trajectories internally
    (unless using its interactive mode). In OpenWorldLib we keep `interactions` for API
    compatibility with other navigation video pipelines.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        super().__init__(operation_types=operation_types or [])
        self.interaction_template = interaction_template or [
            "forward",
            "back",
            "left",
            "right",
            "camera_l",
            "camera_r",
            "forward_left",
            "forward_right",
            "back_left",
            "back_right",
            "nomove",
        ]
        self.interaction_template_init()

    def check_interaction(self, interaction: str) -> bool:
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template {self.interaction_template}")
        return True

    def get_interaction(self, interaction_list: List[str]):
        for act in interaction_list:
            self.check_interaction(act)
        self.current_interaction.append(interaction_list)

    @staticmethod
    def _encode_single_action(action: str):
        """
        Encode one action into MG3 condition format:
        - keyboard_condition: 6-dim (first 4 dims used by movement; last 2 kept for compatibility)
        - mouse_condition: 2-dim camera delta
        """
        keyboard = torch.zeros(6, dtype=torch.float32)
        mouse = torch.zeros(2, dtype=torch.float32)

        keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3}
        combo_map = {
            "forward_left": ["forward", "left"],
            "forward_right": ["forward", "right"],
            "back_left": ["back", "left"],
            "back_right": ["back", "right"],
        }
        cam_map = {
            "camera_l": [0.0, -0.1],
            "camera_r": [0.0, 0.1],
        }

        if action in combo_map:
            for sub_action in combo_map[action]:
                keyboard[keyboard_idx[sub_action]] = 1.0
        elif action in keyboard_idx:
            keyboard[keyboard_idx[action]] = 1.0
        elif action in cam_map:
            mouse = torch.tensor(cam_map[action], dtype=torch.float32)
        elif action == "nomove":
            pass
        else:
            raise ValueError(f"Unsupported action: {action}")

        return keyboard, mouse

    def _build_sequence(self, interactions: List[str], num_frames: int):
        if len(interactions) == 0:
            interactions = ["nomove"]
        frames_per_action = max(1, num_frames // len(interactions))
        padded_actions = []
        for action in interactions:
            padded_actions.extend([action] * frames_per_action)
        while len(padded_actions) < num_frames:
            padded_actions.append(padded_actions[-1])
        padded_actions = padded_actions[:num_frames]

        keyboard_list = []
        mouse_list = []
        for action in padded_actions:
            kb, ms = self._encode_single_action(action)
            keyboard_list.append(kb)
            mouse_list.append(ms)

        keyboard_tensor = torch.stack(keyboard_list, dim=0)
        mouse_tensor = torch.stack(mouse_list, dim=0)
        return {
            "keyboard_condition": keyboard_tensor,
            "mouse_condition": mouse_tensor,
            "actions_per_frame": padded_actions,
        }

    def process_interaction(self, interactions: Optional[List[str]], num_frames: int = 57) -> Dict[str, torch.Tensor]:
        interactions = interactions or ["nomove"]
        for action in interactions:
            self.check_interaction(action)
        return self._build_sequence(interactions, num_frames=num_frames)

