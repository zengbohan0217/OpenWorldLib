from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base_operator import BaseOperator
from ..synthesis.visual_generation.infinite_world.infworld.configs import (
    bucket_config as bucket_config_module,
)


MOVE_ACTION_MAP = {
    "no-op": 0,
    "go forward": 1,
    "go back": 2,
    "go left": 3,
    "go right": 4,
    "go forward and go left": 5,
    "go forward and go right": 6,
    "go back and go left": 7,
    "go back and go right": 8,
    "uncertain": 9,
}

VIEW_ACTION_MAP = {
    "no-op": 0,
    "turn up": 1,
    "turn down": 2,
    "turn left": 3,
    "turn right": 4,
    "turn up and turn left": 5,
    "turn up and turn right": 6,
    "turn down and turn left": 7,
    "turn down and turn right": 8,
    "uncertain": 9,
}


ImageInput = Union[Image.Image, np.ndarray]
VideoInput = Union[Image.Image, np.ndarray, Sequence[ImageInput]]


class InfiniteWorldOperator(BaseOperator):
    def __init__(
        self,
        operation_types: Optional[List[str]] = None,
        bucket_config_name: str = "ASPECT_RATIO_627_F64",
    ):
        super().__init__(operation_types=operation_types or ["action_instruction"])
        self.bucket_config_name = bucket_config_name
        self.interaction_template = [
            "idle",
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
            "camera_up",
            "camera_down",
            "camera_l",
            "camera_r",
            "camera_ul",
            "camera_ur",
            "camera_dl",
            "camera_dr",
        ]
        self.interaction_aliases = {
            "camera_up_l": "camera_ul",
            "camera_up_r": "camera_ur",
            "camera_down_l": "camera_dl",
            "camera_down_r": "camera_dr",
        }
        self.move_actions = {
            "idle": "no-op",
            "forward": "go forward",
            "backward": "go back",
            "left": "go left",
            "right": "go right",
            "forward_left": "go forward and go left",
            "forward_right": "go forward and go right",
            "backward_left": "go back and go left",
            "backward_right": "go back and go right",
        }
        self.view_actions = {
            "camera_up": "turn up",
            "camera_down": "turn down",
            "camera_l": "turn left",
            "camera_r": "turn right",
            "camera_ul": "turn up and turn left",
            "camera_ur": "turn up and turn right",
            "camera_dl": "turn down and turn left",
            "camera_dr": "turn down and turn right",
        }
        self.interaction_template_init()

    def check_interaction(self, interaction: str):
        if not isinstance(interaction, str):
            raise TypeError(f"interaction must be str, got {type(interaction)}")
        self._parse_interaction(interaction)
        return True

    def get_interaction(self, interaction):
        if not isinstance(interaction, list):
            interaction = [interaction]
        if len(interaction) == 0:
            raise ValueError("interaction list cannot be empty")
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.append(interaction)

    def _parse_interaction(self, interaction: str) -> Tuple[str, str]:
        tokens = [token.strip() for token in interaction.split("+") if token.strip()]
        if len(tokens) == 0:
            raise ValueError("interaction cannot be empty")

        move_label = "no-op"
        view_label = "no-op"
        for token in tokens:
            canonical_token = self.interaction_aliases.get(token, token)
            if canonical_token in self.move_actions:
                if move_label != "no-op":
                    raise ValueError(f"multiple move actions found in '{interaction}'")
                move_label = self.move_actions[canonical_token]
            elif canonical_token in self.view_actions:
                if view_label != "no-op":
                    raise ValueError(f"multiple view actions found in '{interaction}'")
                view_label = self.view_actions[canonical_token]
            else:
                raise ValueError(
                    f"Unsupported interaction token '{token}'. "
                    f"Supported base actions: {self.interaction_template} and '+' combinations."
                )
        return move_label, view_label

    def _build_action_sequence(
        self, parsed_actions: List[Tuple[str, str]], total_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_steps = max(int(total_steps), len(parsed_actions), 1)
        base, remainder = divmod(total_steps, len(parsed_actions))

        move_ids: List[int] = []
        view_ids: List[int] = []
        for idx, (move_label, view_label) in enumerate(parsed_actions):
            repeat = base + (1 if idx < remainder else 0)
            move_ids.extend([MOVE_ACTION_MAP[move_label]] * repeat)
            view_ids.extend([VIEW_ACTION_MAP[view_label]] * repeat)

        move_tensor = torch.tensor(move_ids[:total_steps], dtype=torch.long)
        view_tensor = torch.tensor(view_ids[:total_steps], dtype=torch.long)
        return move_tensor, view_tensor

    def process_interaction(self, num_frames: int):
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")

        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        parsed_actions = [self._parse_interaction(action) for action in now_interaction]
        move_ids, view_ids = self._build_action_sequence(parsed_actions, num_frames)

        return {
            "move_ids": move_ids,
            "view_ids": view_ids,
        }

    def _bucket_config(self) -> Dict[str, Tuple[Tuple[int, int], int]]:
        if not hasattr(bucket_config_module, self.bucket_config_name):
            raise ValueError(f"Unknown bucket config: {self.bucket_config_name}")
        return getattr(bucket_config_module, self.bucket_config_name)

    def _resolve_target_size(
        self, frame: Image.Image, size: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        if size is not None:
            target_h, target_w = size
            if target_h % 64 != 0 or target_w % 64 != 0:
                raise ValueError(
                    f"size must be multiples of 64 for Infinite-World, got {size}"
                )
            return int(target_h), int(target_w)

        bucket_config = self._bucket_config()
        ratio = frame.height / frame.width
        closest_bucket = min(bucket_config, key=lambda key: abs(float(key) - ratio))
        target_h, target_w = bucket_config[closest_bucket][0]
        return int(target_h), int(target_w)

    def _to_pil(self, frame: ImageInput) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return Image.fromarray(frame).convert("RGB")
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _resize_center_crop(self, frame: Image.Image, size: Tuple[int, int]) -> Image.Image:
        target_h, target_w = size
        if frame.size == (target_w, target_h):
            return frame

        orig_w, orig_h = frame.size
        scale = max(target_h / orig_h, target_w / orig_w)
        resize_h = int(round(orig_h * scale))
        resize_w = int(round(orig_w * scale))
        resized = TF.resize(frame, (resize_h, resize_w), antialias=True)
        return TF.center_crop(resized, (target_h, target_w))

    def process_perception(
        self,
        images: VideoInput,
        size: Optional[Tuple[int, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
        weight_dtype=torch.float32,
    ):
        if isinstance(images, (Image.Image, np.ndarray)):
            pil_frames = [self._to_pil(images)]
        elif isinstance(images, Sequence):
            pil_frames = [self._to_pil(frame) for frame in images]
        else:
            raise TypeError(f"Unsupported image input type: {type(images)}")

        if len(pil_frames) == 0:
            raise ValueError("At least one frame is required")

        target_size = self._resolve_target_size(pil_frames[0], size)
        processed_frames = [
            self._resize_center_crop(frame, target_size) for frame in pil_frames
        ]

        frame_tensors = []
        exported_frames = []
        for frame in processed_frames:
            exported_frames.append(np.array(frame, dtype=np.uint8))
            tensor = TF.pil_to_tensor(frame).float().div(255.0)
            tensor = tensor.sub(0.5).div(0.5)
            frame_tensors.append(tensor)

        video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0)
        if device is not None:
            video_tensor = video_tensor.to(device=device, dtype=weight_dtype)

        return {
            "video": video_tensor,
            "size": target_size,
            "frames": exported_frames,
        }
