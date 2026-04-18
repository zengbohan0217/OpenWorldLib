from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ...base_memory import BaseMemory


def _np_to_pil(frame: np.ndarray) -> Image.Image:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


class InfiniteWorldMemory(BaseMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []
        self.all_frames: List[np.ndarray] = []
        self.target_size: Optional[Tuple[int, int]] = None

    def has_frames(self) -> bool:
        return len(self.all_frames) > 0

    def record(self, data, processed_frames=None, target_size=None, **kwargs):
        if isinstance(data, Image.Image):
            frames = processed_frames or [np.array(data.convert("RGB"), dtype=np.uint8)]
            if len(frames) == 0:
                raise ValueError("processed_frames cannot be empty")
            self.all_frames = [np.array(frame, dtype=np.uint8) for frame in frames]
            self.target_size = target_size
            current_image = _np_to_pil(self.all_frames[-1])
        elif isinstance(data, list):
            frames = [np.array(frame, dtype=np.uint8) for frame in data]
            if len(frames) == 0:
                raise ValueError("generated frame list cannot be empty")
            self.all_frames.extend(frames)
            current_image = _np_to_pil(self.all_frames[-1])
        else:
            raise TypeError(f"Unsupported data type for record(): {type(data)}")

        self.storage.append(
            {
                "content": current_image,
                "type": "image",
                "timestamp": len(self.all_frames),
                "metadata": {
                    "target_size": self.target_size,
                    "num_frames": len(self.all_frames),
                },
            }
        )

    def select(self, **kwargs) -> Optional[Image.Image]:
        if len(self.storage) == 0:
            return None
        return self.storage[-1]["content"]

    def select_frames(self) -> List[np.ndarray]:
        return list(self.all_frames)

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []
            self.all_frames = []
            self.target_size = None
