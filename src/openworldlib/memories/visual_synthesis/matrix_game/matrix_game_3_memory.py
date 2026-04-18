from ...base_memory import BaseMemory
from typing import Optional
from PIL import Image


class MatrixGame3Memory(BaseMemory):
    """
    Minimal memory module for Matrix-Game-3.

    For now, we only store the latest input image so `pipeline.stream(...)` can
    behave similarly to Matrix-Game-2. The generated artifact is an mp4 path.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []

    def record(self, data, **kwargs):
        if isinstance(data, Image.Image):
            self.storage.append(
                {
                    "content": data,
                    "type": "image",
                    "timestamp": len(self.storage),
                    "metadata": {},
                }
            )

    def select(self, **kwargs) -> Optional[Image.Image]:
        if not self.storage:
            return None
        return self.storage[-1]["content"]

    def manage(self, action: str = "reset", **kwargs):
        if action == "reset":
            self.storage = []

