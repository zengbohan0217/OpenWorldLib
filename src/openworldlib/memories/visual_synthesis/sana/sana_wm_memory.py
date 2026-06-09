from ...base_memory import BaseMemory
import numpy as np
from PIL import Image
from typing import Optional, Union


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert ``(H, W, C)`` numpy array (float [0,1] or uint8) to PIL Image."""
    if image_array.dtype in (np.float32, np.float16):
        image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)


class SanaWMMemory(BaseMemory):
    """Sana-WM memory module implementing ``BaseMemory``.

    Stores context images for multi-turn streaming: the first call records
    the input image, subsequent calls append the last generated frame
    and use it as the start image for the next turn.
    """

    def __init__(self, capacity: int = 10, **kwargs):
        super().__init__(capacity=capacity, **kwargs)
        self.storage = []       # Context per round (PIL Images)
        self.all_frames = []    # All generated frames (numpy arrays)

    def record(self, data: Union[Image.Image, np.ndarray], type: str = "image", metadata=None, **kwargs):
        """Record a new observation.

        Args:
            data: PIL image (initial input) or ``(T, H, W, C)`` numpy video.
            type: ``"image"`` or ``"video_chunk"``.
            metadata: Optional metadata dict.
        """
        current_image = None

        if isinstance(data, Image.Image):
            current_image = data

        elif isinstance(data, np.ndarray):
            # Append to all_frames for final assembly
            self.all_frames.append(data)
            # Use the last frame as context for next turn
            last_frame = data[-1]
            current_image = numpy_to_pil(last_frame)

        if current_image is not None:
            entry = {
                "content": current_image,
                "type": type,
                "metadata": metadata or {},
            }
            self.storage.append(entry)
            if self.capacity and len(self.storage) > self.capacity:
                self.storage.pop(0)

    def select(self, context_query=None, **kwargs) -> Optional[Image.Image]:
        """Return the most recent stored image as the start frame."""
        if not self.storage:
            return None
        return self.storage[-1]["content"]

    def manage(self, action: str = "reset", **kwargs):
        """Manage storage lifecycle."""
        if action == "reset":
            self.storage = []
            self.all_frames = []