import os

from diffusers.utils import export_to_video
import imageio
from PIL import Image
import numpy as np

from openworldlib.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline

def save_uint8_video(video_frames, output_path, fps=30):
    # Infinite-World returns uint8 frames in [0, 255]; diffusers.export_to_video
    # multiplies ndarray frames by 255 again, which causes white/gray overflow artifacts.
    with imageio.get_writer(output_path, fps=fps, quality=8) as writer:
        for frame in video_frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)

image_path = "./data/test_case/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pretrained_model_path = os.environ.get("INFINITE_WORLD_MODEL_PATH", "MeiGen-AI/Infinite-World")

pipeline = InfiniteWorldPipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
)

output_video = pipeline(
    images=input_image,
    prompt="A serene campus walkway lined with modern glass buildings and soft daylight.",
    interactions=["forward", "forward+camera_r", "forward", "camera_l"],
    num_frames=80,
    size=(384, 1024),
)

# export_to_video(output_video, "infinite_world_demo.mp4", fps=30)


# Save uint8 frames directly to avoid diffusers.export_to_video color overflow.
save_uint8_video(output_video, "infinite_world_demo.mp4", fps=30)