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

turns = [
    ["forward", "forward+camera_r"],
    ["camera_l", "forward"],
    ["forward_left", "camera_r"],
]

for turn_idx, actions in enumerate(turns):
    start_image = input_image if turn_idx == 0 else None
    video_output = pipeline.stream(
        images=start_image,
        prompt="A serene campus walkway lined with modern glass buildings and soft daylight.",
        interactions=actions,
        num_frames=40,
        size=(384, 1024),
    )
    print(f"Turn {turn_idx}: generated {len(video_output)} frames")

# export_to_video(pipeline.memory_module.all_frames, "infinite_world_stream_demo.mp4", fps=30)

save_uint8_video(pipeline.memory_module.all_frames, "infinite_world_stream_demo.mp4", fps=30)
