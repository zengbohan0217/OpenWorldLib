import os

from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline


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

export_to_video(pipeline.memory_module.all_frames, "infinite_world_stream_demo.mp4", fps=30)
