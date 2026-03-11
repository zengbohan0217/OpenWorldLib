import sys

sys.path.append("..")

from src.sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = "/YOUR/IMAGE/OR/DIRECTORY/PATH"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./vggt_stream_output"

POINT_CONF_THRESHOLD = 0.2
RESOLUTION = 518
PREPROCESS_MODE = "crop"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 480
FPS = 12


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

# test 仅通过 stream 使用 pipeline，高层任务由 task_type 指定
output_video_path = pipeline.stream(
    DATA_PATH,
    task_type="vggt_two_stage_3dgs_stream_cli",
    output_dir=OUTPUT_DIR,
    point_conf_threshold=POINT_CONF_THRESHOLD,
    resolution=RESOLUTION,
    preprocess_mode=PREPROCESS_MODE,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    fps=FPS,
    output_name="vggt_stream_demo.mp4",
)

print(f"Rendered VGGT stream video saved to: {output_video_path}")
