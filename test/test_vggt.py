import sys
sys.path.append("..")

from openworldlib.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./vggt_output"

# Interactions follow unified 3D schema, e.g.:
# ["forward", "left", "camera_zoom_in"]
INTERACTION = ["left", "camera_zoom_in"]

# camera_view: [dx, dy, dz, theta_x, theta_z]
CAMERA_VIEW = None

POINT_CONF_THRESHOLD = 0.2
RESOLUTION = 518
PREPROCESS_MODE = "crop"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 480
FPS = 12


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

output_video_path = pipeline(
    DATA_PATH,
    interaction=INTERACTION,
    task_type="vggt_two_stage_3dgs",
    output_dir=OUTPUT_DIR,
    point_conf_threshold=POINT_CONF_THRESHOLD,
    resolution=RESOLUTION,
    preprocess_mode=PREPROCESS_MODE,
    camera_view=CAMERA_VIEW,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    output_name="vggt_3dgs_demo.mp4",
    fps=FPS,
)

print(f"Rendered VGGT 3DGS video saved to: {output_video_path}")

