import torch
from PIL import Image

from openworldlib.pipelines.cambrian_s.pipeline_cambrian_s import CambrianSPipeline


MODEL_PATH = "nyu-visionx/Cambrian-S-0.5B"
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16

IMAGE_PATH = "./data/test_case/test_image_case1/ref_image.png"
VIDEO_PATH = "./data/test_case/test_video_case1/talking_man.mp4"


def test_cambrian_s_pipeline_pil_image():
    pipe = CambrianSPipeline.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        weight_dtype=WEIGHT_DTYPE,
    )
    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    instruction = "Describe the scene."
    output = pipe(
        prompt=instruction,
        images=pil_image,
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print("[PIL.Image] output:", output[0])


def test_cambrian_s_pipeline_video():
    pipe = CambrianSPipeline.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        weight_dtype=WEIGHT_DTYPE,
    )
    instruction = "Summarize the video content."
    output = pipe(
        prompt=instruction,
        videos=VIDEO_PATH,
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print("[video] output:", output[0])


if __name__ == "__main__":
    test_cambrian_s_pipeline_pil_image()
    test_cambrian_s_pipeline_video()
