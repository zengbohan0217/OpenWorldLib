import torch
from PIL import Image
from openworldlib.pipelines.spatial_ladder.pipeline_spatial_ladder import SpatialLadderPipeline


MODEL_PATH = "hongxingli/SpatialLadder-3B"
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16

IMAGE_PATH = "./data/test_case/test_image_case1/ref_image.png"
VIDEO_PATH = "./data/test_case/test_video_case1/talking_man.mp4"


def load_video_frames(video_path: str, max_frames: int = 8):
    """Uniformly sample frames from a video file and return them as a list of PIL.Image."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / max_frames) for i in range(max_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def test_spatial_ladder_pipeline_pil_image():
    """Test image inference using a PIL.Image as input."""
    pipe = SpatialLadderPipeline.from_pretrained(
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


def test_spatial_ladder_pipeline_pil_video():
    """Test video inference using a list of PIL.Image frames as input."""
    pipe = SpatialLadderPipeline.from_pretrained(
        model_path=MODEL_PATH,
        device=DEVICE,
        weight_dtype=WEIGHT_DTYPE,
    )
    frames = load_video_frames(VIDEO_PATH, max_frames=8)
    instruction = "Summarize the video content."
    output = pipe(
        prompt=instruction,
        videos=frames,
        max_new_tokens=64,
    )
    assert isinstance(output, list) and len(output) == 1
    print("[list[PIL.Image]] output:", output[0])


if __name__ == "__main__":
    test_spatial_ladder_pipeline_pil_image()
    test_spatial_ladder_pipeline_pil_video()
