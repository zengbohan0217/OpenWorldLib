from openworldlib.pipelines.omnivinci.pipeline_omnivinci import OmniVinciPipeline
from PIL import Image
from decord import VideoReader
import numpy as np
import soundfile as sf
import os

#  Model path
model_path = "nvidia/omnivinci"

# Prepare inputs
prompt = "Describe this image"

# Process image
image_path = "./data/test_case/test_image_case1/ref_image.png"
images = Image.open(image_path)
if images.mode == 'RGBA':
    background = Image.new('RGB', images.size, (0, 0, 0))
    background.paste(images, mask=images.split()[3])
    images = background
elif images.mode != 'RGB':
    images = images.convert('RGB')

# Process video (if needed)
video_path = None  # e.g., "./data/test_video.mp4"
if video_path is not None:
    video_reader = VideoReader(video_path)
    total_frames_target = 33
    start_idx = 0

    # Sample frames from the video
    target_times = np.arange(total_frames_target) / 30
    original_indices = np.round(target_times * 30).astype(int)
    batch_index = [idx + start_idx for idx in original_indices]
    if len(batch_index) < total_frames_target:
        batch_index = batch_index[:total_frames_target]

    videos = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index]
else:
    videos = None

# Process audio (if needed)
audio_path = None  # e.g., "./data/test_audio.wav"
if audio_path is not None:
    audio_data, sample_rate = sf.read(audio_path)
    audios = (audio_data, sample_rate)
else:
    audios = None

# Initialize pipeline with OmniVinci-specific parameters
pipeline = OmniVinciPipeline.from_pretrained(
    pretrained_model_path=model_path,
    load_audio_in_video=True,
    num_video_frames=128,      # Number of frames to extract from video
)

# Run inference
text = pipeline(
    prompt=prompt,
    images=images,
    videos=videos,
    audios=audios
)

print(text)

