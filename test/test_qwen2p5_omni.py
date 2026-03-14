from openworldlib.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
import soundfile as sf
from PIL import Image
from decord import VideoReader
import numpy as np

#Model path
model_path = "Qwen/Qwen2.5-Omni-7B"

# Prepare inputs
prompt = "Describe this video"
return_audio = False

# Process image
image_path = "./data/test_case/test_image_case1/ref_image.png"
images = Image.open(image_path)
if images.mode == 'RGBA':
    background = Image.new('RGB', images.size, (0, 0, 0))
    background.paste(images, mask=images.split()[3])
    images = background

# Process video
video_path = "./data/test_video_case1/talking_man.mp4"
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

# Process audio (if needed)
audio_path = None  # e.g., "./data/test_audio.wav"
if audio_path is not None:
    audio_data, sample_rate = sf.read(audio_path)
    audios = (audio_data, sample_rate)  # tuple of (numpy array, sample_rate)
else:
    audios = None

# Initialize pipeline
pipeline = Qwen2p5OmniPipeline.from_pretrained(
    pretrained_model_path=model_path,
    use_audio_in_video=False,
)

# Run inference
if return_audio:
    text, audio = pipeline(
        prompt=prompt,
        images=None,  # Can be PIL Image or list of PIL Images
        videos=videos,  # List of PIL Images from video
        audios=audios,
        return_audio=return_audio
    )
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
else:
    text = pipeline(
        prompt=prompt,
        images=None,  # Can be PIL Image or list of PIL Images
        videos=videos,  # List of PIL Images from video
        audios=audios,
        return_audio=return_audio
    )
print(text)
