from openworldlib.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
from PIL import Image
from decord import VideoReader
import numpy as np
import soundfile as sf
import os

# Model configuration
model_path = "Qwen/Qwen2.5-Omni-7B"
pipeline = Qwen2p5OmniPipeline.from_pretrained(
    pretrained_model_path=model_path,
    use_audio_in_video=False,
)

# Supported input types
AVAILABLE_INPUTS = {
    "prompt": "Text prompt",
    "image": "Image file path",
    "audio": "Audio file path",
    "video": "Video file path"
}

print("=== Qwen2.5-Omni Interactive Stream ===")
print("\nAvailable input types:")
for input_type, description in AVAILABLE_INPUTS.items():
    print(f"  - {input_type}: {description}")
print("\nTips:")
print("  - Each turn can include prompt, image, audio, or video inputs")
print("  - Input 'reset' to clear conversation history")
print("  - Input 'quit' or 'q' to stop and exit")
print("  - Leave input empty to skip that modality")

print("\n--- Interactive Stream Started ---")
turn_idx = 0

while True:
    print(f"\n{'='*50}")
    print(f"[Turn {turn_idx}]")
    print(f"{'='*50}")
    
    # Check for special commands
    command = input("Enter command (or press Enter to continue): ").strip().lower()
    
    if command in ['quit', 'q']:
        print("Exiting interactive stream...")
        break
    elif command == 'reset':
        pipeline.memory_module.manage(action="reset")
        print("Conversation history cleared")
        turn_idx = 0
        continue
    
    # Collect inputs for this turn
    prompt_input = input("Text prompt: ").strip()
    if not prompt_input:
        prompt_input = None
    
    # Process image input
    image_path = input("Image path (or press Enter to skip): ").strip()
    if image_path and os.path.exists(image_path):
        try:
            images = Image.open(image_path)
            if images.mode == 'RGBA':
                background = Image.new('RGB', images.size, (0, 0, 0))
                background.paste(images, mask=images.split()[3])
                images = background
            elif images.mode != 'RGB':
                images = images.convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            images = None
    else:
        if image_path:
            print(f"Image file not found: {image_path}")
        images = None
    
    # Process audio input
    audio_path = input("Audio path (or press Enter to skip): ").strip()
    if audio_path and os.path.exists(audio_path):
        try:
            audio_data, sample_rate = sf.read(audio_path)
            audios = (audio_data, sample_rate)
        except Exception as e:
            print(f"Error loading audio: {e}")
            audios = None
    else:
        if audio_path:
            print(f"Audio file not found: {audio_path}")
        audios = None
    
    # Process video input
    video_path = input("Video path (or press Enter to skip): ").strip()
    if video_path and os.path.exists(video_path):
        try:
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
        except Exception as e:
            print(f"Error loading video: {e}")
            videos = None
    else:
        if video_path:
            print(f"Video file not found: {video_path}")
        videos = None
    
    # Check if any input provided
    if not any([prompt_input, images, audios, videos]):
        print("No input provided. Please provide at least one input.")
        continue

    # Optional: return_audio
    return_audio_str = input("Return audio? (y/n, default: n): ").strip().lower()
    return_audio = return_audio_str == 'y'

    print(f"\nProcessing Turn {turn_idx}...")

    try:
        # Call stream method
        if return_audio:
            result, audio = pipeline.stream(
                prompt=prompt_input,
                images=images,
                audios=audios,
                videos=videos,
                use_history=True,
                return_audio=return_audio,
                reset_memory=False
            )
            print(f"\nResponse: {result}")

            # Save audio if generated
            audio_output_path = f"output_turn_{turn_idx}.wav"
            sf.write(
                audio_output_path,
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            print(f"Audio saved to: {audio_output_path}")
        else:
            result = pipeline.stream(
                prompt=prompt_input,
                images=images,
                audios=audios,
                videos=videos,
                use_history=True,
                return_audio=return_audio,
                reset_memory=False
            )
            print(f"\nResponse: {result}")
        
        turn_idx += 1
        print(f"\nTotal conversation turns: {len(pipeline.memory_module.storage)}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n=== Stream Session Ended ===")
print(f"Total turns: {turn_idx}")
print(f"Memory storage length: {len(pipeline.memory_module.storage)}")
