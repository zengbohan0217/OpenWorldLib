"""Sana-WM basic inference test.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python test/test_sana_wm.py

Usage (no GPU — dry-run only, very slow):
    python test/test_sana_wm.py
"""

import os
import torch
from PIL import Image
from diffusers.utils import export_to_video
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.openworldlib.pipelines.sana_wm.pipeline_sana_wm import SanaWMPipeline
# from .pipelines.sana_wm.pipeline_sana_wm import SanaWMPipeline


image_path = "/home/dataset-assist-0/usr/lh/hdl/Sana-main/asset/sana_wm/demo_0.png"
pretrained_model_path = "/home/dataset-assist-0/usr/lh/hdl/sana-ckpts"
gemma_path = "/home/dataset-assist-0/usr/lh/hdl/gemma"
input_image = Image.open(image_path).convert("RGB")
prompt = "A first-person view from a strictly stationary observation point across an immense dry lakebed bordered by low mountain ranges. A black sports car occupies the central foreground on the pale, compacted surface, aligned toward the open horizon beneath a vast blue sky. The environment is broad and minimal, with flat beige desert crust, faint tire-worn texture, distant rocky ridgelines, and a few thin clouds stretching across the upper sky. Bright midday sunlight creates crisp shadows under the vehicle and a clean, high-visibility atmosphere of speed, openness, and isolation. The observer�s perspective remains fixed, with no dynamic camera movement and no actions taken by the person recording. Autonomous motion belongs to the world itself: dust trails sweep low across the ground, heat haze shimmers near the horizon, clouds drift slowly, and the car�s tires kick up fine desert grit."

pipeline = SanaWMPipeline.from_pretrained(
    model_path=pretrained_model_path,
    text_encoder_path=gemma_path,
    device="cuda",
    offload_vae=False,
    offload_refiner=False,
    enable_refiner=False,
)

# codeflicker-fix: API-Issue-007/nrcp8vjpoyrpfjxql0ey
# Add explicit generation parameters (follow upstream defaults).
action_commands = ["w-80","jw-40","w-40","lw-60","w-100"]
output_video = pipeline(
    images=input_image,
    num_frames=321,
    prompt=prompt,
    interactions=action_commands,
    seed=42,
    fps=16,
    step=60,
    cfg_scale=5.0,
    sampling_algo="flow_euler_ltx",
    negative_prompt="",
)

if output_video is not None:
    export_to_video([frame / 255.0 for frame in output_video], "sana_wm_output.mp4", fps=16)
    print("Done! Video saved.")