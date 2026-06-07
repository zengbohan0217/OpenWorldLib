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
from openworldlib.pipelines.sana_wm.pipeline_sana_wm import SanaWMPipeline


image_path = "./data/test_case/test_image_case1/ref_image.png"
pretrained_model_path = "Efficient-Large-Model/SANA-WM_bidirectional"
gemma_path = "Efficient-Large-Model/gemma-2-2b-it"
input_image: Image = Image.open(image_path).convert("RGB")
prompt = "A charming medieval village with cobblestone streets, thatched-roof houses."

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