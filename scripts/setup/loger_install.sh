#!/bin/bash
# scripts/setup/install_logger.sh
# Description: Install logger / visualization dependencies
# Usage: bash scripts/setup/install_logger.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.6.0 torchvision==0.21.0

echo "=== [2/3] Installing the requirements ==="
pip install \
    numpy==1.26.4 \
    pillow \
    opencv-python \
    plyfile \
    huggingface_hub \
    safetensors \
    natsort \
    einops \
    gradio \
    trimesh \
    matplotlib \
    scipy \
    viser \
    roma \
    evo \
    accelerate

echo "=== [3/3] Installing the logger extra dependencies ==="
echo "=== Setup completed! ==="