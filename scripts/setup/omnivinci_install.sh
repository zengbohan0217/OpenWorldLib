#!/bin/bash
# scripts/setup/omnivinci_install.sh
# Description: Setup environment for OmniVinci model dependencies
# Usage: bash scripts/setup/omnivinci_install.sh

echo "=== [1/4] Installing core model dependencies ==="
pip install torch==2.5.1 torchvision torchaudio
pip install bitsandbytes==0.43.2 einops-exts==0.0.4

echo "=== [2/4] Installing vision and video dependencies ==="
pip install opencv-python-headless==4.8.0.76 pytorchvideo==0.1.5

echo "=== [3/4] Installing audio dependencies ==="
pip install -e ".[audio_default]"
pip install openai-whisper kaldiio

echo "=== [4/4] Installing utility dependencies ==="
pip install requests beartype
pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

echo "=== Setup completed! ==="
