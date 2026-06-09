#!/bin/bash
# Sana-WM environment setup for OpenWorldLib
# This script configures the conda environment for Sana-WM inference.
# Usage: bash scripts/setup/sana_wm_install.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== [2/3] Installing the requirements ==="
pip install -e ".[transformers_high]"
pip install termcolor flash-linear-attention==0.4.2 timm setuptools==79.0.1 pytz omegaconf
pip install mmcv==1.7.2 --no-build-isolation

echo "=== [3/3] Installing the flash attention ==="
pip install flash-attn --no-build-isolation

echo "=== Setup completed! ==="

