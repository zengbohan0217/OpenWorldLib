#!/bin/bash
# scripts/setup/cambrian_s_install.sh
# Description: Setup environment for Cambrian-S inference in OpenWorldLib
# Usage: bash scripts/setup/cambrian_s_install.sh

echo "=== [1/4] Installing the base environment ==="
pip install torch==2.6.9 torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git

echo "=== [2/4] Installing the OpenWorldLib requirements (transformers_low extra) ==="
pip install -e ".[transformers_low]"

echo "=== [3/4] Installing Cambrian-S runtime dependencies ==="
pip install sentencepiece decord

echo "=== [4/4] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="
