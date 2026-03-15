#!/bin/bash
# scripts/setup/audio_generation_default_install.sh
# Description: Setup environment for audio generation dependencies of OpenWorldLib
# Usage: bash scripts/setup/audio_generation_default_install.sh

set -euo pipefail

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install git+https://github.com/openai/CLIP.git

echo "=== [2/3] Installing the requirements ==="
pip install -e ".[audio_generation_default]"

echo "=== [3/3] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="

