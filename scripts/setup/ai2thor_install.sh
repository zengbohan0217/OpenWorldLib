#!/usr/bin/env bash
# scripts/setup/install_ai2thor.sh
# Description: Install AI2-THOR Unity build + Python deps into submodules/thor
# Usage: bash scripts/setup/install_ai2thor.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TARGET_DIR="$PROJECT_ROOT/submodules/ai2thor"
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
ZIP_NAME="thor-Linux64.zip"

echo "=== [1/3] Installing the requirements ==="
pip install \
    "Flask==2.0.1" "Werkzeug==2.0.1" \
    numpy pyyaml requests progressbar2 \
    botocore aws-requests-auth compress_pickle objathor \
    Pillow opencv-python "python-xlib==0.21" "msgpack==1.1.2"

echo "=== [2/3] Downloading and extracting AI2-THOR build ==="
mkdir -p "$TARGET_DIR" && cd "$TARGET_DIR"

if [[ ! -f "$ZIP_NAME" ]]; then
    wget -O "$ZIP_NAME" "$URL"
fi
unzip -q -o "$ZIP_NAME" && rm -f "$ZIP_NAME"

echo "=== [3/3] AI2-THOR installation completed ==="
echo "  executable_path = '$TARGET_DIR/thor-Linux64-local/thor-Linux64-local'"
