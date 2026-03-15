#!/bin/bash
# scripts/setup/libero_install.sh
# Description: Setup environment for LIBERO installation of OpenWorldLib
# Usage: bash scripts/setup/libero_install.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.6.0 torchvision torchaudio

echo "=== [2/2] Installing the requirements ==="
pip install hydra-core==1.2.0 robomimic==0.2.0 thop==0.1.1-2209072238 robosuite==1.5.0 bddl==1.0.1 future==0.18.2 cloudpickle==2.1.0 gym==0.25.2
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git

echo "=== Setup completed! ==="
