#!/usr/bin/env bash
# install_cuda_ubuntu.sh — Linux Mint 22.3 / Ubuntu 24.04 helper.
#
# Two paths to install nvcc:
#
# (A) Distro package — older but easy:
#     sudo apt update
#     sudo apt install -y nvidia-cuda-toolkit
#
# (B) NVIDIA network repo — newest CUDA (recommended for RTX 4090, sm_89):
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#     sudo dpkg -i cuda-keyring_1.1-1_all.deb
#     sudo apt update
#     sudo apt install -y cuda-toolkit-13-0
#     # Then add nvcc to PATH:
#     echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
#     echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
#     source ~/.bashrc
#
# This script does NOT auto-sudo. Run the commands above manually after
# reading them.

set -euo pipefail
echo "Read this file. Do not pipe to bash. Pick path (A) or (B) and run by hand."
echo "Once nvcc is on PATH:  rm -rf build-linux && ./scripts/build.sh"
