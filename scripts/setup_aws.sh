#!/usr/bin/env bash
# ==============================================================================
# CMU 11-785 Intro to Deep Learning: GS-Fuse AWS Cloud Environment Setup Script
# ==============================================================================
# Target Instance: AWS EC2 g5.xlarge (NVIDIA A10G 24GB VRAM)
# Target OS: Ubuntu 22.04 LTS (Deep Learning AMI or Standard Ubuntu 22.04)
# ==============================================================================

set -e

echo "=========================================================="
echo " Starting Automated GS-Fuse AWS Cloud Environment Setup"
echo "=========================================================="

# 1. Update system packages
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git unzip curl build-essential

# 2. Check NVIDIA GPU
echo "[2/6] Checking NVIDIA GPU and drivers..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not detected! Please ensure you selected an NVIDIA GPU instance (e.g. g5.xlarge) with GPU drivers installed."
fi

# 3. Create and activate Python Virtual Environment
echo "[3/6] Configuring isolated Python virtual environment..."
python3 -m venv ~/venv_gsfuse
source ~/venv_gsfuse/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install CUDA-enabled PyTorch
echo "[4/6] Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install Project Dependencies & MLOps Tools
echo "[5/6] Installing GS-Fuse dependencies..."
pip install \
    transformers \
    accelerate \
    bitsandbytes \
    momentfm \
    wandb \
    einops \
    scikit-learn \
    scipy \
    matplotlib \
    pandas \
    tqdm \
    gdown \
    jupyterlab

# 6. Verify PyTorch CUDA Acceleration
echo "[6/6] Verifying PyTorch CUDA acceleration..."
python3 -c "
import torch
print('='*50)
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:    ', torch.cuda.get_device_name(0))
    print('VRAM Available: ', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB')
print('='*50)
"

echo "=========================================================="
echo " AWS Cloud Environment Setup Complete!"
echo " Activate environment with: source ~/venv_gsfuse/bin/activate"
echo "=========================================================="
