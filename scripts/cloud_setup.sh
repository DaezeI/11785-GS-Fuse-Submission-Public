#!/usr/bin/env bash
# One-time environment setup on a Linux GPU host. Run from the repo root:
#   bash scripts/cloud_setup.sh
# Mirrors default to mainland-China mirrors because PyPI / repo.anaconda.com are slow from there;
# override with PIP_INDEX=https://pypi.org/simple if the host has direct access.
set -euo pipefail

PIP_INDEX=${PIP_INDEX:-https://mirrors.cloud.tencent.com/pypi/simple}
CONDA_MIRROR=${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

source "$(dirname "$0")/conda_env.sh" 2>/dev/null || {
  echo "== Installing Miniconda =="
  wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash >/dev/null
  source "$(dirname "$0")/conda_env.sh"
}

if ! conda env list | grep -q "^gs-fuse "; then
  echo "== Creating conda env gs-fuse =="
  conda create -y -n gs-fuse python=3.11.9 --override-channels -c "$CONDA_MIRROR"
fi
set +u; conda activate gs-fuse; set -u
export PYTHONNOUSERSITE=1

echo "== Installing Python packages =="
pip install -i "$PIP_INDEX" \
  torch==2.3.0 triton==2.3.0 \
  numpy==1.25.2 pandas==2.2.2 scikit-learn==1.5.0 scipy==1.13.1 matplotlib==3.8.4 \
  transformers==4.43.1 tokenizers==0.19.1 accelerate==1.9.0 huggingface-hub==0.24.0 \
  safetensors==0.4.3 sentencepiece==0.2.0 einops==0.8.1 fsspec==2024.5.0 filelock==3.14.0 \
  tqdm==4.66.4 pyyaml==6.0.1 requests==2.32.3 psutil==6.0.0 pytest wandb==0.31.0
# momentfm pins an older transformers; the versions above are the ones GS-FUSE is tested with.
pip install -i "$PIP_INDEX" --no-deps momentfm==0.1.4

echo "== Sanity checks =="
python - <<'EOF'
import torch, transformers, momentfm, wandb
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__, "| wandb", wandb.__version__)
EOF
python -m pytest -q tests

echo "== Can this host reach wandb.ai? =="
if curl -s -m 8 -o /dev/null -w "%{http_code}\n" https://api.wandb.ai | grep -qE "^[234]"; then
  echo "yes -> run 'wandb login' and train with WANDB_MODE=online"
else
  echo "no  -> train with WANDB_MODE=offline, then sync the wandb/ folders from a machine with access"
fi
