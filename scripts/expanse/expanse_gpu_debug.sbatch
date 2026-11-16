#!/bin/bash
#SBATCH --job-name=gsfuse-expanse-check
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=expanse-%x-%j.out

set -euo pipefail
umask 007

: "${EXPANSE_PROJECT:?Set EXPANSE_PROJECT and submit with --account}"
: "${ENV_ACTIVATE:?Set ENV_ACTIVATE to the tested environment activation script}"
if [[ "${SLURM_JOB_ACCOUNT:-}" != "$EXPANSE_PROJECT" ]]; then
  echo "Expected account $EXPANSE_PROJECT, got ${SLURM_JOB_ACCOUNT:-unset}" >&2
  exit 2
fi

type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load gpu
module load slurm
source "$ENV_ACTIVATE"

echo "job_id=$SLURM_JOB_ID"
echo "account=$SLURM_JOB_ACCOUNT"
echo "node=$(hostname)"
date -Is
module list 2>&1
nvidia-smi

python - <<'PY'
import platform
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
p = torch.cuda.get_device_properties(0)
print("gpu", p.name)
print("gpu_memory_bytes", p.total_memory)
print("cuda_runtime", torch.version.cuda)
PY
