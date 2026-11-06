#!/bin/bash
#SBATCH --job-name=gsfuse-cache
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/logs/%x-%j.out

set -euo pipefail
mkdir -p /work/hdd/bjjg/gs-fuse/runs/logs
# shellcheck disable=SC1091
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh

test -d pretrained/Llama-3.2-1B || { echo "missing pretrained/Llama-3.2-1B"; exit 1; }
test -d data/event && test -d data/series || { echo "missing data/event or data/series"; exit 1; }

python -u scripts/precompute_text_cache.py \
  --text-model-path pretrained/Llama-3.2-1B \
  --event-dir data/event \
  --series-dir data/series \
  --out cache/text_inst_llama1b_tf32.pt \
  --tf32

echo "cache done: $(ls -lh cache/text_inst_llama1b_tf32.pt)"
