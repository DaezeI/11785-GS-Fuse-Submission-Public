#!/bin/bash
#SBATCH --job-name=nicole-c1-smoke
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/c1-smoke-%j.out

set -euo pipefail
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /work/hdd/bjjg/gs-fuse/nicole/c1-nov10/repo
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -u smoke_train.py \
  --c1-manifest splits/manifests/purged_split_c1_macro_eras.json \
  --c1-timing-policy category4_1400_strict_before \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 2 \
  --n-event-train 16 --n-event-eval 8 \
  --n-sw-train 32 --n-sw-eval 16 --n-interpret 0 \
  --stage1-epochs 1 --stage2-epochs 1 --num-epochs 1 \
  --output-path /work/hdd/bjjg/gs-fuse/nicole/c1-nov10/smoke/run1 \
  --wandb --wandb-mode offline --wandb-name nicole-c1-smoke-2026-11-10
