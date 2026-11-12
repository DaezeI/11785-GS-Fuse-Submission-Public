#!/bin/bash
#SBATCH --job-name=nicole-c1-date-screen
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/logs/c1-screen-train-%j.out

set -euo pipefail
umask 007
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/repo
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# This is a date-screened timing-sensitivity run, not a verified release-time benchmark.
# Use a fresh output directory so old Stage-1/2 checkpoints cannot resume.
MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
python -u train.py \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --c1-allow-resume \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 8 --sw-stride 10 \
  --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 \
  --fusion-type crossattn --gate-type softmax \
  --output-path /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/full_sensitivity_v1 \
  --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
  --wandb-name nicole-c1-date-screen-v1-2026-11-12
