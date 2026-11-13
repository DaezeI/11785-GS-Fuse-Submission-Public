#!/bin/bash
#SBATCH --job-name=nicole-c1-return-smoke
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/logs/return-smoke-%j.out

set -euo pipefail
umask 007
RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1
OUTPUT="$RUN_ROOT/smoke-v1"
test ! -e "$OUTPUT"
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd "$RUN_ROOT/repo"
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
git rev-parse HEAD

python -u smoke_train.py \
  --forecast-task return_bp \
  --raw-ts-residual --learnable-last-value \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 2 \
  --n-event-train 16 --n-event-eval 8 --n-sw-train 32 --n-sw-eval 16 \
  --n-interpret 0 --stage1-epochs 1 --stage2-epochs 1 --num-epochs 1 \
  --fusion-type crossattn --gate-type softmax \
  --output-path "$OUTPUT" \
  --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
  --wandb-name nicole-c1-return-smoke-v1
