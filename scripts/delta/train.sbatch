#!/bin/bash
#SBATCH --job-name=gsfuse-train
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/logs/%x-%j.out

# Train one variant. Submit with:
#   sbatch --export=ALL,VARIANT=baseline scripts/delta/train.sbatch
# VARIANT is one of: baseline moe qformer
set -euo pipefail
mkdir -p /work/hdd/bjjg/gs-fuse/runs/logs
# shellcheck disable=SC1091
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh

VARIANT="${VARIANT:-baseline}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WANDB_MODE="${WANDB_MODE:-offline}"
TEXT_CACHE="${TEXT_CACHE:-cache/text_inst_llama1b_tf32.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/full}"

case "$VARIANT" in
  baseline) flags=(--fusion-type crossattn --gate-type softmax) ;;
  moe)      flags=(--fusion-type crossattn --gate-type moe) ;;
  qformer)  flags=(--fusion-type qformer   --gate-type softmax) ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

test -f "$TEXT_CACHE" || { echo "missing text cache $TEXT_CACHE — submit precompute.sbatch first"; exit 1; }
test -d pretrained/MOMENT-1-large || { echo "missing MOMENT weights"; exit 1; }

out="$OUT_ROOT/$VARIANT"
mkdir -p "$out"
echo "==== $(date) $VARIANT -> $out wandb=$WANDB_MODE ===="

python -u train.py \
  --moment-path pretrained/MOMENT-1-large \
  --text-model-path pretrained/Llama-3.2-1B \
  --event-dir data/event \
  --series-dir data/series \
  --batch-size "$BATCH_SIZE" \
  --text-cache "$TEXT_CACHE" \
  --tf32 --no-4bit \
  "${flags[@]}" \
  --output-path "$out" \
  --num-epochs "$EPOCHS" \
  --stage1-epochs "$EPOCHS" \
  --stage2-epochs "$EPOCHS" \
  --interpret-samples 200 \
  --wandb --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$VARIANT" --wandb-tags "delta,full" \
  2>&1 | tee "$out/console.log"

echo "==== $(date) $VARIANT done. checkpoint: $out/best_model.pth ===="
ls -lh "$out"/*.pth
