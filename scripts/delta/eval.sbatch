#!/bin/bash
#SBATCH --job-name=gsfuse-eval
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/logs/%x-%j.out

set -euo pipefail
mkdir -p /work/hdd/bjjg/gs-fuse/runs/logs
# shellcheck disable=SC1091
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh

VARIANTS="${VARIANTS:-baseline moe qformer}"
mkdir -p outputs/eval
for v in $VARIANTS; do
  case $v in
    baseline) flags=(--fusion-type crossattn --gate-type softmax) ;;
    moe)      flags=(--fusion-type crossattn --gate-type moe) ;;
    qformer)  flags=(--fusion-type qformer --gate-type softmax) ;;
  esac
  ckpt="outputs/full/$v/best_model.pth"
  test -f "$ckpt" || { echo "skip $v, no $ckpt"; continue; }
  for split in test vali; do
    python -u test_pretrained_model.py \
      --model-path "$ckpt" "${flags[@]}" \
      --text-model-path pretrained/Llama-3.2-1B \
      --moment-path pretrained/MOMENT-1-large \
      --series-id SP500 --batch-size 8 --split "$split" --feature-idx 3 \
      --text-cache cache/text_inst_llama1b_tf32.pt --tf32 --no-4bit \
      --output-json "outputs/eval/${v}_${split}.json" \
      --save-preds "outputs/eval/${v}_${split}_preds.npz" \
      2>&1 | tee "outputs/eval/${v}_${split}.log"
    echo "$v $split exit ${PIPESTATUS[0]}"
  done
done
