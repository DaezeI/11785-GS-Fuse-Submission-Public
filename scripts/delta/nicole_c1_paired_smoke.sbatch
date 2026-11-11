#!/bin/bash
#SBATCH --job-name=nicole-c1-paired
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/c1-paired-%j.out

set -euo pipefail
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /work/hdd/bjjg/gs-fuse/nicole/c1-nov10/repo
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CHECKPOINT=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/smoke/run1/best_model.pth
OUTPUT=/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/paired/run1
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT"

for MODE in zeroed_article_tokens fusion_bypass_ts_inst; do
  python -u scripts/run_text_controls.py \
    --model-path "$CHECKPOINT" \
    --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
    --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
    --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
    --event-dir /work/hdd/bjjg/gs-fuse/data/event \
    --series-dir /work/hdd/bjjg/gs-fuse/data/series \
    --c1-manifest splits/manifests/purged_split_c1_macro_eras.json \
    --c1-timing-policy category4_1400_strict_before \
    --checkpoint-training-split c1_smoke_16_training_events \
    --max-events 8 --shuffles 3 --seed 23 --batch-size 2 \
    --no-4bit --tf32 --no-text-mode "$MODE" \
    --out "$OUTPUT/$MODE.npz"
  python -u scripts/analyze_text_predictions.py \
    --predictions "$OUTPUT/$MODE.npz" \
    --out "$OUTPUT/$MODE.analysis.json"
done
