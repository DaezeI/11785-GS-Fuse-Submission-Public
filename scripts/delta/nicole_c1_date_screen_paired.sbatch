#!/bin/bash
#SBATCH --job-name=nicole-c1-screen-paired
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/logs/c1-screen-paired-%x-%j.out

set -euo pipefail
umask 007
: "${MODE:?submit with MODE=zeroed_article_tokens or MODE=fusion_bypass_ts_inst}"
case "$MODE" in
  zeroed_article_tokens|fusion_bypass_ts_inst) ;;
  *) echo "Unexpected MODE=$MODE" >&2; exit 2 ;;
esac

module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/repo
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CHECKPOINT=${CHECKPOINT:-/work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/full_sensitivity_v1/best_model.pth}
OUTPUT=${OUTPUT:-/work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/paired_sensitivity_v1}
TRAINING_SPLIT=${TRAINING_SPLIT:-c1_date_screen_v1_category4_1400}
MAX_EVENTS=${MAX_EVENTS:-0}
MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT"
test ! -e "$OUTPUT/$MODE.npz"

python -u scripts/run_text_controls.py \
  --model-path "$CHECKPOINT" \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --checkpoint-training-split "$TRAINING_SPLIT" \
  --max-events "$MAX_EVENTS" --shuffles 3 --seed 23 --batch-size 8 \
  --no-4bit --tf32 --no-text-mode "$MODE" \
  --out "$OUTPUT/$MODE.npz"

python -u scripts/analyze_text_predictions.py \
  --predictions "$OUTPUT/$MODE.npz" \
  --out "$OUTPUT/$MODE.analysis.json"
