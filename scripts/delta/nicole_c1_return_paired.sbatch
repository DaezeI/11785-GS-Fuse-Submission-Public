#!/bin/bash
#SBATCH --job-name=nicole-c1-return-paired
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/logs/return-paired-%j.out

set -euo pipefail
umask 007
: "${RUN_KIND:?set RUN_KIND=smoke-v1 or full-v1}"
case "$RUN_KIND" in smoke-v1|full-v1) ;; *) exit 2 ;; esac
MODE=${MODE:-zeroed_article_tokens}
case "$MODE" in zeroed_article_tokens|fusion_bypass_ts_inst) ;; *) exit 2 ;; esac
RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1
RUN_DIR="$RUN_ROOT/$RUN_KIND"
OUTPUT="$RUN_ROOT/paired-$RUN_KIND"
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
test -s "$RUN_DIR/best_model.pth"
test -s "$RUN_DIR/resolved_config.json"
mkdir -p "$OUTPUT"
test ! -e "$OUTPUT/$MODE.npz"
git rev-parse HEAD

MAX_EVENTS=0
if [ "$RUN_KIND" = smoke-v1 ]; then MAX_EVENTS=8; fi
python -u scripts/run_text_controls.py \
  --forecast-task return_bp \
  --task-config-json "$RUN_DIR/resolved_config.json" \
  --model-path "$RUN_DIR/best_model.pth" \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --fusion-type crossattn --gate-type softmax \
  --max-events "$MAX_EVENTS" --shuffles 3 --seed 23 --batch-size 8 \
  --no-4bit --tf32 --no-text-mode "$MODE" \
  --out "$OUTPUT/$MODE.npz"

python -u scripts/analyze_return_predictions.py \
  --predictions "$OUTPUT/$MODE.npz" \
  --out "$OUTPUT/$MODE.analysis.json"
