#!/bin/bash
#SBATCH --job-name=adaeze-return-metrics
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/logs/return-metrics-replay-%j.out

set -euo pipefail
umask 007

RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8
REPO_ROOT="$RUN_ROOT/repo"
SOURCE_RUN="$RUN_ROOT/smoke-speed-v1"
CHECKPOINT="$SOURCE_RUN/best_model.pth"
OUTPUT="$RUN_ROOT/metrics-replay-v1"
test ! -e "$OUTPUT"
test -r "$CHECKPOINT"
test -r "$SOURCE_RUN/resolved_config.json"
test -r "$SOURCE_RUN/c1_data_audit.json"

module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd "$REPO_ROOT"
export HF_HOME="$RUN_ROOT/hf-cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
CACHE=/work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt
test -r "$CACHE"
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a

echo "job_id=${SLURM_JOB_ID}"
echo "replay_commit=$(git rev-parse HEAD)"
echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
nvidia-smi

python -u scripts/run_return_metrics_replay.py \
  --source-run "$SOURCE_RUN" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --manifest "$MANIFEST" \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series-dir /work/hdd/bjjg/gs-fuse/data/series \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --text-cache "$CACHE"

find "$OUTPUT" -type d -exec chmod g+rx {} +
find "$OUTPUT" -type f -exec chmod g+r {} +
