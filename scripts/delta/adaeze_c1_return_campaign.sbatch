#!/bin/bash
#SBATCH --job-name=adaeze-return-campaign
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-campaign-v1/logs/return-campaign-%j.out

set -euo pipefail
umask 007

RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-campaign-v1
REPO_ROOT="$RUN_ROOT/repo"
CAMPAIGN_ROOT="$RUN_ROOT/outputs"
mkdir -p "$CAMPAIGN_ROOT"

module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd "$REPO_ROOT"
export HF_HOME=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
CACHE=/work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt
test -r "$CACHE"
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a

SOURCE_COMMIT=$(git rev-parse HEAD)
if [[ -e "$CAMPAIGN_ROOT/source_commit.txt" ]]; then
  test "$(cat "$CAMPAIGN_ROOT/source_commit.txt")" = "$SOURCE_COMMIT"
else
  printf '%s\n' "$SOURCE_COMMIT" >"$CAMPAIGN_ROOT/source_commit.txt"
fi

echo "job_id=${SLURM_JOB_ID}"
echo "commit=$SOURCE_COMMIT"
echo "manifest_sha256=$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
echo "price_sha256=$(sha256sum "$PRICE" | cut -d' ' -f1)"
nvidia-smi

launch_variant() {
  local name=$1
  local direction_weight=$2
  local learnable_last_value=$3
  local output="$CAMPAIGN_ROOT/$name"
  local completed="$output/COMPLETED"
  local last_value_flag=--learnable-last-value
  if [[ "$learnable_last_value" == false ]]; then
    last_value_flag=--no-learnable-last-value
  fi

  if [[ -e "$completed" ]]; then
    echo "[$name] already completed; skipping"
    return 0
  fi
  mkdir -p "$CAMPAIGN_ROOT/wandb-$name"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=40G --gpus-per-task=1 --gpu-bind=verbose,per_task:1 \
    env WANDB_DIR="$CAMPAIGN_ROOT/wandb-$name" \
    python -u train.py \
      --forecast-task return_bp \
      --raw-ts-residual "$last_value_flag" \
      --direction-weight "$direction_weight" \
      --c1-manifest "$MANIFEST" \
      --c1-timing-policy category4_1400_strict_before \
      --c1-allow-resume \
      --event-dir /work/hdd/bjjg/gs-fuse/data/event \
      --series-dir /work/hdd/bjjg/gs-fuse/data/series \
      --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
      --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
      --text-cache "$CACHE" \
      --no-4bit --tf32 --batch-size 8 --sw-stride 10 \
      --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 \
      --fusion-type crossattn --gate-type softmax \
      --output-path "$output" \
      --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
      --wandb-name "adaeze-c1-return-$name-campaign-v1" \
      >"$CAMPAIGN_ROOT/$name.log" 2>&1
  printf 'job_id=%s\ncommit=%s\n' "$SLURM_JOB_ID" "$SOURCE_COMMIT" >"$completed"
}

launch_variant baseline-lambda1 1 true &
pid_baseline=$!
launch_variant lambda025 0.25 true &
pid_lambda025=$!
launch_variant lambda4 4 true &
pid_lambda4=$!
launch_variant no-last-value 1 false &
pid_no_last_value=$!

status=0
for pid in "$pid_baseline" "$pid_lambda025" "$pid_lambda4" "$pid_no_last_value"; do
  if ! wait "$pid"; then
    status=1
  fi
done

find "$RUN_ROOT" -type d -exec chmod g+rx {} +
find "$RUN_ROOT" -type f -exec chmod g+r {} +
exit "$status"
