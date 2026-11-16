#!/bin/bash
#SBATCH --job-name=gsfuse-c1-smoke
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=expanse-%x-%j.out

set -euo pipefail
umask 007

for name in EXPANSE_PROJECT REPO_ROOT RUN_ROOT EVENT_DIR SERIES_DIR MOMENT_PATH TEXT_MODEL_PATH TEXT_CACHE ENV_ACTIVATE; do
  [[ -n "${!name:-}" ]] || { echo "Set $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ACCOUNT:-}" == "$EXPANSE_PROJECT" ]] || { echo "Wrong Slurm account" >&2; exit 2; }

type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load gpu
module load slurm
source "$ENV_ACTIVATE"
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-$RUN_ROOT/hf-cache}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MANIFEST="${MANIFEST:-$REPO_ROOT/splits/manifests/purged_split_c1_date_screen_v1.json}"
PRICE="$SERIES_DIR/SP500.csv"
OUTPUT="${OUTPUT:-$RUN_ROOT/smoke}"
PRECISION_MODE="${PRECISION_MODE:-4bit}"
BATCH_SIZE="${BATCH_SIZE:-2}"
case "$PRECISION_MODE" in
  4bit) precision_args=() ;;
  fp32) precision_args=(--no-4bit) ;;
  *) echo "PRECISION_MODE must be 4bit or fp32" >&2; exit 2 ;;
esac

test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
for checkpoint in stage1_best_model.pth stage2_best_model.pth best_model.pth; do
  test ! -e "$OUTPUT/$checkpoint"
done
mkdir -p "$OUTPUT" "$HF_HOME"

echo "job_id=$SLURM_JOB_ID"
echo "account=$SLURM_JOB_ACCOUNT"
echo "code_commit=$(git rev-parse HEAD)"
echo "manifest_sha256=$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
echo "price_sha256=$(sha256sum "$PRICE" | cut -d' ' -f1)"
echo "precision_mode=$PRECISION_MODE"
echo "batch_size=$BATCH_SIZE"
module list 2>&1
nvidia-smi
python --version

python -u smoke_train.py \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --event-dir "$EVENT_DIR" --series-dir "$SERIES_DIR" \
  --moment-path "$MOMENT_PATH" --text-model-path "$TEXT_MODEL_PATH" \
  --text-cache "$TEXT_CACHE" "${precision_args[@]}" --batch-size "$BATCH_SIZE" \
  --n-event-train 16 --n-event-eval 8 \
  --n-sw-train 32 --n-sw-eval 16 --n-interpret 0 \
  --stage1-epochs 1 --stage2-epochs 1 --num-epochs 1 \
  --output-path "$OUTPUT" \
  --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
  --wandb-name "adaeze-expanse-c1-smoke-${PRECISION_MODE}"
