#!/usr/bin/env bash
# Train the three GS-FUSE variants one after another with identical settings. Run from the repo root,
# inside tmux so the job survives SSH disconnects:
#   MODE=smoke bash scripts/run_variants.sh   # few samples, checks the GPU pipeline + timing
#   bash scripts/run_variants.sh              # full data
# Env overrides: VARIANTS="baseline moe qformer", BATCH_SIZE, EPOCHS (all three stages),
#                WANDB_MODE=online|offline|disabled, WANDB_PROJECT, WANDB_ENTITY, OUT_ROOT, INTERPRET_SAMPLES,
#                TEXT_CACHE=<file from scripts/precompute_text_cache.py>, TF32=1, QUANT_4BIT=1 (both must match
#                how the cache was built; QUANT_4BIT defaults to 0, the fp32 setting of the midterm runs),
#                SW_STRIDE (stage-1 sliding-window stride, default 1 as in the midterm runs)
set -euo pipefail

source "$(dirname "$0")/conda_env.sh"
set +u; conda activate gs-fuse; set -u
export PYTHONNOUSERSITE=1

MODE=${MODE:-full}
VARIANTS=${VARIANTS:-"baseline moe qformer"}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-2}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-11785-GS-Fuse}
OUT_ROOT=${OUT_ROOT:-outputs/${MODE}}
INTERPRET_SAMPLES=${INTERPRET_SAMPLES:-200}

common=(--moment-path pretrained/MOMENT-1-large --text-model-path pretrained/Llama-3.2-1B
        --event-dir data/event --series-dir data/series
        --batch-size "$BATCH_SIZE")
if [ -n "${TEXT_CACHE:-}" ]; then common+=(--text-cache "$TEXT_CACHE"); fi
if [ "${TF32:-0}" = 1 ]; then common+=(--tf32); fi
if [ "${QUANT_4BIT:-0}" != 1 ]; then common+=(--no-4bit); fi
if [ "$WANDB_MODE" != "disabled" ]; then
  common+=(--wandb --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" --wandb-tags "$MODE")
  if [ -n "${WANDB_ENTITY:-}" ]; then common+=(--wandb-entity "$WANDB_ENTITY"); fi
fi

for v in $VARIANTS; do
  case $v in
    baseline) flags=(--fusion-type crossattn --gate-type softmax) ;;
    moe)      flags=(--fusion-type crossattn --gate-type moe) ;;
    qformer)  flags=(--fusion-type qformer   --gate-type softmax) ;;
    *) echo "unknown variant: $v"; exit 1 ;;
  esac
  out="$OUT_ROOT/$v"
  mkdir -p "$out"
  echo "==== [$(date '+%F %T')] $v -> $out ===="
  start=$(date +%s)
  if [ "$MODE" = smoke ]; then
    python -u smoke_train.py "${common[@]}" "${flags[@]}" --output-path "$out" \
      --wandb-name "smoke_$v" --skip-checkpoints --n-event-train 64 --n-event-eval 32 \
      2>&1 | tee "$out/console.log"
  else
    python -u train.py "${common[@]}" "${flags[@]}" --output-path "$out" \
      --wandb-name "$v" --num-epochs "$EPOCHS" --stage1-epochs "$EPOCHS" --stage2-epochs "$EPOCHS" \
      --sw-stride "${SW_STRIDE:-1}" \
      --interpret-samples "$INTERPRET_SAMPLES" 2>&1 | tee "$out/console.log"
  fi
  echo "==== $v finished in $(( ($(date +%s) - start) / 60 )) min ===="
done
