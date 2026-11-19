#!/usr/bin/env bash
# ==============================================================================
# Turnkey Paired Text Evaluation Runner for Return Variants (R0, R1, R2, R3)
# Suitable for standalone GPU execution (H200, A100) or ABCI PBS submission.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Defaults (override by setting environment variables before running)
SNAPSHOT="${SNAPSHOT:-data/snapshots/c1-screened-return-v1}"
MOMENT_PATH="${MOMENT_PATH:-pretrained/MOMENT-1-large}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-pretrained/Llama-3.2-1B}"
TEXT_CACHE="${TEXT_CACHE:-cache/text_inst_llama1b_tf32.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs/adaeze/return-r0-r3-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/h200/paired-return-evaluations}"

# Allow specifying a single variant or evaluating all available
VARIANTS=("${@:-ablation-r0-v1 ablation-r1-v1 ablation-r2-v1}")
MODES=("zeroed_article_tokens" "fusion_bypass_ts_inst")

echo "=========================================================================="
echo "GS-FUSE RETURN VARIANTS: PAIRED TEXT CONTROL EVALUATION"
echo "Code Root       : $ROOT_DIR"
echo "Snapshot Dir    : $SNAPSHOT"
echo "Checkpoints From: $CHECKPOINT_ROOT"
echo "Output Root     : $OUTPUT_ROOT"
echo "Variants        : ${VARIANTS[*]}"
echo "Modes           : ${MODES[*]}"
echo "=========================================================================="

MANIFEST="$SNAPSHOT/manifest.json"
SERIES_DIR="$SNAPSHOT/series"
EVENT_DIR="$SNAPSHOT/event"

for VARIANT in "${VARIANTS[@]}"; do
  VARIANT_DIR="$CHECKPOINT_ROOT/$VARIANT"
  CKPT="$VARIANT_DIR/best_model.pth"
  CONFIG="$VARIANT_DIR/resolved_config.json"

  if [[ ! -f "$CKPT" ]]; then
    echo "WARNING: Checkpoint missing for $VARIANT at $CKPT, skipping."
    continue
  fi

  if [[ ! -f "$CONFIG" ]]; then
    echo "WARNING: Config missing for $VARIANT at $CONFIG, skipping."
    continue
  fi

  echo ""
  echo ">>> Processing Variant: $VARIANT"
  echo "    Checkpoint: $CKPT"

  OUT_DIR="$OUTPUT_ROOT/$VARIANT"
  mkdir -p "$OUT_DIR"

  for MODE in "${MODES[@]}"; do
    NPZ_OUT="$OUT_DIR/${MODE}.npz"
    ANALYSIS_OUT="$OUT_DIR/${MODE}.analysis.json"

    if [[ -f "$NPZ_OUT" ]]; then
      echo "    [Mode: $MODE] $NPZ_OUT already exists. Skipping inference."
    else
      echo "    [Mode: $MODE] Running text controls evaluation..."
      python -u scripts/run_text_controls.py \
        --forecast-task return_bp \
        --task-config-json "$CONFIG" \
        --model-path "$CKPT" \
        --text-model-path "$TEXT_MODEL_PATH" \
        --moment-path "$MOMENT_PATH" \
        --text-cache "$TEXT_CACHE" \
        --event-dir "$EVENT_DIR" \
        --series-dir "$SERIES_DIR" \
        --c1-manifest "$MANIFEST" \
        --c1-timing-policy category4_1400_strict_before \
        --no-4bit --tf32 \
        --zero-mode "$MODE" \
        --max-events 0 \
        --shuffles 3 \
        --out "$NPZ_OUT"
    fi

    if [[ -f "$NPZ_OUT" && ! -f "$ANALYSIS_OUT" ]]; then
      echo "    [Mode: $MODE] Computing bootstrap analysis and metrics..."
      python -u scripts/analyze_return_predictions.py \
        --predictions "$NPZ_OUT" \
        --out "$ANALYSIS_OUT"
    fi

    echo "    [Mode: $MODE] Completed: $NPZ_OUT"
  done
done

echo ""
echo "=========================================================================="
echo "All requested paired evaluations complete."
echo "Results stored in: $OUTPUT_ROOT"
echo "=========================================================================="
