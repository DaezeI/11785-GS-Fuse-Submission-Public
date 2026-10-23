#!/usr/bin/env bash
# Unattended pipeline: build the text-embedding cache, then train every variant against it.
#   tmux new -d -s train "bash scripts/overnight.sh 2>&1 | tee outputs/overnight.log"
# Env overrides: TF32 (default 1), TEXT_CACHE, WANDB_MODE (default offline), plus anything run_variants.sh reads.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/conda_env.sh
set +u; conda activate gs-fuse; set -u
export PYTHONNOUSERSITE=1

export TF32=${TF32:-1}
suffix=$([ "$TF32" = 1 ] && echo tf32 || echo fp32)
export TEXT_CACHE=${TEXT_CACHE:-cache/text_inst_llama1b_${suffix}.pt}
export WANDB_MODE=${WANDB_MODE:-offline}

echo "==== [$(date '+%F %T')] text cache -> $TEXT_CACHE ===="
cache_flags=()
if [ "$TF32" = 1 ]; then cache_flags+=(--tf32); fi
if [ "${QUANT_4BIT:-0}" = 1 ]; then cache_flags+=(--use-4bit); fi
python -u scripts/precompute_text_cache.py --text-model-path pretrained/Llama-3.2-1B \
  --out "$TEXT_CACHE" "${cache_flags[@]}"

bash scripts/run_variants.sh
echo "==== [$(date '+%F %T')] all done ===="
