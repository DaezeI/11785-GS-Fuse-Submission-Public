#!/usr/bin/env bash
# Evaluate finished variants on vali and test with financial metrics; saves predictions for baseline comparisons.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/conda_env.sh; set +u; conda activate gs-fuse; set -u
export PYTHONNOUSERSITE=1
for v in ${VARIANTS:-baseline moe}; do
  case $v in
    baseline) flags=(--fusion-type crossattn --gate-type softmax) ;;
    moe)      flags=(--fusion-type crossattn --gate-type moe) ;;
    qformer)  flags=(--fusion-type qformer --gate-type softmax) ;;
  esac
  for split in test vali; do
    python -u test_pretrained_model.py --model-path outputs/full/$v/best_model.pth "${flags[@]}" \
      --text-model-path pretrained/Llama-3.2-1B --moment-path pretrained/MOMENT-1-large \
      --series-id SP500 --batch-size 8 --split $split --feature-idx 3 \
      --text-cache cache/text_inst_llama1b_tf32.pt --tf32 --no-4bit \
      --output-json outputs/eval/${v}_${split}.json --save-preds outputs/eval/${v}_${split}_preds.npz \
      > outputs/eval/${v}_${split}.log 2>&1
    echo "$v $split exit $?"
  done
done
