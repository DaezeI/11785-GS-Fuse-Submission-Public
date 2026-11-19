#!/usr/bin/env bash
# Portable Linux/Colab entry point. Pass an approved frozen snapshot and a new output folder.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  printf 'Usage: bash %s SNAPSHOT_ROOT NEW_OUTPUT_ROOT\n' "$0" >&2
  exit 2
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python_exe="${PYTHON:-python}"
device="${DEVICE:-cuda}"
out="$2"
if [[ -e "$out" ]]; then
  printf 'Output already exists; use a new directory: %s\n' "$out" >&2
  exit 2
fi
mkdir -p -- "$out"
echo 'Smoke: seed 0, one epoch per context length'
"$python_exe" -u "$repo/scripts/improvement/train_compact_context_ablation.py" \
  --snapshot-root "$1" --out "$out/smoke" --device "$device" --seeds 0 --epochs 1 | tee "$out/smoke.log"
echo 'Full queue: seeds 0, 1, 2; two context lengths; one fit at a time'
"$python_exe" -u "$repo/scripts/improvement/train_compact_context_ablation.py" \
  --snapshot-root "$1" --out "$out/validation" --device "$device" --seeds 0 1 2 --epochs 30 | tee "$out/validation.log"
printf 'Done: %s/validation/context-ablation-summary.json\n' "$out"
