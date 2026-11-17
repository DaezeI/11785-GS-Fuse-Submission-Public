#!/bin/bash
#SBATCH --job-name=nicole-c1-return-audit
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=12G
#SBATCH --time=00:25:00

set -euo pipefail
umask 007
# Delta requires a GPU allocation for this account. The audit uses only the CPU.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate

ROOT=/work/hdd/bjjg/gs-fuse
: "${AUDIT_OUT:?Set a fresh output directory under the shared run root}"
: "${EXPECTED_CODE_COMMIT:?Set the audited checkout commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_CODE_COMMIT"
test -z "$(git status --porcelain)"
mkdir "$AUDIT_OUT"
MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE="$ROOT/data/series/SP500.csv"
SNAPSHOT="$ROOT/data/snapshots/c1-screened-return-v1"
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
if [[ ! -e "$SNAPSHOT" ]]; then
  python -u scripts/freeze_c1_inputs.py \
    --manifest "$MANIFEST" --event-dir "$ROOT/data/event" --series "$PRICE" --out "$SNAPSHOT"
fi
cmp "$MANIFEST" "$SNAPSHOT/manifest.json"
cmp "$PRICE" "$SNAPSHOT/series/SP500.csv"
python -u scripts/audit_c1_protocol.py --snapshot-root "$SNAPSHOT" \
  --forecast-task return_bp --timing-policy category4_1400_strict_before \
  --out "$AUDIT_OUT/summary.json" --windows-csv "$AUDIT_OUT/windows.csv"
