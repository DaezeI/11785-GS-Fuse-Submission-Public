#!/bin/bash
#SBATCH --job-name=nicole-c1-date-audit
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/logs/date-audit-%j.out

set -euo pipefail
umask 007
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/repo

MANIFEST=splits/manifests/purged_split_c1_date_screen_v1.json
PRICE=/work/hdd/bjjg/gs-fuse/data/series/SP500.csv
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a

python -u scripts/audit_c1_protocol.py \
  --manifest "$MANIFEST" \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series "$PRICE" \
  --timing-policy category4_1400_strict_before \
  --out /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/audit/summary.json \
  --windows-csv /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/audit/windows.csv
