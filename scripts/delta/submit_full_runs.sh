#!/usr/bin/env bash
# Submit cache -> baseline -> moe -> qformer -> eval as a dependent chain.
# Run from the repo root on a Delta login node:
#   bash scripts/delta/submit_full_runs.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
mkdir -p /work/hdd/bjjg/gs-fuse/runs/logs

cache_j=$(sbatch --parsable "$HERE/precompute.sbatch")
echo "cache     $cache_j"
b_j=$(sbatch --parsable --dependency=afterok:"$cache_j" --export=ALL,VARIANT=baseline --job-name=gsfuse-baseline "$HERE/train.sbatch")
echo "baseline  $b_j  afterok:$cache_j"
m_j=$(sbatch --parsable --dependency=afterok:"$b_j" --export=ALL,VARIANT=moe --job-name=gsfuse-moe "$HERE/train.sbatch")
echo "moe       $m_j  afterok:$b_j"
q_j=$(sbatch --parsable --dependency=afterok:"$m_j" --export=ALL,VARIANT=qformer --job-name=gsfuse-qformer "$HERE/train.sbatch")
echo "qformer   $q_j  afterok:$m_j"
e_j=$(sbatch --parsable --dependency=afterok:"$q_j" --job-name=gsfuse-eval "$HERE/eval.sbatch")
echo "eval      $e_j  afterok:$q_j"
echo
echo "Watch:  squeue -u \$USER"
echo "Logs:   ls -lt /work/hdd/bjjg/gs-fuse/runs/logs | head"
echo "Ckpts:  ls -lh /work/hdd/bjjg/gs-fuse/runs/full/*/best_model.pth"
