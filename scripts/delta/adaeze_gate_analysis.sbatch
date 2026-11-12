#!/bin/bash
#SBATCH --job-name=adaeze-c1-gate
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=adaeze-gate-%j.out

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to your own Delta checkout containing these scripts}"
: "${OUT_ROOT:?Set OUT_ROOT to a new output directory you own}"
PHASE=${PHASE:-smoke}
case "$PHASE" in smoke|full) ;; *) echo 'PHASE must be smoke or full' >&2; exit 2 ;; esac
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-$OUT_ROOT/hf-cache}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CONFIG=${CONFIG:-configs/adaeze_gate_c1.json}
RUN_DIR="$OUT_ROOT/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
python -c 'import sys, torch, numpy; print(sys.version); print("torch",torch.__version__,"numpy",numpy.__version__,"CUDA",torch.cuda.is_available())' > "$RUN_DIR/environment.txt"
python -c 'import importlib.metadata as m; print("\n".join(sorted(str(d.metadata["Name"])+"=="+d.version for d in m.distributions())))' > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
git status --short > "$RUN_DIR/code_status.txt"
if [[ "$PHASE" == smoke ]]; then
  python -u scripts/run_gate_analysis.py --config "$CONFIG" --split vali \
    --max-events 16 --probe-gradients --out "$RUN_DIR/vali-smoke"
  python scripts/analyze_gate_analysis.py --predictions "$RUN_DIR/vali-smoke/predictions.npz" \
    --out "$RUN_DIR/vali-smoke/analysis"
else
  for SPLIT in vali test; do
    python -u scripts/run_gate_analysis.py --config "$CONFIG" --split "$SPLIT" \
      --max-events 0 --out "$RUN_DIR/$SPLIT"
    python scripts/analyze_gate_analysis.py --predictions "$RUN_DIR/$SPLIT/predictions.npz" \
      --out "$RUN_DIR/$SPLIT/analysis"
  done
fi
