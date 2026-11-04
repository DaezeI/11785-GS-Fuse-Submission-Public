# Shared NCSA Delta environment for the whole bjjg group.
# Usage (login node or compute node):
#   source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
#
# Do NOT `conda activate gs-fuse` on Delta. That env does not exist here.
# The shared interpreter is a venv on top of `module load pytorch-conda/2.8`.

umask 002

export GSFUSE_ROOT="${GSFUSE_ROOT:-/projects/bjjg/gs-fuse}"
export GSFUSE_REPO="${GSFUSE_REPO:-$GSFUSE_ROOT/repo/GS-FUSE}"
export GSFUSE_ENV="${GSFUSE_ENV:-$GSFUSE_ROOT/env/gsfuse}"
export GSFUSE_DATA="${GSFUSE_DATA:-/work/hdd/bjjg/gs-fuse/data}"
export GSFUSE_CACHE="${GSFUSE_CACHE:-/work/hdd/bjjg/gs-fuse/cache}"
export GSFUSE_RUNS="${GSFUSE_RUNS:-/work/hdd/bjjg/gs-fuse/runs}"
export GSFUSE_PRETRAINED="${GSFUSE_PRETRAINED:-/work/hdd/bjjg/gs-fuse/pretrained}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$GSFUSE_CACHE/hf}"
export TRANSFORMERS_CACHE="$HF_HOME"
export WANDB_DIR="${WANDB_DIR:-$GSFUSE_RUNS/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-11785-GS-Fuse}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$GSFUSE_DATA" "$GSFUSE_CACHE" "$GSFUSE_RUNS" "$GSFUSE_PRETRAINED" \
         "$HF_HOME" "$WANDB_DIR"

module load pytorch-conda/2.8 >/dev/null
# shellcheck disable=SC1091
source "$GSFUSE_ENV/bin/activate"

cd "$GSFUSE_REPO" || return 1
export PYTHONPATH="$GSFUSE_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "GS-FUSE Delta env"
echo "  repo:       $GSFUSE_REPO"
echo "  python:     $(which python)  ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "  torch:      $(python -c 'import torch; print(torch.__version__, \"cuda\" if torch.cuda.is_available() else \"cpu\")')"
echo "  data:       $GSFUSE_DATA"
echo "  cache:      $GSFUSE_CACHE"
echo "  runs:       $GSFUSE_RUNS"
echo "  pretrained: $GSFUSE_PRETRAINED"
