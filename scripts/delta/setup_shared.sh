#!/usr/bin/env bash
# One-time (or after git pull) setup on Delta. Run on a login node:
#   bash /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/setup_shared.sh
#
# Installs the extra Python packages into the shared venv, makes the tree
# group-writable, and points the repo at the shared data/cache/run directories.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

echo "== group permissions =="
for d in "$GSFUSE_ROOT" /work/hdd/bjjg/gs-fuse /work/nvme/bjjg/gs-fuse; do
  [ -d "$d" ] || continue
  chmod -R g+rwX "$d" || true
  find "$d" -type d -exec chmod g+s {} + 2>/dev/null || true
done

echo "== Python extras (do not reinstall torch; Delta provides 2.8+cu128) =="
python -m pip install --upgrade pip
python -m pip install \
  wandb \
  einops \
  sentencepiece \
  scikit-learn \
  pytest \
  pyyaml \
  tqdm \
  requests \
  psutil
# momentfm pins an old transformers; keep Delta's module stack and only take momentfm itself.
python -m pip install --no-deps momentfm==0.1.4

echo "== sanity =="
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda built:", torch.version.cuda)
for m in ("transformers", "accelerate", "einops", "sklearn", "wandb", "momentfm", "sentencepiece"):
    importlib.import_module(m if m != "sklearn" else "sklearn")
    print(" ok", m)
PY
python -m pytest -q tests

echo "== repo symlinks into shared storage =="
link() {
  local dest="$1" src="$2"
  mkdir -p "$src"
  if [ -L "$dest" ]; then
    ln -sfn "$src" "$dest"
  elif [ -d "$dest" ] && [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
    rmdir "$dest"
    ln -s "$src" "$dest"
  elif [ ! -e "$dest" ]; then
    ln -s "$src" "$dest"
  else
    echo "keep existing $dest"
  fi
}
link "$GSFUSE_REPO/data/event"      "$GSFUSE_DATA/event"
link "$GSFUSE_REPO/data/series"     "$GSFUSE_DATA/series"
link "$GSFUSE_REPO/pretrained"      "$GSFUSE_PRETRAINED"
link "$GSFUSE_REPO/cache"           "$GSFUSE_CACHE"
link "$GSFUSE_REPO/outputs"         "$GSFUSE_RUNS"

echo
echo "Next:"
echo "  1. Copy data + Llama/MOMENT weights (see docs/DELTA.md)."
echo "  2. sbatch scripts/delta/precompute.sbatch"
echo "  3. sbatch scripts/delta/submit_full_runs.sh"
echo "Shared env is $GSFUSE_ENV — teammates only need: source $HERE/env.sh"
