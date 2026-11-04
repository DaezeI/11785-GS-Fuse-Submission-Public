# NCSA Delta: shared env and midterm checkpoints

The 晨涧云 `best_model.pth` files were never copied off that GPU. Recreate them here on the `feat/moe-qformer-wandb-cache` branch so the whole `delta_bjjg` group can keep experimenting from the same weights.

## Layout (group-writable)

| Path | What |
|---|---|
| `/projects/bjjg/gs-fuse/repo/GS-FUSE` | git clone (this branch) |
| `/projects/bjjg/gs-fuse/env/gsfuse` | shared venv on top of `pytorch-conda/2.8` |
| `/work/hdd/bjjg/gs-fuse/data/{event,series}` | CAMEF events + SP500 bars |
| `/work/hdd/bjjg/gs-fuse/pretrained` | `Llama-3.2-1B`, `MOMENT-1-large` |
| `/work/hdd/bjjg/gs-fuse/cache` | `text_inst_llama1b_tf32.pt` (~222MB) |
| `/work/hdd/bjjg/gs-fuse/runs/full/{baseline,moe,qformer}/best_model.pth` | the matching checkpoints |

Do **not** `conda activate gs-fuse` on Delta. That name does not exist here. Always:

```bash
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
```

## 1. One-time env (any group member, login node)

```bash
cd /projects/bjjg/gs-fuse/repo/GS-FUSE
git switch feat/moe-qformer-wandb-cache
git pull
bash scripts/delta/setup_shared.sh
```

This installs `wandb`, `momentfm` (`--no-deps`), `einops`, etc. into the shared venv and **does not** replace Delta's PyTorch 2.8. It also symlinks `data/`, `pretrained/`, `cache/`, `outputs/` to `/work/hdd/...` so `train.py` keeps its relative paths.

Teammates after that only need `source .../scripts/delta/env.sh`.

## 2. Copy data and frozen backbones from the Mac

From your laptop (will prompt NCSA password + Duo). About 6 GB:

```bash
# series + events (~1 GB)
rsync -avP --partial \
  ~/Desktop/project/GS-FUSE/data/series \
  ~/Desktop/project/GS-FUSE/data/event \
  zliu69@login.delta.ncsa.illinois.edu:/work/hdd/bjjg/gs-fuse/data/

# Llama-3.2-1B + MOMENT-1-large (~5 GB). Prefer this over HuggingFace; Llama is gated.
rsync -avP --partial \
  ~/Desktop/project/GS-FUSE/pretrained/Llama-3.2-1B \
  ~/Desktop/project/GS-FUSE/pretrained/MOMENT-1-large \
  zliu69@login.delta.ncsa.illinois.edu:/work/hdd/bjjg/gs-fuse/pretrained/
```

On Delta, confirm:

```bash
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
ls data/series/SP500.csv data/event/1 | head
ls pretrained/Llama-3.2-1B/config.json pretrained/MOMENT-1-large/config.json
```

## 3. Recreate the three checkpoints (batch GPU, not interactive)

Interactive partition allows only **one** queued/running job per user. Training belongs on `gpuA100x4`.

```bash
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
bash scripts/delta/submit_full_runs.sh
squeue -u $USER
```

Chain: text cache (~2 h) → baseline (~3 h) → MoE (~3 h) → Q-Former (~3 h) → eval. Each GPU job is one A100-40GB. Checkpoints land at:

```
/work/hdd/bjjg/gs-fuse/runs/full/baseline/best_model.pth
/work/hdd/bjjg/gs-fuse/runs/full/moe/best_model.pth
/work/hdd/bjjg/gs-fuse/runs/full/qformer/best_model.pth
```

(`outputs/full/...` in the repo is a symlink to that directory.)

Single variant:

```bash
sbatch --export=ALL,VARIANT=baseline scripts/delta/train.sbatch
```

Smoke test on the interactive GPU (15 min, 64 events):

```bash
srun --account=bjjg-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --mem=32G --time=00:20:00 --pty bash
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
python -u smoke_train.py --moment-path pretrained/MOMENT-1-large \
  --text-model-path pretrained/Llama-3.2-1B --batch-size 8 --tf32 \
  --fusion-type crossattn --gate-type softmax --skip-checkpoints \
  --n-event-train 64 --n-event-eval 32 --output-path outputs/smoke/baseline
```

## 4. W&B

Default jobs are `WANDB_MODE=offline` so a missing key cannot kill training. After `wandb login` on Delta (once per user):

```bash
export WANDB_MODE=online
# then resubmit, or:
wandb sync /work/hdd/bjjg/gs-fuse/runs/full/*/wandb/offline-run-*
```

## 5. Continue experiments from a checkpoint

```bash
source /projects/bjjg/gs-fuse/repo/GS-FUSE/scripts/delta/env.sh
python -u test_pretrained_model.py \
  --model-path outputs/full/baseline/best_model.pth \
  --fusion-type crossattn --gate-type softmax \
  --text-model-path pretrained/Llama-3.2-1B \
  --moment-path pretrained/MOMENT-1-large \
  --text-cache cache/text_inst_llama1b_tf32.pt --tf32 --no-4bit \
  --split test --feature-idx 3
```

Same `--tf32` / cache signature as training. `--no-4bit` matches the midterm weights (fp32 Llama).

## Notes

- Login node reboot: Thu Nov 5, 07:00 CST. Running batch jobs keep going; `sbatch`/`squeue` pause during the reboot.
- Allocation: `bjjg-delta-gpu`, 1000 GPU-hours. Three full variants + cache ≈ 12 GPU-hours.
- File quota: put weights/data/runs on `/work/hdd/bjjg` (1 TB), not `$HOME` (100 GB).
