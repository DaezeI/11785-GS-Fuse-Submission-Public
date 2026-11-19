# Compact GS-Fuse on Windows, Colab, and Linux

The fixed 35-bar baseline and locked test ran on an RTX 4060 Laptop with
PyTorch 2.5.1+cu124. The Python code has a CPU path and does not require an RTX
4060, Slurm, Llama, MOMENT, or any model download. A CUDA runtime on another
NVIDIA GPU is a reasonable target, not a verified reproduction. Colab availability,
GPU type, session limits, and runtime versions vary. See the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Inputs and environment

- Check out the agreed commit from GitHub. Do not upload the dataset, checkpoints,
  or event-level arrays to Git. Use the approved team Drive for those files.
- For Colab, select a GPU runtime, mount Drive, and copy/extract the team's
  approved frozen snapshot to local runtime disk. The snapshot root must contain
  `manifest.json`, `snapshot.json`, `event/` and `series/SP500.csv`. Preserve
  bytes and layout; do not substitute a bare dataset ZIP for the frozen snapshot.
  If the team shares only the original dataset, prepare the canonical snapshot
  first using the [staging runbook](RAW_NEWS_RUNBOOK.md).
- Use an environment with PyTorch, NumPy, pandas, and scikit-learn. Check the
  existing environment first; do not install the large upstream backbone
  dependencies for the compact jobs. Use a CUDA-enabled PyTorch build appropriate
  for your GPU; a CPU smoke is useful when CUDA isn't available.

```bash
python -c 'import torch,numpy,pandas,sklearn; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

## Portable 35-bar baseline smoke

Run from the repository root. Replace the two paths and use a fresh output
folder. In a Colab notebook, prepend `!` to a single-shell-line command or use
a `%%bash` cell. For CPU, use `--device cpu`; the example is a one-seed,
one-epoch execution check, not a reported result.

```bash
python -u scripts/improvement/train_compact_path.py --snapshot-root /content/gs-fuse-snapshot --out /content/gs-fuse-portability-smoke --device cuda --mode price --seed 0 --epochs 1 --batch-size 32
```

For a new context experiment on the compact-ablation branch, see its runbook
and Bash wrapper. A scheduler on a managed cluster may wrap the same Python
or Bash command after the site allocates one GPU. On an ordinary workstation,
run it directly. Save the output folders or ZIPs to team storage before ending a
Colab runtime. Preserve the GPU/PyTorch details and smoke log, and report any
differences; exact numerical reproducibility across GPU/PyTorch versions is not
assumed.
