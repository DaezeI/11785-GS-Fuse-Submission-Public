# Run the compact multimodal experiments on Adaeze's GPU

The target remains **prices plus news**. Price-only is the baseline. The
default local runner trains the price branch, freezes it, then trains the
small news correction and scores matched/shuffled/frozen-price controls.
No Llama or MOMENT weights are loaded during this local training loop.

## What is verified on this laptop

- GPU: NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB VRAM.
- The shared `C:\Users\Adaeze Ilodianya\Documents\Fall 2026\.venv_gpu`
  contains **PyTorch 2.5.1+cu124**, not CPU-only PyTorch. Its python.exe launcher
  cannot find the original Python 3.12.10 base installation at its recorded path.
- The original environment was left unchanged. A repository-local
  `.venv-local-gpu` uses the available Python 3.12.14 runtime and a `.pth`
  reference to the existing CUDA packages. CUDA tensor operations and a
  synthetic multimodal training/inference benchmark passed through this
  environment. Keep the referenced shared packages in place.
- The compact price/text model has 82,598 parameters with 2,048-dimensional
  text input. The saved synthetic state was 336,480 bytes. PyTorch peak
  reserved GPU memory was 46,137,344 bytes in that synthetic check, excluding
  raw-text encoding and other process/context allocations. This is not a
  forecasting-quality result. See [the measured check](compact-4060-multimodal-synthetic.json).

The remaining data prerequisite is the portable export below. The real
dataset/cache are not in this checkout, so no real-data local run is claimed.

## 1. Nicole exports two files once

On the host that already has the frozen snapshot and verified pooled cache,
fetch this branch and run the following in its project environment. The
cache checksum must come from the actual operator-verified file/record.
Use a fresh export path with sufficient quota. This is a CPU/data export,
not a new LLM run or model download.

```bash
python scripts/improvement/compact_features.py \
  --snapshot-root "$SNAPSHOT" \
  --cache "$FROZEN_POOLED_CACHE" \
  --cache-sha256 "$VERIFIED_CACHE_SHA256" \
  --include-prices \
  --out "$APPROVED_EXPORT_DIR/compact-c1-l35.npz"
```

Keep **both** outputs together:

1. `compact-c1-l35.npz`
2. `compact-c1-l35.npz.receipt.json`

The NPZ contains numeric inputs/close-return endpoint targets, frozen pooled
news vectors, event IDs/roles/content hashes, bar intervals and provenance.
It omits raw articles, token states and backbone weights. The source snapshot
and cache are checksum-verified before export; missing features fail rather
than being silently generated. The laptop rechecks the transferred checksum,
population, split bar separation, input lengths and training-only scale.
The export retains the publication-time limitation; it is not a new dataset
or a timing repair. Transfer through the agreed private team storage; do not
commit the pack to GitHub. No transfer is performed automatically by this code.

## 2. Put the files in this checkout

In PowerShell:

```powershell
Set-Location -LiteralPath 'C:\Users\Adaeze Ilodianya\Documents\Fall 2026\11785 Intro to Deep Learning\GS-FUSE-main\11785-GS-Fuse-Submission-Public'
git status --short
git fetch origin
git switch adaeze/controlled-improvement-campaign
git pull --ff-only
New-Item -ItemType Directory -Force -Path '.\campaign-local\inputs' | Out-Null
```

Download/copy Nicole's two files into `campaign-local\inputs`. This directory
is ignored by Git. Preserve any unrelated local edits before switching branches.
The already-created local GPU environment does not need reinstallation.

## 3. Confirm the working GPU environment

```powershell
& '.\.venv-local-gpu\Scripts\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.ones(1,device='cuda'); print(x.item())"
```

Expect `2.5.1+cu124`, `True`, the RTX 4060 Laptop name, and `1.0`.
This calls the interpreter explicitly; activation is optional. Do not switch
back to the broken shared launcher or the CPU-only `.venv-campaign` for GPU runs.
The local environment bridge depends on the referenced shared CUDA packages
and working base runtime. If either is moved, recreate a stable Python 3.12
environment and install CUDA-enabled PyTorch before rerunning; the runner
refuses a silent CPU fallback.

## 4. One-epoch execution check on your GPU

```powershell
& '.\scripts\improvement\run_local.ps1' `
  -InputPack '.\campaign-local\inputs\compact-c1-l35.npz' `
  -Receipt '.\campaign-local\inputs\compact-c1-l35.npz.receipt.json' `
  -Mode multimodal -Epochs 1 -Seed 0
```

This checks both compact stages and validation controls on the portable
population. It is distinct from Nicole's 16/8/8 foundation-model smoke.
The runner verifies the export receipt and an actual CUDA tensor before
training. It creates a fresh output directory under `campaign-local\runs`.

## 5. Run the prespecified local comparisons

After the execution check succeeds:

```powershell
# Main price + news experiment, with its own fitted/frozen price reference:
& '.\scripts\improvement\run_local.ps1' `
  -InputPack '.\campaign-local\inputs\compact-c1-l35.npz' `
  -Receipt '.\campaign-local\inputs\compact-c1-l35.npz.receipt.json' `
  -Mode multimodal -Epochs 30 -Seed 0

# Optional independently executed price-only reference:
& '.\scripts\improvement\run_local.ps1' `
  -InputPack '.\campaign-local\inputs\compact-c1-l35.npz' `
  -Receipt '.\campaign-local\inputs\compact-c1-l35.npz.receipt.json' `
  -Mode price -Epochs 30 -Seed 0
```

Use seeds 1/2 only after choosing the comparison from validation and fixing
its settings. Default commands do not score test. A later test run requires
a locked selection JSON with source commit, width/seq_len, input-pack and
text-feature checksums, validation evidence and selection rule.

## 6. Read the outputs

`resolved_config.json` records stage budgets, feature provenance, validation
errors at H35/70/140, zero return and the frozen price-only reference. The
`vali.npz` predictions share all validation targets; `vali-text-controls.npz`
uses a separately reported donor-eligible subset with three earlier-date
article assignments. Report those counts rather than comparing mixed
populations. Check `best.pt` for the selected model and fitted feature scaler.

Treat the one-epoch run as an execution check. A useful multimodal result
must improve appropriate matched price/zero-return references and show
article-matching value, with uncertainty; the design does not guarantee it.
Keep large per-event files private and publish only reviewed aggregates,
figures, provenance and registry updates through the result-report workflow.
