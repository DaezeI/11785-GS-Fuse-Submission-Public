# Compact Treasury pilot: token attention × text gate

This is a validation-only run on the separate Fed-statement/daily-Treasury pilot. It forecasts the next five observed 10Y Treasury yield changes in basis points relative to the last observation. No C1/equity result, no pretrained text model, no news cache, and no test scoring. The market series is published daily par yields, not intraday trades; statement source evidence is a corroborated publisher archive, not immutable minute-level availability.

## Provenance and source distribution

The actual Windows run used **GS-Fuse-Compact-Treasury-CrossAttention.py**, SHA-256 `86ffa70cbdda87c2a02297cb2637bddf16f32aa67944d0e7127f2e482f25b0dc`. Here the source-only counterpart is [run_compact_treasury_crossattention.py](../../scripts/improvement/run_compact_treasury_crossattention.py). Only packaging and loading changed: embedded splits/decompression were removed and `--data-dir` loads the same byte-pinned `train.jsonl` and `validation.jsonl` filenames. The loader does not inspect a manifest or open a held-out file. The architecture, initialization, losses, OOF construction, training schedule and evaluation/scoring AST are unchanged. [Machine-readable design and hashes](compact_treasury_crossattention.config.json) record the distinction. The actual run's sanitized [metrics and evidence](../../results/compact-treasury-crossattention-v1/summary.md) are separate from the executable. No raw data are committed.

## Design and checks

| Arm | Token aggregation | Gate |
|---|---|---|
| Pooled | Uniform masked mean of projected token embeddings | Off |
| Cross-attention | Frozen yield summary queries projected token embeddings | Off |
| Pooled + gate | Same pooled model and frozen residual | On |
| Cross-attention + gate | Same cross-attention model and frozen residual | On |

The four cells share a fitted frozen numeric reference. The two token encoders/residual heads have matched initialization, learned parameter count, and batches. Attention uses one nonlearned scaled dot-product head over the 64-dimensional causal yield summary and projected tokens. Both ungated variants use full-path Smooth L1 plus weight-one endpoint sign loss (days 1, 3, 5; fixed 10 bp scale; true zeros excluded). Gated variants train separate identically configured two-way softmax gates on train-only chronological OOF predictions, using a shared baseline in each teacher fold and architecture-specific residual teachers. Soft targets use detached complete task-loss gain, minibatch scale floor 1e-6 and clipping to [-6,6], with 0.1-weighted gate MSE. Dates are strictly separated between each fold's fitting and scoring windows. Train/validation have 36/12 distinct statements (180/60 rows). Seeds 23/24/25, ten fixed epochs, no validation checkpoint selection.

Report event-weighted five-day MAE in bp, MSE in bp², endpoint DHR and first-origin sensitivity. Previous-statement text is a timing-respecting control for all four arms; no-change, frozen yields, fixed-half gates and zero/static-query inference probes aid interpretation. A zero forecast misses nonzero actual direction; actual zero movement is excluded from DHR.

## Reproduce on Windows

Download the separately distributed **GS-Fuse-Fed-Treasury-Training-v1.zip** to Downloads. It contains the exact train and validation data; the old ridge scripts included in that archive are irrelevant here. From the repository root, in PowerShell using the prior CUDA-enabled PyTorch environment:

```powershell
$DataZip = "$env:USERPROFILE\Downloads\GS-Fuse-Fed-Treasury-Training-v1.zip"
$DatasetDir = "$env:USERPROFILE\Downloads\GS-Fuse-Fed-Treasury-CrossAttention-Data"
Expand-Archive -LiteralPath $DataZip -DestinationPath $DatasetDir
python .\scripts\improvement\run_compact_treasury_crossattention.py --device cuda --data-dir "$DatasetDir\data"
```

Optional: add `--verify` to run just the split checks, or use `--smoke` without `--data-dir` for fabricated data only. Results default to `~/gs-fuse-runs/fed-treasury-compact-crossattention-v1/<timestamp>`; put the original run’s `metrics.json` with its result packet rather than replacing it with a rerun. To check the protected AST against a retained historical runner without importing Torch, set `GS_FUSE_EXECUTED_CROSSATTN` to its path and run:

```powershell
python -m unittest discover -s tests -p 'test_compact_treasury_crossattention_source.py'
```
