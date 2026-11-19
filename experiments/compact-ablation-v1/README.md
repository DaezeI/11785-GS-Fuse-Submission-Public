# Compact GS-Fuse ablations

The compact model is the model we are developing. R3 is historical context, not
an acceptance gate. The verified 35-bar full-path test is a closed baseline;
do not use that test to select future variants.

## First batch: input history

Compare 35 vs. 140 observed price bars with width, loss, seeds, forecast origins
and 140-bar OHLC target held fixed. Both arms are price-only. Using the same
eligible cohort for both arms avoids mistaking differences in event coverage for
a context-length effect. Check the cohort audit before interpreting the scores;
they are not a replacement for the original full-C1 test score. There is no test
scoring in this batch.

On the RTX 4060, from the repository folder:

```powershell
& '.\scripts\improvement\run_local_context_ablation.ps1'
```

Defaults use the existing CUDA environment and snapshot. The runner performs a
one-epoch seed-0 smoke, then automatically runs both contexts for seeds 0, 1 and
2, one fit at a time. It stops on the first failed command. Keep the laptop awake.
Output: `$HOME\gs-fuse-runs\compact-context-v1`. Keep all raw inputs, checkpoints
and per-event arrays out of Git. Share run artifacts in the team's Drive.

Record **planned** before launch, **running** only once the full training log
starts, and use the validation summary plus cohort audit before marking results
complete. Do not infer a gain from the smoke.

## Subsequent ablations (not queued)

- Compare shorter vs. longer text budgets for the compact summary encoder.
- Compare a fixed news correction with a gate informed only by pre-origin prices.
- Compare training on correctly matched vs. earlier-date shuffled summaries,
  with matched evaluation and paired same-seed price controls. Track coverage and
  near duplicates. Text availability at the forecast origin is not yet verified.

Do not stack these changes in the first batch. No Llama, Q-Former, or R3 rerun is
needed for the compact campaign.

## Colab and Linux

Use the [portable setup](../improvement-v1/COMPACT_PORTABLE_RUNBOOK.md). On a GPU
runtime, from the same repository checkout:

```bash
bash scripts/improvement/run_local_context_ablation.sh /content/gs-fuse-snapshot /content/gs-fuse-context-v1
```

On a Linux GPU workstation use its own paths. A CPU-only smoke can call
`train_compact_context_ablation.py` with `--device cpu --seeds 0 --epochs 1`,
the snapshot path, and a fresh output directory. The Linux wrapper defaults to
`cuda`; only set `DEVICE=cpu` if you intend to run the whole queue on CPU.
The Bash path has syntax checks, not an end-to-end Colab benchmark.

## Completed reference run

The first queue completed on RTX 4060 without test scoring. See the
[paired validation report](../../results/compact-context-v1/summary.md).
Keep L35 as the reference until the L140 gain is tested for stability across
trading dates.
