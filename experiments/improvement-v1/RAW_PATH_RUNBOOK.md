# Compact GS-Fuse: 140-bar OHLC forecast

Run this validation-only round from the existing GS-Fuse campaign checkout, with the prior raw-news patch and this round's path files in the same checkout. Reuse the existing canonical C1 snapshot and `.venv-local-gpu`. No dataset re-preparation, installs, downloads, external checkpoints, or Llama/MOMENT are needed.

## 1. One-epoch CUDA smoke

In Windows PowerShell opened at the campaign repository root:

```powershell
$repo     = (Get-Location).ProviderPath
$runner   = Join-Path $repo 'scripts\improvement\run_local_path.ps1'
$python   = Join-Path $repo '.venv-local-gpu\Scripts\python.exe'
$snapshot = Join-Path $HOME 'gs-fuse-runs\raw-snapshot-v1'
$smoke    = Join-Path $HOME 'gs-fuse-runs\raw-path-smoke-v1'
$full     = Join-Path $HOME 'gs-fuse-runs\raw-path-validation-v1'

& $runner -Stage Smoke -Python $python -SnapshotRoot $snapshot -OutRoot $smoke
```

The snapshot must already contain its canonical `manifest.json` and `snapshot.json`. The runner checks their reference identities; the trainer verifies the underlying files. If the snapshot is missing or noncanonical, stop and locate the existing canonical copy. There is no exploratory override. If either output directory already exists, use a fresh versioned path instead of deleting or overwriting anything.

Smoke trains price-only followed by residual on CUDA with seed 0, batch size 32, and one epoch per mode. The residual gets the matching seed's price checkpoint; its price model stays frozen. A successful run emits `RAW_PATH_SMOKE_COMPLETE.json` only after checking the mode outputs and their provenance. Review both validation summaries before continuing:

```powershell
Get-Content -LiteralPath (Join-Path $smoke 'price\seed-0\resolved_config.json') -Raw |
    ConvertFrom-Json
Get-Content -LiteralPath (Join-Path $smoke 'residual\seed-0\resolved_config.json') -Raw |
    ConvertFrom-Json
```

Confirm `test_scored = false`, that the residual has `encoder_gradient_observed = true`, and that the matched/no-text/shuffled controls are present. Check the device, memory, and losses. A smoke pass verifies the pipeline, not forecasting performance. Stop on CUDA, input, output, or provenance errors instead of launching the full run.

## 2. Full validation matrix

```powershell
& $runner -Stage Full -Python $python -SnapshotRoot $snapshot -OutRoot $full -SmokeOut $smoke
```

Full trains both modes for seeds 0, 1, and 2, with CUDA batch size 32 and at most 30 epochs. The trainer selects the best checkpoint by validation whole-path MAE (epoch 0 is eligible; patience 5). It trains the price model first for each seed, then passes that run's `best.pt` to its residual. Full refuses a stale smoke receipt, changed snapshot, changed source files, changed Python executable, or altered smoke artifacts.

Each mode/seed directory has `resolved_config.json`, `best.pt`, and `vali.npz`. Residual runs also include `vali-text-controls.npz`. Collect the primary metric and provenance for all six runs:

```powershell
Get-ChildItem -LiteralPath $full -Filter 'resolved_config.json' -File -Recurse |
    ForEach-Object {
        $result = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        [pscustomobject]@{
            Mode = $result.mode
            Seed = $result.seed
            PathMAEbp = $result.vali.matched.path_mae_bp
            ValidationEvents = $result.vali.events
            BestEpoch = $result.best_epoch
            TestScored = $result.test_scored
            Summary = $_.FullName
        }
    } | Sort-Object Seed, Mode | Format-Table -AutoSize
```

Primary comparison: per-seed paired residual vs. same-seed price-only validation MAE over all 140 future OHLC bars, reported as basis-point log returns relative to the last observed close. The close endpoints at 35/70/140 are secondary diagnostics. For matched-text, no-text and shuffled-text comparisons, use the **same donor-eligible subset** and report its coverage and all three shuffle draws; don't compare subset numbers to full-validation numbers. The donor policy uses strictly earlier recorded dates in the same validation split, category and calendar year, excluding normalized exact duplicates; near duplicates and actual publish times are unverified. Compare only compatible task/metric records: sharing the whole-path target and scoring metric does not make this the R3 model or training objective.

## 3. Stop after validation

This runner has no test stage. Preserve all seeds, paired controls, hashes, resolved configurations, and runtime/memory records. Before authorizing any separate test scoring, document and lock the selected method and checkpoints, primary metric, seed aggregation, controls and coverage rules, pass/fail criteria, and canonical input/code identities. Do not tune these after seeing test results. If validation offers no good candidate or matched text does not beat its safeguards, report that rather than forcing a positive claim or treating validation results as test results.

## Results and locked test

See the [validation results](RAW_PATH_VALIDATION.md) and the [inference-only test runbook](RAW_PATH_TEST_RUNBOOK.md). The encoder reads archived summary text, not the original articles; it does not run Llama or load cached embeddings.

The finished baseline is documented in [locked test results](RAW_PATH_TEST_RESULTS.md).
