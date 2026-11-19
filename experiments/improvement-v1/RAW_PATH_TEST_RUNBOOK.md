# GS-Fuse: score the saved whole-path models on test

Use the same GS-Fuse repo folder where `run_local_path.ps1` was run. Keep the
existing trainer code, canonical snapshot, and six validation run folders unchanged.

In PowerShell, from that same repo folder:

```powershell
& '.\scripts\improvement\run_local_path_test.ps1'
```

The defaults use the existing `.venv-local-gpu` in this repo and these folders:

- Input snapshot: `$HOME\gs-fuse-runs\raw-snapshot-v1`
- Saved validation runs: `$HOME\gs-fuse-runs\raw-path-validation-v1`
- New output: `$HOME\gs-fuse-runs\raw-path-test-v1`

Override those paths if you moved the files using `-SnapshotRoot`,
`-ValidationRoot`, `-OutRoot`, or `-Python`. Existing output is not overwritten.
If an interrupted run leaves the output folder behind, keep it for diagnosis;
use a fresh output path only to rerun the same locked configuration, not to tune.

## Fixed before test

Score all six saved validation-selected checkpoints: price-only and frozen-price
plus learned news for each of seeds 0, 1, and 2. Do not select a winner by seed,
tune on test, or retrain. On the screened 2,236 examples, report mean absolute
basis-point log-return error over 140 bars and all four OHLC channels, as well as
seed-by-seed scores and the mean across seeds. Compare the two lightweight modes
using the same test examples. Keep the zero-return baseline in view.

The published screened R3 H200 single-seed result is 80.615 bp and its
zero-return baseline is 80.487 bp. These are descriptive references, not a
statistical head-to-head. Published provenance confirms the same snapshot,
manifest, timing policy, return anchor and units, and event count. The private
R3 per-event predictions were not available for exact ID/order verification.
See [R3 metrics](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/main/results/adaeze-screened-r3-h200-v2/metrics.json)
and [R3 provenance](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/main/results/adaeze-screened-r3-h200-v2/provenance.json). Keep the existing timing screen and document that availability
of the text at forecast time is unverified. The compact encoder reads archived
summary text; it does not run Llama or load its embeddings.

Upload only `test-summary.json` if you want the results checked in chat. Keep
`checkpoint-lock.json` and `test-predictions.npz` locally for reproducibility.

The finished baseline is documented in [locked test results](RAW_PATH_TEST_RESULTS.md).
