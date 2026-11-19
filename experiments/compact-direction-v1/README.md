# Compact price-only direction-loss ablation

**Question:** Does adding a direction penalty improve the sign of forecasts
without worsening whole-path error? The treatment changes only training loss.
No text, gate, context-length change, test scoring, or new backbone.

| Arm | Training objective |
| --- | --- |
| Magnitude reference | Smooth L1 over all 140 OHLC log-return steps, scaled by the train-only C1 scale |
| Direction treatment | Same magnitude loss plus a weight-1 sign loss at H35/H70/H140 close; exclude exactly zero target returns |

Canonical screened C1, 35 observed bars, 140 forecast bars, same events and
forecast origins, width 32, seeds 0/1/2, batch 32, AdamW, same epoch/patience
rules. Select both arms by whole-path validation MAE. Report whole-path MAE
and endpoint direction hit rates by horizon, per seed and as seed averages.
Do not use the locked test data to select hyperparameters, seeds, or variants.

On Windows (from the root of the repository):

```powershell
& '.\scripts\improvement\run_local_direction_ablation.ps1'
```

The runner smokes both variants for one epoch (seed 0), then runs the three
matched seed pairs sequentially. It stops on errors and never overwrites an
existing output directory. Results are under
`$HOME\gs-fuse-runs\compact-direction-v1\validation`. Keep predictions,
checkpoints, and raw logs out of Git. Record the real run as running only after
the full queue starts; do not report a performance result from the smoke.

On Colab/Linux, after staging the same snapshot and checking PyTorch/CUDA,
invoke the same Python trainer directly, first with `--seeds 0 --epochs 1` and
a fresh output folder, then with `--seeds 0 1 2 --epochs 30` and a different
fresh folder. Use `--device cpu` for a CPU smoke. See the
[portable setup](../improvement-v1/COMPACT_PORTABLE_RUNBOOK.md).

## Completed result

The original run completed on RTX 4060. See the
[paired validation result and limits](../../results/compact-direction-v1/summary.md).
Keep Smooth L1 as the default for this configuration.
