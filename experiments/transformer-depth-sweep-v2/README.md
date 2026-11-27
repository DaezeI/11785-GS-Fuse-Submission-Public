# Concurrent Transformer depth 4 / 8 / 12 comparison

Owner Nicole. Does depth improve validation MAE? This replaces the canceled
sequential attempt `transformer-depth-sweep-v1`, whose partial depth8 training
is preserved privately and excluded. Reuse completed depth4 from PR #73.

Compare `none+Transformer128x4~none~none`, `none+Transformer128x8~none~none`
and `none+Transformer128x12~none~none` on `SPX-PX6-GRID5-v1`: 204,694 train /
40,548 validation windows. Only depth changes. Width128, heads4, FFN512,
dropout0.1, 35-bar OHLC histories, six supervised future positions, one shared
learned query, temporal PE, linear input/readout, train-only RMS preprocessing,
SmoothL1, full batch204,694, BF16/checkpointing, AdamW/LR schedule and 1,000
updates stay fixed. [Configuration](config.json).

Depth8 and12 start as **two concurrent processes sharing one H200 allocation**.
Each confirms the exact fixed batch for ten full training updates and fails
instead of shrinking it. Each has a three-hour safety ceiling to accommodate
shared-GPU contention. Incomplete update budgets cannot enter the comparison;
a child failure stops the other child and records the error. No extra GPU or
reserved test is used.

```sh
python scripts/run_prediction_token_depth_sweep.py --prices /path/to/SP500.csv \
  --out /fresh/run --depth4 /reference/depth4 --code-commit COMMIT
```

All rows are evaluated at update0 and every25 updates through1,000. Minimum
mean six-horizon validation MAE selects each checkpoint. After both fits finish,
the controller automatically verifies identical targets/timestamps/scales and
fixed settings, recomputes MAEs from predictions, and saves best/final MAE,
all six horizon scores, full train/validation curves, runtime/parameter counts
and paired date-block intervals for8vs4,12vs4,12vs8.

After both children enter formal training with finite losses and nonzero
gradients, local monitoring stops at the user's request. The remote controller
continues without a local monitor. Each depth has one fit under this fixed
recipe; validation selects checkpoints and depth, and intervals do not adjust
selection. Equal update/sample budgets do not imply equal compute or elapsed
runtime, especially with GPU contention. This is exploratory validation.
