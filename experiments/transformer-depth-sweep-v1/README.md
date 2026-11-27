# Transformer depth 4 / 8 / 12

Owner Nicole. Requested comparison: does depth improve validation MAE?
This branch is stacked on PR #73; depth 4 reuses the completed immutable
`shared-token-transformer-v1` fit rather than retraining an identical reference.

Only layer count changes between `none+Transformer128x4~none~none`,
`none+Transformer128x8~none~none`, and `none+Transformer128x12~none~none`.
All train on `SPX-PX6-GRID5-v1` (204,694 windows) and use identical 40,548-window
validation. Width 128, four heads, FFN 512, dropout 0.1, 35-bar history, six
supervised targets (+5 through +30 minutes), shared prediction vector/temporal PE,
linear input/readout, preprocessing, loss, batch, optimizer and schedule stay fixed.
[Machine-readable configuration](config.json).

Each depth gets 1,000 full-batch updates. Evaluation uses all training and
validation rows at update zero and every 25 updates; minimum mean six-horizon
validation MAE selects its checkpoint. Full traces and final checkpoints are
also retained. Depth 8 and 12 run sequentially on one H200; exact batch 204,694
must pass ten full training updates with the same checkpointing/precision as
depth 4. If it fails, stop rather than changing batch and confounding depth.
The new runs each have a two-hour time ceiling as a safety bound; incomplete
update budgets must not enter the completed comparison.

The automatic comparison validates identical data audit, preprocessing scales,
validation timestamps/targets and fixed settings, then independently recomputes
MAEs from saved per-window predictions. It writes the best/final MAE table,
all six horizon scores, training/validation curves, parameter counts/runtime,
and paired 5-/10-date-block intervals for 8 vs 4, 12 vs 4 and 12 vs 8.

```sh
python scripts/train_prediction_token_transformer.py --prices /path/to/SP500.csv \
  --out /fresh/depth8 --code-commit COMMIT --depth 8 \
  --fixed-batch-size 204694 --max-updates 1000 --eval-every 25 \
  --max-training-seconds 7200
# Repeat depth 12 sequentially, with a distinct fresh output directory.
python scripts/compare_prediction_token_depths.py --depth4 /reference/depth4 \
  --depth8 /fresh/depth8 --depth12 /fresh/depth12 --out /fresh/comparison
```

The question concerns validation under this fixed recipe. Each depth has one fit;
validation chooses checkpoints and compares depths. Conditional bootstrap
intervals do not adjust that selection. Equal updates/sample presentations do
not imply equal FLOPs or runtime, and a depth that needs different optimization
is not tested by this comparison. No text experiment or reserved test is scored.
Repeated local progress polling remains disabled at the user's request; the
remote batch pipeline automatically generates the comparison when both fits finish.

## Superseded before comparison

The user requested concurrent starts. The sequential job was canceled after depth8 entered training; partial artifacts remain private and are excluded. Continue with [concurrent v2](../transformer-depth-sweep-v2/README.md).
