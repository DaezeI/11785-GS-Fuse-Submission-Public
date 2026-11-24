# Fed/SPX price-only baselines

Owner: Adaeze Ilodianya.

How accurately can we forecast the S&P 500 price path around official Fed
announcements using only the price information available before the release?
This run compares last-close persistence with ridge. See the
[result summary](../../results/fed-spx-price-baselines-v1/summary.md#references-for-the-text-model)
for the role of each reference in the text experiment.

## Fixed comparison

- Use [Joye's alignment v1](../../analysis/macro_releases/price_alignment_v1/README.md)
  from `a4269dc` and the frozen `SP500.csv`. Keep the supplied chronological,
  purged split and the 35-input/140-target OHLC windows. The input cutoff is
  the announcement time. The target starts at or after the five-minute lag.
- **Last-close persistence** repeats the final pre-announcement close for every
  step and OHLC channel. **Ridge** reads the flattened 35 × 4 input and predicts
  the 140 × 4 path. Both input and target are log returns in bp relative to the
  last input close. Ridge uses feature standardization fitted on main train,
  an intercept, and a fixed penalty of 10 on the summed weighted squared loss.
  The two baselines use no text, calendar, or event-type features.
- Fit on 1,013 main training rows. Score the same fit on 215 main validation
  rows and report 85 flagged validation rows separately. Each announcement
  instant has unit weight within its cohort. Validation is descriptive, and
  the runner only loads train and validation.
- Report OHLC path MAE and RMSE over the first 35, 70 and 140 steps. Report
  terminal Close MAE and sign accuracy at each horizon. Accuracy excludes
  realized zero moves. Persistence always predicts zero and abstains from
  direction. Count a zero forecast as a miss on eligible rows and report the
  fraction of zero predictions alongside accuracy.
- Quantify the paired percentage change in path MAE with 2,000 circular
  bootstrap samples of blocks of five and ten consecutive announcement dates
  (seed 23). Positive improvement favors ridge. The intervals describe the
  fitted models on this validation period and account for overlapping paths
  only to the extent captured by these block lengths.

## Reproduce and pair forecasts

From the repository root, with Python, NumPy and pandas installed:

```text
python scripts/run_fed_price_baselines.py --price-csv downloads/SP500.csv --out-dir downloads/fed-spx-price-baselines-v1
```

Use the original frozen price CSV bytes. The runner checks the SHA-256 from
Joye's audit. Keep the output under `downloads/`. It includes
`metrics.json`, `ridge.npz` with the fitted coefficients, and separate
`val-main-paired.npz` and `val-flagged-paired.npz` files. The paired files contain
`event_ids`, `announcement_dates`, `weights`, `targets`, `last_close` and
`ridge`. Join on `event_ids` before computing paired differences. The forecast
and target tensors are channel-last OHLC in bp. In the paired file, `last_close`
is the persistence forecast array. To reuse the scoring convention with an
aligned path, import `score_paths` and pass the same cohort weights.
