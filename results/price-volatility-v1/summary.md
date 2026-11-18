# Price controls by pre-event volatility

**Status:** exploratory CPU analysis of the completed screened-price-v1 run; no new training or evaluation. One checkpoint, 2,236 test events, and three shuffle draws. The saved analysis already contained these subgroups; this pack presents them and adds block-length sensitivity.

Correctly matched and shuffled text have almost identical prefix OHLC MSE in every volatility group at horizons 35, 70, and 140. Every corresponding 95% interval includes zero for each tested block length (5, 10, 20, and 40 global event dates). For high volatility at H140, the saved gain is −0.00093% (20-date interval −0.02961% to 0.02422%). Positive gain means matched text has lower MSE than the named control: `100 × (1 − matched MSE / control MSE)`.

For every group, horizon, and block length, the interval for matched text versus **zeroed article tokens** is negative, favoring that inference intervention. Versus **fusion bypass**, intervals are negative at H35 and H70 and include zero at H140 in all groups and tested block lengths. The two controls modify inference differently and are not independently trained time-series-only models. These results do not establish a difference between volatility groups; subgroup contrasts and multiplicity were not tested or adjusted.

## Grouping and uncertainty

Volatility is the standard deviation of log returns of the 35 pre-event raw close prices. Training-only tercile cutoffs are 0.000411070475820452 and 0.0006358354003168641; test events are assigned using those fixed thresholds. Low, medium, and high contain 521, 697, and 1,018 events across 200, 275, and 382 distinct dates respectively. Group dates can overlap. This is prior volatility, not a measure of future shocks. MSE is averaged over each forecast prefix and all four OHLC channels in training-standardized price units.

The paired bootstrap samples circular contiguous blocks from the **global** event-date sequence, retaining all paired rows for each selected date, then evaluates each subgroup. Each interval uses 1,000 replicates and seed 23. They are conditional on the single checkpoint and the three supplied shuffles, not training-seed uncertainty; block length does not guarantee all dependence is captured. Retained article publication times are unverified, and shuffled donors can be from later test dates. Do not generalize this single-checkpoint exploratory result to all text models.

## Files, provenance, and regeneration

- `saved_subgroups.csv` and `volatility-comparisons.png`/`.pdf` reproduce the existing saved aggregate estimates and 20-date intervals; each panel has its own y scale. `saved_aggregate.json` contains the extracted subgroup records and source JSON hashes.
- `block_sensitivity.csv` adds float64 recalculation for 5/10/20/40 dates, using the same bootstrap procedure. Recomputed groups and MSE agree with the original saved data within relative tolerance 2e-6; the largest difference in gain or interval endpoint at 20 dates is 0.0000249 percentage points, reflecting input-precision arithmetic. The original values were not overwritten.
- Analysis based on repository main commit `c89a8deb3674d9fa8094f4ee7b444eed4c250ea0`; source training code commit `bb09396881cc92c5e654c4210944ce5c6da17ced`; checkpoint SHA-256 `5358bbf48b4061ab325eb2a13390d7acdcb75ad7601f6969d3bb73152a2a2a52`; snapshot SHA-256 `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`; manifest SHA-256 `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`. See [baseline provenance](../screened-price-v1/provenance.json).
- Prediction NPZ SHA-256: zeroed tokens `fcc5f06018ef63c5137be2b74d7b3777c8a7ce3bc0f92775f056c1be5921a84f`; bypass `000274d307ee4e6f952102ffc651c0945356a6413aea8e924752d9b3fa03945c`. Local copies matched pre-existing source checksums. The source aggregate JSON hashes are in `saved_aggregate.json`.

From the repository root, `python results/analyze_price_volatility.py` rebuilds the saved-estimate CSV and figure. To additionally regenerate sensitivity, supply both verified private files with `--zero-npz <zeroed-predictions.npz> --bypass-npz <bypass-predictions.npz>`. The script checks their hashes, metadata, group counts, and MSE agreement. It calls the existing `scripts/analyze_text_predictions.py` bootstrap and volatility functions; raw predictions are not included in this result pack.
