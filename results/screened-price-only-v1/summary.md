# Screened price-only training comparison

**Owner Nicole · Completed November 18, 2026.** Both price-only models
completed their smoke tests, training and evaluation on one H200 each.
The comparison used 2,236 identical held-out examples across 845 dates.

## What we learned

The news-trained package lowered 140-bar price error by **12.51%**
(seed 0, paired 95% interval 8.40%–17.85%) and **26.18%** (seed 1,
21.76%–31.34%) versus the independently trained price-only models.
Both gains persisted with 5-, 10-, 20- and 40-date bootstrap blocks.

The simple rule of repeating the last observed prices beat both trained
models at all three forecast lengths. At 140 bars it lowered error by
**8.21%** and **18.18%** relative to the news models. Bypassing fusion in
the news-trained checkpoint also improved point estimates at every
horizon. The earlier [seed-0](../screened-price-v1/summary.md) and
[seed-1](../screened-price-seed1-v1/summary.md) controls found essentially
the same error with correct and shuffled articles. The evidence supports
a difference between these training setups, with little measured
advantage for the particular article and no win against persistence.

| Training seed | Last-price rule MSE | News-trained, fusion bypassed MSE | News model MSE | Trained price-only MSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.010442 | 0.011157 | 0.011376 | 0.013002 |
| 1 | 0.010442 | 0.011649 | 0.012763 | 0.017290 |

The table shows 140-bar MSE in training-standardized OHLC price units.
Lower is better. All horizons and intervals are in [metrics.csv](metrics.csv).
The zero-return endpoint reference also had lower close-return MAE than
the trained forecasts at all horizons for both seeds. At 140 bars,
always-up accuracy was 52.59%, compared with 51.65% and 51.16% for the
news forecasts. See [seed 0](seed-0.json) and [seed 1](seed-1.json) for
those secondary metrics and their eligibility counts.

## Training design and verification

- The [runner](../../scripts/research_cycle/train_price_only.py) used the
  original price-model code and each seed’s own selected price-only
  pretraining checkpoint. The source logs show both saved their Stage-1
  checkpoint at the last pretraining epoch. No article model was
  initialized in the new runner. It used the frozen MOMENT backbone,
  the same price projection, multiscale features, raw-price residual
  and decoder. The price-only model had no text warm-up, fusion, or
  text-related losses.
- The two models each ran two event-training epochs at batch 8, with
  the existing price regression loss, optimizer settings, TF32,
  screened windows, and training-only scaler. Checkpoint selection
  used the lowest validation MSE. Seeds 0 and 1 selected epochs 2
  and 1, respectively.
- Both GPU smoke tests reproduced the original fusion-bypass
  predictions exactly on 16 examples when given the same original
  checkpoint. They also exercised new training, checkpoint save
  and reload. The full evaluation verified exact agreement of
  input and target arrays and ordered test IDs with each saved
  news prediction archive. The scheduler reported exit 0 for both
  jobs. Aggregate records include source and artifact fingerprints.
- The bootstrap resamples consecutive blocks of event dates. Its
  uncertainty is conditional on the trained seed and these test
  examples. This comparison changes the training package, including
  the warm-up and fusion. It cannot assign the gain to the
  meaning of the article. The registered data screen leaves retained
  publication times unverified.

## Implication for the paper

Use both simple references in the main results table. Report the
training-package gain against the independently trained model and
the absence of a gain against persistence. Keep the correct-versus-
shuffled result alongside both. Adaeze’s upcoming [return comparisons](../screened-return-r0-r2-v1/summary.md)
address a different target. After collecting the existing paired
return outputs, a train/validation-selected shallow text model would
help determine whether the articles contain signal the current fusion
path misses.
