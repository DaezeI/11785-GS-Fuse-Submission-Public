# Adaeze’s R0–R2 results and the zero-return reference

The first three full return runs finished on November 18. Their mean absolute
error was nearly identical. A simple zero-return forecast had slightly lower
test error than all three on the same 2,236 screened examples. Predicting
the index would go up every time landed near the models on direction.

| Forecast | Mean absolute error (basis points) | Mean squared error (basis points²) | Up/down accuracy at 35, 70 and 140 bars |
| --- | ---: | ---: | --- |
| R0, return reference | 80.918 | 12,992.88 | 52.33%, 50.81%, 52.50% |
| R1, add direction loss | 81.077 | 13,056.19 | 50.49%, 50.63%, 52.86% |
| R2, change gate supervision | 81.080 | 13,057.05 | 51.30%, 50.72%, 52.73% |
| Zero return or always up | 80.487 | 12,692.11 | 53.26%, 52.46%, 52.59% |

The zero-return reference predicts zero for every forecast channel. The
separate always-up reference is used for the direction column. Direction
accuracy excludes exactly flat moves and uses 2,236, 2,232 and 2,236 eligible
events at 35, 70 and 140 bars, respectively.

Adaeze’s selected test metrics and scheduler accounting corroborate the
[owner handoff](../../docs/updates/2026-11-18-adaeze-r0-r2-results.md). To
reproduce the simple reference, use the frozen price CSV and screened test
manifest with `load_price_timeline` and `window_for_event` from
[`data/c1_protocol.py`](../../data/c1_protocol.py). Use the agreed
`category4_1400_strict_before` rule and 35 input / 140 forecast bars. For each
test ID, apply `prices_to_log_returns_bp` from
[`data/forecast_task.py`](../../data/forecast_task.py) to the four target
channels, using the last input close as the anchor. The aggregate errors are
the mean absolute and mean squared target returns. The always-up accuracy is
the fraction of positive realized close returns among nonflat outcomes.
The run’s manifest and price hashes, ordered test IDs and eligibility counts
matched the saved screened price evidence. [Aggregate evidence](metrics.json)

These are descriptive differences on one training seed. They do not locate
the cause of the gate behavior or give uncertainty on the R0–R2 model
differences. R3 and the paired article controls were queued at the November
18 check. The article controls are needed before concluding whether these
changes help the models use the correct text. Retained article publication
times remain unverified.
