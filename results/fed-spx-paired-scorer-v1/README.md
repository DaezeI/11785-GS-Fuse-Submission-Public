# Paired Fed/SPX scorer and forecast format (issue #54)

Owner: Adaeze Ilodianya, November 25, 2026. This package is the shared way to score any
20/30-minute Fed/SPX forecaster against the baselines on `FED-PXTX-ALL-REL5-v2` validation. It
contains the scorer, the file format every text arm must write, and the frozen baseline forecasts
to pair against. Test is not loaded. The scorer refuses a test-split reference unless
`--allow-test` is passed, and that is a team decision.

| File | What it is |
|---|---|
| [`scripts/score_fed_spx_paired.py`](../../scripts/score_fed_spx_paired.py) | Scorer and converters. Needs numpy and pandas only. |
| `reference-FED-PXTX-ALL-REL5-v2-val.csv` | The 149 validation releases: `event_id`, `split`, `event_type`, `evidence_class`, `market_et`, `origin_et`, `target_20m_bp`, `target_30m_bp`. SHA-256 `c6150199…f192b`. |
| `baselines-FED-PXTX-ALL-REL5-v2-val.csv` | Frozen validation forecasts of `none+CNN~none~none` (seeds 0/1/2), `none+DLinear~none~none` (seeds 0/1/2) and `none+Ridge~none~none` (seed 0) in the forecast format. SHA-256 `6eef4b9b…e740`. |
| [`metrics.json`](metrics.json) | The baselines scored by this package. |

Both CSVs were converted from the `fed-spx-matched-control-v2` per-release file, whose SHA-256
(`fcfd8790…397a`) matches the published `paired_forecasts_sha256`.

## Forecast file the text arm writes

One CSV per run, UTF-8, header required, one row per release × label × seed:

| Column | Meaning |
|---|---|
| `event_id` | Release ID from the reference file. |
| `label` | Architecture ID from [docs/NAMING.md](../../docs/NAMING.md), with a control suffix where it applies: `<ID>`, `<ID>[text=zero]`, `<ID>[text=shuffled-<donor seed>]`. No spaces or commas. `Persistence` is built in and must not be supplied. |
| `seed` | Integer training seed. Deterministic forecasters use `0`. |
| `forecast_20m_bp`, `forecast_30m_bp` | Forecast log return from the decision-time Close, in basis points. |

The scorer enforces these rules:

- Every `event_id` is in the reference. Each `(event_id, label, seed)` appears once. Values are finite.
- All seeds of one label cover the same releases.
- A label may cover only part of the reference. A shuffled control omits releases with no eligible
  past donor. Each comparison uses the releases both sides cover and reports how many.
- A label may appear in only one of the files scored together.

`scripts/run_fed_spx_text.py` writes `val-paired.npz`, not this CSV. Convert its output with:

```sh
python scripts/score_fed_spx_paired.py from-text-run --npz <run>/val-paired.npz \
    --reference results/fed-spx-paired-scorer-v1/reference-FED-PXTX-ALL-REL5-v2-val.csv \
    --baselines results/fed-spx-paired-scorer-v1/baselines-FED-PXTX-ALL-REL5-v2-val.csv \
    --out <run>/val-forecasts.csv
```

The converter checks that the run's targets match the reference. It writes the real-text arm,
`[text=zero]`, and `[text=shuffled-11|23|37]` for seeds 0/1/2. Shuffled rows are kept only where
all three donor draws exist, as in the training script. With `--baselines`, it also prints how far
the run's text-off output is from the packaged `none+CNN~none~none` forecasts.

## Scoring

```sh
python scripts/score_fed_spx_paired.py score \
    --reference results/fed-spx-paired-scorer-v1/reference-FED-PXTX-ALL-REL5-v2-val.csv \
    --forecasts results/fed-spx-paired-scorer-v1/baselines-FED-PXTX-ALL-REL5-v2-val.csv <run>/val-forecasts.csv \
    --text-arm "Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add" --out <run>/paired-metrics.json
```

- **Subsets:** `ALL` is all 149 releases. `STRICT` drops the inferred evidence classes, leaving 27 releases (`FED-PXTX-STRICT-REL5-v2`).
- **Weights:** every release instant gets total weight 1 within each scored row set.
- **Metrics:** for each label, weighted MAE and direction accuracy of the seed-average forecast. The per-seed means are also reported.
- **Comparisons:**
  - every label is compared with `Persistence`.
  - with `--text-arm`, the arm is also compared with `none+CNN~none~none` and with each of its controls.
  - `--compare LABEL OTHER` adds any other pair.
- **Gain:** `loss(OTHER) − loss(LABEL)`, so a positive gain favours LABEL. The 95% intervals come from a paired bootstrap over 5- and 10-date blocks (seed 23, 2,000 replicates). The interval is omitted when there are fewer than three blocks of dates.
- **Seeds:** the primary gain uses the seed-average forecast, as in `run_fed_spx_text.py`. `seed_mean_loss_*` averages each release's loss over seeds instead, as in `score_fed_spx_matched_control.py`.

## Verification

- **Reproduction:** scoring the packaged baselines reproduces every MAE, direction accuracy, gain
  and interval in `results/fed-spx-matched-control-v2/metrics.json` to within 2e-9 bp. The
  difference comes from rounding the CSVs to 10 significant digits. Gains are compared through
  `seed_mean_loss_*`, the definition that run used.
- **New tests:** `tests/test_fed_spx_paired_scorer.py` has 15 tests. They cover format checks,
  instant weights, gain sign, partial coverage, the STRICT subset, the text-run converter and
  this reproduction.
- **Existing tests:** `endpoint_metrics` and `bootstrap_gain` moved here unchanged and are
  re-exported from `scripts/run_fed_spx_endpoints_price.py`. `tests/test_fed_spx_endpoints.py` and
  `tests/test_fed_spx_text.py` pass. They ran with `exchange_calendars` stubbed out, because that
  package is missing from the local environment.

Baseline gains over `Persistence` on ALL, seed-average forecast, with 10-date-block intervals:

| Comparison label | Gain 20 / 30 min |
|---|---|
| `none+CNN~none~none @ train:FED-PX-REL5-v1 → score:FED-PXTX-ALL-REL5-v2 val` | +0.097 [−0.099, +0.327] / +0.092 [−0.182, +0.382] |
| `none+DLinear~none~none @ train:FED-PX-REL5-v1 → score:FED-PXTX-ALL-REL5-v2 val` | −0.596 [−1.146, −0.130] / −0.408 [−1.126, +0.297] |
| `none+Ridge~none~none @ train:FED-PX-REL5-v1 → score:FED-PXTX-ALL-REL5-v2 val` | −3.140 [−5.674, −1.055] / −3.444 [−7.174, −0.699] |

The [matched-control-v2 summary](../fed-spx-matched-control-v2/summary.md) reports the
seed-mean-loss gains: +0.088 / +0.054 bp for `none+CNN~none~none`. Both definitions agree that it
ties `Persistence`.

## Limits

- **Validation is exploratory:** the `none+CNN~none~none` epochs were chosen on the 207 `FED-PX-REL5-v1` validation rows, which include these 149.
- **STRICT has too few dates for intervals:** 27 releases on 27 dates is under three 10-date blocks.
- **The text arm's own report weights differently:** `run_fed_spx_text.py` keeps full-cohort instant weights on the shuffled subset, while this scorer reweights instants on the common rows. Report this scorer's numbers.
- **Test evidence level is undecided:** the team hasn't chosen between ALL and STRICT for the final test score (#53).
