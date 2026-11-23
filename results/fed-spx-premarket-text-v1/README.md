# Pre-market Fed releases: `FED-PX-OPEN-v1` and `FED-PXTX-ALL-OPEN-v1`

Owner: Adaeze Ilodianya. Built November 23, 2026. **Cohort audit only.** No price value,
return, label or model output was read for any split; prices were read for bar timestamps only.

## Rule (open-anchored clock, `OPEN`)

Release-anchored rows (`REL5`) drop Fed releases that come out before the 09:30 open. This adds
them under a separate clock, defined in `scripts/build_fed_spx_premarket_population.py`:

- **Qualifies:** verified market time before 09:30 ET on a day with a regular 09:30 XNYS open.
  Press-conference transcripts and midnight date-only entries are excluded.
- **Input:** the same 35 OHLC bars alignment v1 assigns, all ending by the release time.
- **Decision time:** 09:35, the end of the first regular bar. Targets: the Close at 09:55 and 10:05
  against the 09:35 Close (labels not built yet).
- **Text cutoff:** 09:30, five minutes before the decision time, the same rule as `REL5`.
- **Splits:** from alignment v1. Windows that would share a bar with another split are dropped
  (none were).

Alignment v1 is unchanged.

## Counts

| Cohort | Train | Val | Test |
|---|---:|---:|---:|
| `FED-PX-OPEN-v1` (price window usable) | 153 | 37 | 43 |
| `FED-PXTX-ALL-OPEN-v1` (text proven public by 09:30) | **19** | **12** | **22** |
| `FED-PXTX-STRICT-OPEN-v1` (official record or pre-cutoff archive copy) | 0 | 1 | 6 |

Of 309 pre-market rows in alignment v1, 75 press-conference transcripts and 1 midnight entry were
excluded, leaving 233. Text eligibility used the same audit scripts and rules as `REL5`:

| Type | Usable | Text verified | Main reason for the rest |
|---|---:|---:|---|
| Speeches | 163 | 46 | 100 have only a scheduled delivery time; 17 changed wording |
| Board releases | 31 | 6 | 19 have no archive copy near the release; 6 changed wording |
| Testimony | 33 | 1 | 29 insufficient evidence |
| FOMC statements and other policy releases | 5 | 0 | Emergency actions whose minutes give no release time |
| FOMC minutes | 1 | 0 | Insufficient evidence |

Combined with `FED-PXTX-ALL-REL5-v1` (160 / 123 / 121), `FED-PXTX-ALL-REL5+OPEN-v1` would have
**179 / 135 / 143** rows. Combining mixes two forecast clocks, so label every result by clock.

- [Per-release ledger](premarket_text_ledger.csv) with disposition, reason, fingerprint and
  evidence URL.
- [Counts](counts.json) by type, evidence class and split.
- [Price population summary](price_population.json), including the population SHA-256.

## Reproduce

```sh
python scripts/build_fed_spx_premarket_population.py --price-csv ../fed-person3/downloads/SP500.csv \
    --out downloads/fed-spx-premarket-v1/population.csv
# Text checks: verify_fomc_statement_versions.py (statement IDs), verify_minutes_speech_versions.py
# --types fomc_minutes fed_speech fed_testimony, recheck_board_archive.py + finalize_board_recheck.py,
# each with --population downloads/fed-spx-premarket-v1/population.csv and --allow-test.
python scripts/summarize_fed_spx_test_text.py --src downloads/fed-spx-premarket-v1 \
    --population population.csv --out results/fed-spx-premarket-text-v1
```

Set `PYTHONPATH` to `downloads/fed-spx-deps` and `downloads/codex-validation-deps`. Raw pages stay
in `downloads/`.
