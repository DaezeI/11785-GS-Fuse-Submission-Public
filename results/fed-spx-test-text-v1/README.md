# Held-out test: exact-version text check

Owner: Adaeze Ilodianya. Run November 23, 2026. **Text-eligibility audit only.** No test price
value, return, label or model output was read. The population was built from bar timestamps
alone (`scripts/build_fed_spx_test_population.py`), and the test stays reserved for scoring.

Train and validation releases were already text-checked; this applies the same rules to test
so the frozen comparison knows its test rows before any scoring.

## Counts

216 of the 218 main test rows pass the 20/30-minute clock rules (2 fall outside the XNYS
session). 121 of those have exact-version text evidence.

| Type | Test rows | Eligible | Rule |
|---|---:|---:|---|
| FOMC statements | 21 | 21 | Minutes reproduce the text and its release time |
| FOMC minutes | 19 | 16 | Official release record, unmodified PDF |
| Speeches | 118 | 41 | PDF "For release on delivery" line, unmodified PDF |
| Board releases | 43 | 39 | Release line plus release-day archive copy (1 by a pre-cutoff copy) |
| Testimony | 15 | 4 | Archive copy before the cutoff |
| **Total** | **216** | **121** | 104 release instants, 98 dates |

| Evidence allowed | Test rows |
|---|---:|
| All audited classes | 121 |
| Without the Board release-day capture class | 83 |
| Without the speech release-line class | 80 |
| Strictest (official record or pre-cutoff archive copy only) | 26 |

The rest are version conflicts (21 speeches, 3 minutes, 3 Board, 6 testimony) or have
insufficient evidence. Exact dispositions per event are in [test_text_ledger.csv](test_text_ledger.csv),
and all counts are in [counts.json](counts.json).

## Matching train/validation change

Testimony had been proved only through House committee PDFs, leaving 110 train/validation
testimony rows unchecked. They were checked with the same document rule used here
([testimony re-check](../fed-spx-testimony-recheck-v1/counts.json)): 2 validation rows became
eligible. The matched cohort is now **160 train / 123 validation** (113 validation instants).

## Reproduce

```sh
python scripts/build_fed_spx_test_population.py --price-csv ../fed-person3/downloads/SP500.csv \
    --out downloads/fed-spx-test-text-v1/test_population.csv
python scripts/verify_fomc_statement_versions.py --allow-test --population downloads/fed-spx-test-text-v1/test_population.csv \
    --catalog analysis/macro_releases/fed_announcements_v2.csv --minutes-overrides analysis/macro_releases/fomc_minutes_overrides.json \
    --events <test stmt_ ids> --raw-dir downloads/fed-spx-test-text-v1/statements/raw --out downloads/fed-spx-test-text-v1/statements/ledger.json
python scripts/verify_minutes_speech_versions.py --allow-test --population downloads/fed-spx-test-text-v1/test_population.csv \
    --catalog analysis/macro_releases/fed_announcements_v2.csv --raw-dir downloads/fed-spx-test-text-v1/minutes-speeches/raw \
    --out downloads/fed-spx-test-text-v1/minutes-speeches/ledger.json      # repeat with --types fed_testimony
python scripts/recheck_board_archive.py --ledger downloads/fed-spx-test-text-v1/board/stub_ledger.csv \
    --catalog analysis/macro_releases/fed_announcements_v2.csv --out downloads/fed-spx-test-text-v1/board/shard0.jsonl --budget 3600
python scripts/finalize_board_recheck.py downloads/fed-spx-test-text-v1/board/shard0.jsonl downloads/fed-spx-test-text-v1/board/empty.jsonl \
    analysis/macro_releases/fed_announcements_v2.csv downloads/fed-spx-test-text-v1/board/final
python scripts/summarize_fed_spx_test_text.py --out results/fed-spx-test-text-v1
```

Dependencies: set `PYTHONPATH` to `downloads/fed-spx-deps` and `downloads/codex-validation-deps`
(html5lib). Raw pages stay in `downloads/` (not committed).
