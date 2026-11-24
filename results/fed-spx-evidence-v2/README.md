# Fed text evidence v2: corrected release-line parser and wording check

Owner: Adaeze Ilodianya. November 23, 2026. **Re-decision from saved evidence only:** no new
lookups, no prices or returns read, test outcomes untouched. v1 results are left as they were.

## Why

A review of every rejection found three bugs in the minutes, speech and testimony checks
(`scripts/verify_minutes_speech_versions.py`). Each made some releases fail that the evidence rules
would accept.

| Bug | What went wrong | Fix |
|---|---|---|
| Release-line parser | `pdf_info()` read "For release on delivery" only as "H:MM a.m. EDT Month D, YYYY". It missed two-zone lines ("9:30 a.m. EDT (8:30 a.m. CDT)", "12:30 p.m. CDT (1:30 p.m. EDT)"), "Noon", whole hours ("10 a.m."), capital "A.M.", "ET" and a weekday before the date. | `scripts/release_lines.py`. On 930 saved Fed PDFs it reads 746 lines; 9 stay unread (7 have no time, 2 have garbled text). Where the old parser also read a line, the two agree in every case. |
| Stale wording check | Slices s000–s342 used an earlier `compare_words` that did not skip web-only lines such as "Accessible Version". | Recompute the check from the saved page and PDF for every row with `words_match = False`. |
| Web-only lines | Video-player help ("The [Tab] key may be used…") and testimony web notes ("…submitted identical remarks to…") appear only on the web page, never in the released PDF, but counted as wording changes. | Skip them (`EXTRA_WEB_ONLY`). |

`scripts/redecide_fed_document_versions.py` applies all three. For each audited row it confirms the
saved PDF's SHA-256 equals the recorded `pdf_sha256`, replaces only the release-line and wording
fields, re-runs the original `decide()`, and re-applies the verifier's fingerprint rule. Archive
lookups and every other field are kept as recorded. No eligible row became ineligible.

Checks that found no bug: "PDF modified after release" gaps run from 5 minutes to days with no 4–5
hour cluster (no timezone error); every FOMC minutes release record was read; only 1 Board release
lacked a readable release time.

## Effect

| Cohort | v1 train / val / test | v2 train / val / test |
|---|---:|---:|
| `FED-PXTX-ALL-REL5` | 160 / 123 / 121 | **173 / 149 / 172** |
| `FED-PXTX-STRICT-REL5` | 90 / 27 / 26 | 90 / 27 / 30 |
| `FED-PXTX-ALL-OPEN` | 19 / 12 / 22 | **27 / 18 / 34** |

Train/val FOMC minutes and speeches eligible: 60 / 63 → 73 / 87. Validation release instants:
113 → 139.

| Set re-decided | Moved to eligible | Moved to version conflict | Unchanged |
|---|---:|---:|---:|
| Train/val minutes and speeches (all slices, before de-duplication) | 64 | 23 | 505 |
| Test minutes and speeches | 46 | 10 | 81 |
| Test testimony | 5 | 0 | 10 |
| Pre-market minutes, speeches, testimony | 26 | 11 | 160 |
| Train/val testimony re-check | 2 | 2 | 106 |

"Moved to version conflict" rows now have a readable release line but a PDF modified after it.

## Results in this folder set

- [`fed-spx-minutes-speeches-evidence-v2`](../fed-spx-minutes-speeches-evidence-v2/counts.json): train/val minutes and speeches.
- [`fed-spx-testimony-recheck-v2`](../fed-spx-testimony-recheck-v2/counts.json): train/val testimony.
- [`fed-spx-news-cohort-v2`](../fed-spx-news-cohort-v2/counts.json): `FED-PXTX-*-REL5-v2` train/val.
- [`fed-spx-test-text-v2`](../fed-spx-test-text-v2/counts.json): test.
- [`fed-spx-premarket-text-v2`](../fed-spx-premarket-text-v2/counts.json): `FED-PXTX-*-OPEN-v2`.
- [`fed-spx-matched-control-v2`](../fed-spx-matched-control-v2/summary.md): baselines re-scored on v2 validation.

## Open items

- 104 Board train releases marked `not_verified_no_timely_capture_in_checked_index` were never
  given the Board re-check's archive-window search.
- The wording check stays conservative: small letter mismatches and numbers-only differences
  remain conflicts without manual review.
- Statement Aug 1, 2012 differs only by hyphenation ("ex-tend"); left as a conflict.

## Reproduce

Requires the git-ignored evidence folders under `downloads/` from the v1 audits.

```sh
R=downloads/redecide-v2; M=downloads/minutes-speech-evidence-v1
python scripts/redecide_fed_document_versions.py --raw-dirs $M/raw $M/validation-archive-raw \
  --inputs $M/slices/s0*.json $M/slices/s1*.json $M/slices/s2*.json $M/slices/s3*.json $M/wording-rerun.json \
           $M/codex-validation-publisher.json $M/slices/codex-s*.json --out-dir $R/trainval-minutes-speeches
for pair in test-minutes-speeches:downloads/fed-spx-test-text-v1/minutes-speeches \
            test-testimony:downloads/fed-spx-test-text-v1/testimony \
            premarket-minutes-speeches:downloads/fed-spx-premarket-v1/minutes-speeches \
            trainval-testimony:downloads/testimony-recheck-v1; do
  python scripts/redecide_fed_document_versions.py --raw-dirs ${pair#*:}/raw \
    --inputs ${pair#*:}/ledger.json --out-dir $R/${pair%%:*}
done
python scripts/build_minutes_speech_ledger.py --population downloads/fed-spx-endpoints-price-v1/population.csv \
  --inputs $R/trainval-minutes-speeches/s0*.v2.json $R/trainval-minutes-speeches/s1*.v2.json \
           $R/trainval-minutes-speeches/s2*.v2.json $R/trainval-minutes-speeches/s3*.v2.json \
           $R/trainval-minutes-speeches/wording-rerun.v2.json \
           $R/trainval-minutes-speeches/codex-validation-publisher.v2.json $R/trainval-minutes-speeches/codex-s*.v2.json \
  --out-dir results/fed-spx-minutes-speeches-evidence-v2
python scripts/finalize_testimony_recheck.py $R/trainval-testimony/ledger.v2.json results/fed-spx-testimony-recheck-v2
python scripts/build_fed_spx_news_cohort.py --price-csv ../fed-person3/downloads/SP500.csv \
  --out-dir results/fed-spx-news-cohort-v2 --evidence-version v2
# Test and OPEN: copy the v1 population, statements/ledger.json and board/final into a *-v2-src folder,
# add minutes-speeches/ledger.json (and testimony/ledger.json for test) from $R, then:
python scripts/summarize_fed_spx_test_text.py --src downloads/fed-spx-test-text-v2-src --out results/fed-spx-test-text-v2
python scripts/summarize_fed_spx_test_text.py --src downloads/fed-spx-premarket-text-v2-src \
  --population population.csv --out results/fed-spx-premarket-text-v2   # then rename the ledger to premarket_text_ledger.csv
```

Environment: set `PYTHONPATH` to `downloads/fed-spx-deps` and `downloads/codex-validation-deps`.
