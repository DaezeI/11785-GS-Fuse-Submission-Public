# November 23, 2026: Fed/SPX exact-version text evidence and dataset scale

Owner: Adaeze Ilodianya. Branch `adaeze/fed-spx-compact-endpoints` (PR #46 head `c8c89b5`).
Status: data audit only. No training ran, and the held-out test was not read. Every file below is
uncommitted in the endpoint checkout and is meant for a draft PR stacked on #46.

## Protocol decisions (Adaeze)

- **Exact-version rule kept:** the model sees only wording that was public by its text cutoff.
  Proof may be an official record that reproduces the released text and ties it to its release
  time; a Wayback copy is not required.
- **Release times:** use each release's actual time (12:30, 2:15 or 2:00).
- **Processing buffer:** keep five minutes. For FOMC statements, shortening it would recover no
  rows.
- **Horizons:** count eligibility for the 20- and 30-minute horizons separately, and compare both
  arms on identical rows.
- **Macro releases:** use a document-based rule, recorded as a limitation. A release qualifies when
  its own embargo time is at or before the cutoff, the file was prepared before release, and archive
  copies show no change (or a release-day copy has the same bytes). Archive copies are corroboration
  where they exist; their absence alone does not fail a row.

## Results

| Evidence | Train | Val | Record |
|---|---:|---:|---|
| FOMC statements, minutes reproduce the text and release time | 89 | 18 | [statement audit](../../results/fed-spx-text-evidence-v1/README.md) |
| Minutes and speeches, inferred release-time classes | 60 | 63 | [minutes/speech ledger](../../results/fed-spx-minutes-speeches-evidence-v1/README.md), validation continued by Codex |
| Testimony, timestamped House PDFs | 1 | 7 | [Board and testimony audit](../../results/fed-spx-board-testimony-evidence-v1/README.md) |
| Combined eligible cohort before the Board re-check | 150 | 88 | [news cohort](../../results/fed-spx-news-cohort-v1/README.md) |
| Board releases, document rule with release-day capture | 10 | 33 | [Board re-check](../../results/fed-spx-board-recheck-v1/README.md) |

- **Statements:** for FOMC statements, price windows lose no rows. All 110 in-session statements
  have 35 input bars, complete 20- and 30-minute targets, and zero shared train/validation bars.
- **Combined count:** with the Board re-check the eligible count is about 160 train / 121
  validation. The cohort file must be rebuilt with `scripts/build_fed_spx_news_cohort.py`.
- **Wording changed after release:**
  - The July, September and November 2018 FOMC minutes.
  - Two 2020–2021 Board releases.
  - The October 11, 2019 statement page, which gained a link line, so its catalog hash names a
    version that did not exist at release.
  - Eight 2020 Board pages, which gained term-sheet links after release; their release-day body is
    fingerprinted instead.
- **Macro coverage:** [macro coverage](../../results/macro-coverage-v1/README.md) (BLS jobs, CPI,
  PPI, JOLTS; BEA GDP, personal income; weekly claims) gives 2,158 candidate training events.
  Neither forecast rule (intraday and premarket) reaches 2,000 alone. Text checks have not started.
- **Macro pilot:** a 12-release pilot found same-bytes release-day copies for 4. Under the
  document rule, 6 pass, 1 fails (CPI 2011, modified after release) and 5 are pending.

## Unsuccessful or stopped work

- **Minutes and speech audit:** stopped at 366 of 472 rows by decision; Codex finished the
  validation rows.
- **Wayback:** almost no archive copies exist before a release, and older Fed PDFs were rewritten
  during site migrations in 2014–2017, so most pre-2017 minutes and speeches remain insufficient.
- **bls.gov:** refuses non-browser clients; BLS files were checked through Wayback copies only.
- **DOL claims:** the 2009 claims file was not found at the expected address.

## Limitations

- **Inferred release-time classes:** the minutes, speech, Board and macro classes rely on
  inferences (release records or release lines plus file dates or release-day copies). Keep the
  evidence class in every comparison.
- **Validation size:** validation remains small for detecting MAE differences of the size seen in
  the price-only run.
- **Premarket rule:** the premarket forecast rule for 8:30 releases is a separate design that has
  not yet been specified with Joye.

## Next steps

1. Rebuild the eligible cohort with the Board re-check, and re-check the 110 incomplete testimony
   rows the same way.
2. Build the verified text files and the text-plus-price runner. Then refit the price-only CNN on
   the same training rows and compare at 20 and 30 minutes on identical validation rows.
3. **Jessica:** add flags and a verified-body mode to the catalog for the rows above.
   **Joye:** own or review any alignment extension to macro releases.
4. Commit everything on a stacked draft PR. GitHub write access was unavailable from the
   assistant's workspace in this session.
