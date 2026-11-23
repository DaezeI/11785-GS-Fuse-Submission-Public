# Fed/SPX news evidence and matched cohort

Owner: Adaeze Ilodianya. This is the frozen PR #46 Fed/SPX cohort, based on
endpoint commit `c8c89b557f3a1a934b993d9241dea1d2de76eae2`. It supports a
matched comparison of SPX forecasts with and without Fed news. The broader
U.S. macro coverage study has its own [candidate count](../macro-coverage-v1/README.md).

- [Merged evidence ledger](merged_evidence_ledger.csv) lists all 1,185 unique
  frozen train and validation event IDs, including unsuccessful lookups.
- [Eligible cohort](eligible_cohort.csv) identifies the candidate news input,
  artifact or body fingerprint, source, evidence class, and frozen clocks.
- [Counts](counts.json) gives all dispositions by event type and split, along
  with row counts, unique release instants and dates under each eligibility
  scenario.
- [Independent audit comparison](independent-comparison/comparison.json) compares
  the minutes and speech dispositions on the same event IDs and cutoffs.
  The comparison's primary label refers to the ledger built under the primary
  rules, including the Codex validation continuation.

The principal evidence sources were [official minutes for statements](../fed-spx-text-evidence-v1/README.md),
[timestamped House PDFs for testimony](../fed-spx-board-testimony-evidence-v1/README.md),
and [the primary rules for minutes and speeches](../fed-spx-minutes-speeches-evidence-v1/README.md).
The minutes class `eligible_official_release_record_unmodified_file` and
speech class `eligible_document_release_line_unmodified_file` rely on
inferences about the version publicly available at the release minute.
Keep the evidence class in any comparison. The count scenarios show the
superset, the cohort without the speech release-line class, and the cohort
without either inferred class. The manifest carries the audited dispositions
without treating either inferred class as proof from a timestamped capture.

| Evidence included | Train rows / instants / dates | Validation rows / instants / dates |
| --- | ---: | ---: |
| All audited eligibility classes | 150 / 150 / 150 | 88 / 87 / 84 |
| Without the speech release-line class | 117 / 117 / 117 | 36 / 36 / 35 |
| Without either inferred release class | 90 / 90 / 90 | 25 / 25 / 25 |

The validation continuation covered all 106 previously unattempted events.
It added 50 speech and 11 minutes candidates under the inferred release-time
rules. The remaining rows had 20 version conflicts requiring review and 25
with insufficient evidence. The archive follow-up did not change the
eligibility count. The independent pass found zero eligible rows under
its stricter rule across the same 472 events. The difference comes from
using combinations of official release records or document release lines
with PDF metadata and corroboration.

The builder reruns the normal price loader with the frozen price, Fed catalog
and alignment hashes, then compares every event and clock with the frozen
population. The manifest flags complete 20- and 30-minute targets per row.
Same-instant rows share a release ID. `frozen_events_at_release` describes
the full population at that instant and `eligible_events_at_release` counts
matching evidence rows before any sensitivity filters. Apply the chosen
evidence filter before bundling documents or weighting forecast events.
The two PDF timestamp flags show where the server's reported write clock
falls after the cutoff or forecast origin. A server header is not a
public-access log, and an archive-backed row may still carry a later
server timestamp after a site migration.

For PDF evidence, check the artifact hash before extracting text and record
the extracted input's separate hash. For the statement evidence, hash the
normalized approved body. URLs and fingerprints identify evidence.
Raw research inputs stay in their existing local locations. No model was
trained and the held-out test was left reserved.

```powershell
python scripts/build_fed_spx_news_cohort.py --price-csv ../fed-person3/downloads/SP500.csv --out-dir results/fed-spx-news-cohort-v1
python scripts/compare_fed_text_audits.py --codex results/fed-spx-minutes-speeches-codex-v1/codex_minutes_speeches_evidence_ledger.csv --primary results/fed-spx-minutes-speeches-evidence-v1/minutes_speeches_evidence_ledger.csv --out results/fed-spx-news-cohort-v1/independent-comparison
```
