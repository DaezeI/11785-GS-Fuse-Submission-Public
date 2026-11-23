# Independent Fed/SPX minutes and speech text audit

Adaeze Ilodianya owns the endpoint design. This pass used the frozen PR #46
population from head `c8c89b557f3a1a934b993d9241dea1d2de76eae2`. The ledger
has one disposition per event and was produced without using the primary ledger.

| Scope | Train | Validation | Current bodies matching catalog | Exact bodies established by cutoff |
| --- | ---: | ---: | ---: | ---: |
| FOMC minutes | 87 | 17 | 103 | 0 |
| Fed speeches | 273 | 95 | 367 | 0 |
| Total | 360 | 112 | 470 | 0 |

All 104 Fed minutes release notices included a release clock and linked to
the minutes document. They establish the release of the named document.
The current PDF, matching HTML digest, and editable PDF clock metadata
do not independently establish the wording served at release. This pass
retrieved 103 minutes PDFs and 317 speech PDFs. PDF modification metadata
was after the frozen cutoff for 5 minutes and 50 speeches. That flag
cannot tell whether the wording, formatting, or metadata changed.
For speeches, 304 of the linked PDFs gave a release clock on the cover.
This is evidence of the intended release schedule. The pass found no
record linking actual availability of those exact bodies to the cutoff.

Two fresh current-body fingerprints differed from the catalog. They are the
[April 2008 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20080430.htm)
and [Fischer's March 2016 speech](https://www.federalreserve.gov/newsevents/speech/fischer20160307a.htm).
These are extraction or version discrepancies and do not establish edits
to either historical body.

The [July 6, 2016 Fed notice](https://www.federalreserve.gov/newsevents/pressreleases/other20160706a.htm)
records that one media organization prematurely transmitted some embargoed
information and that others experienced delays. The July 6 train row is
flagged for review. The [March 2013 minutes incident](https://oig.federalreserve.gov/reports/board-executive-summary-20130827a.htm)
is outside the frozen population. The Inspector General's
[publication-process audit](https://oig.federalreserve.gov/reports/board-full-report-20130827a-intro.htm)
describes the approval and posting procedure.

The fresh archive pilot found only post-cutoff captures for four events.
For example, the nearest capture returned for Powell's August 17, 2021
welcome remarks was at 17:33:04 UTC, 3 minutes and 4 seconds after the
17:30 UTC cutoff. An Arquivo.pt request timed out. The historical replay
check was a pilot. An exhaustive archive or wire search across all 472
events remains outside these results. The alternative proof sources in
The primary ledger can be evaluated event by event when available.

## Reproduce and compare

Run from the endpoint checkout with the frozen population and catalog in
place. Python needs `requests`, `beautifulsoup4`, `lxml` and `pypdf`.

```powershell
python scripts/audit_fed_minutes_speeches_independent.py --out results/fed-spx-minutes-speeches-codex-v1
python scripts/audit_fed_minutes_speeches_independent.py --recover-minutes-pdfs --out results/fed-spx-minutes-speeches-codex-v1
python scripts/audit_fed_minutes_speeches_independent.py --archive-pilot --out results/fed-spx-minutes-speeches-codex-v1
python scripts/compare_fed_text_audits.py --codex results/fed-spx-minutes-speeches-codex-v1/codex_minutes_speeches_evidence_ledger.csv --primary <primary-ledger.csv> --out results/fed-spx-minutes-speeches-comparison-v1
```

The comparison checks identical event IDs, splits, and cutoffs and lists
differences in eligibility or body fingerprints for review. A fingerprint
difference can come from extraction or normalization. The frozen split
and cutoffs stayed fixed. No training or held-out scoring was performed.
