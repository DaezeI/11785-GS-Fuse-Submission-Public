# Exact-version text evidence for FOMC minutes and Fed speeches

Scope: the 472 `fomc_minutes` and `fed_speech` rows of the frozen endpoint population
(`downloads/fed-spx-endpoints-price-v1/population.csv`, PR #46). This file records
The original partial pass and the subsequent validation continuation.
The held-out test remains reserved. No training was run.

## Original partial pass

| Type | Split | Eligible | Version conflict | Insufficient evidence | Blocked | Not attempted | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| fomc_minutes | train | 27 | 6 | 54 | 0 | 0 | 87 |
| fomc_minutes | val | 0 | 0 | 0 | 0 | 17 | 17 |
| fed_speech | train | 33 | 29 | 211 | 0 | 0 | 273 |
| fed_speech | val | 2 | 0 | 4 | 0 | 89 | 95 |
| **Total** | | **62** | **35** | **269** | **0** | **106** | **472** |

The original pass left 106 validation events unattempted. Current exact
counts are in `counts.json`, and one row per event is in `minutes_speeches_evidence_ledger.csv`.
It uses the statement ledger's columns first (merge on `event_id`) and adds `event_type`,
`status_class`, `reason` and the file-level evidence fields.

## Evidence rules

All times are compared in UTC; ET fields use America/New_York with the offset written out.
The text cutoff is the population's `text_cutoff_utc` (release time; forecast origin five minutes later).

- **Minutes, `eligible_official_release_record_unmodified_file`.** The Fed press release named in
  the catalog says "For release at 2:00 p.m." and links this minutes document. The minutes PDF's
  internal ModDate precedes the release. Either the Fed server wrote the PDF within ten minutes after
  release (so it has not been replaced since), or a Wayback capture within 24 hours shows the same
  bytes. Every paragraph of today's HTML minutes matches the PDF wording. Fingerprint = SHA-256 of
  the PDF bytes.
- **Speeches, `eligible_document_release_line_unmodified_file`.** The speech PDF itself opens with
  "For release on delivery H:MM a.m./p.m. ET, date", no later than the cutoff. The other conditions
  are the same as for minutes. **This class needs Adaeze's acceptance.** It rests on the document's
  own release instruction plus the server write time, not only on the scheduled delivery time.
  Rows in this class can be excluded by filtering on `eligibility`.
- **Timing at the seconds level.** Files were typically written 10–20 seconds after the release
  minute, e.g. 14:00:17 for a 2:00 p.m. release. Eligibility treats the release minute as the
  public time, as the statement audit did. Exact offsets are in `pdf_written_minutes_after_release`.
- **`version_conflict_review`.** The PDF was modified after its release time (28 rows), or the
  released PDF wording differs from today's HTML (7 rows). Confirmed wording changes:
  - `min_20180801`: the released PDF reads "relative to the objectives"; today's HTML reads
    "relative to the Committee's objectives".
  - `min_20180926`: today's HTML notes that footnote 4 originally included text from footnote 3.
  - `min_20181108`: today's HTML adds information "inadvertently omitted in the original
    publication".
  The other four are speech rows whose examples are in `note`; they may include equation or
  reference-formatting artefacts and need a manual look.
- **`insufficient_evidence`.** Mostly documents from before about 2017. Their PDFs were rewritten
  during later site migrations (server dates in 2014, 2015 and March 2017), no archive copy within
  24 hours shows the original bytes, and no capture exists before the cutoff. HTML-only speeches
  (no PDF, mainly 2008–2010) have only a scheduled delivery time.
- **Formatting normalization** for the wording check: letters only, compared per paragraph. Page
  headers, page numbers and line-break hyphenation in the PDF are removed. Footnote markers
  (`<sup>`), "Return to text", "(PDF)" labels, video-player help, "Accessible Version" links and
  LaTeX equation lines in the HTML are skipped. Numbers are checked separately, and every HTML number
  must occur in the PDF. A year followed by a footnote marker, such as "2013.7", also counts as the year.

## Reproduce

From the endpoint checkout (`campaign-local/fed-spx-compact-endpoints`), with `requests`,
`beautifulsoup4`, `html5lib` and `pypdf` installed. Raw pages, PDFs and archive responses go to
`downloads/minutes-speech-evidence-v1/` (Git-ignored).

```sh
d=downloads/minutes-speech-evidence-v1
# audit in slices; --start/--count index the 472 assigned rows in population order
python scripts/verify_minutes_speech_versions.py \
  --catalog analysis/macro_releases/fed_announcements_v2.csv \
  --population downloads/fed-spx-endpoints-price-v1/population.csv \
  --start 0 --count 16 --raw-dir $d/raw --out $d/slices/s000.json
# ... repeated for later slices; rows 366-471 not yet run
python scripts/build_minutes_speech_ledger.py \
  --population downloads/fed-spx-endpoints-price-v1/population.csv \
  --inputs $d/slices/s*.json $d/wording-rerun.json \
  --out-dir results/fed-spx-minutes-speeches-evidence-v1
```

Script versions: slices s000–s342 ran with `verify_minutes_speech_versions.py` SHA-256 `fdd70f8a…7666`;
slice s354 and the re-run of all 14 wording-conflict rows (`wording-rerun.json`) used the final
`ab6fbdf2e16586a30c6b0e43adaae66f9388cea50d838e85e9da28a4f7ed0fc0`. The final change only relaxed
the wording check (more web-only lines skipped, footnote-marker years accepted). It cannot change
any row whose disposition did not depend on wording, and every wording-dependent conflict was re-run.
Ledger builder: `build_minutes_speech_ledger.py` `cbf6ebe78915ef0f296b27e1892f1c4b01bf34cb79fc286f4d304232f2b61bf7`.
Wayback lookups were made only for rows that were not already eligible (`wayback_checked`).

## Validation continuation

Codex ran the same final verifier on offsets 366 through 471, first with
`--no-wayback` and then with archive checks for unresolved rows. The raw
continuation outputs are named `codex-validation-publisher.json` and
`codex-s*.json` under the same local downloads directory. To reproduce the
merged minutes and speech ledger, pass the original slices, then
`wording-rerun.json`, then the publisher pass and the archive slices to
the builder. Later inputs take precedence. The [combined Fed/SPX cohort](../fed-spx-news-cohort-v1/README.md)
reports sensitivity to the original speech and minutes evidence rules.
