# FOMC releases with verified public times (Person 1, v1)

Owner: Jessica (Person 1, news detective). Built 2026-11-20.
This is a **new, separate** dataset. It does not edit or replace the old
CAMEF / GS-Fuse text folders (`data/event/...`) or any existing split file.

## Files

| File | What it is |
|---|---|
| `fomc_releases_v1.csv` | The deliverable: one row per official release, with release time, UTC time, safe availability time, source link and text fingerprint. |
| `fomc_release_evidence_v1.json` | The reviewed evidence the CSV is built from (source URL, exact time sentence quoted from an official page, text SHA-256). |
| `old_c1_vs_official_fomc_v1.csv` | Read-only comparison of the old event folders against the official dates. |
| `../../scripts/build_fomc_releases.py` | Builds the CSV, re-verifies everything online, and runs the "available before prediction?" check. |
| `../../tests/test_fomc_releases.py` | Offline tests (no network). |

No source text is committed (team rule). Run `verify --save-text` to download
the clean texts into `downloads/fomc_releases_v1/`, which is git-ignored.

## What is in it

311 releases from 2008-01-22 to 2026-11-18, all from www.federalreserve.gov:

- 157 FOMC statements: 150 at scheduled meetings, 7 unscheduled (conference calls, notation votes, the March 2020 emergency actions)
- 151 sets of FOMC minutes, dated by their **release** date (about three weeks after the meeting), never the meeting date
- 3 other FOMC policy releases (2019-10-11 implementation statement, 2020-03-19 swap lines, 2020-03-31 FIMA repo facility)

Inside the CAMEF price window (2008-01 to 2024-06) there are 272 releases (138 statements, 131 minutes, 3 other).

## How each time was verified (`evidence_level`)

- **A_release_line_on_source_page / A_release_line_on_press_release** (245 rows): the official page itself says "For release at 2:00 p.m. EDT" (or another time). For minutes, this line is on the press release that announced them.
- **B_fomc_minutes_vote_record** (65 statements, 2008-2015): these pages only say "For immediate release". The time comes from the FOMC's own minutes, e.g. "The vote encompassed approval of the statement below to be released at 2:15 p.m." The quote and its URL are in `time_evidence` and `time_evidence_url`.
- **C_date_only_time_unverified** (1 row, `stmt_20080311a`): the date is confirmed, but no official page states the time. `usable_intraday = False`, and it counts as available only from midnight ET after the release day.

Times found: 2:00 p.m. (most), 2:15 p.m. (2008 to Jan 2013), 12:30 p.m. (the eight 2011-2012 press-conference meetings), plus special times such as 7:00 a.m. (2008-10-08 coordinated cut), 9:15 p.m. Sunday (2010-05-09), 5:00 p.m. Sunday (2020-03-15) and 9:00 a.m. (minutes released 2013-04-10).

## Timing columns (answers Nicole's two Person 1 checks)

1. **Is the timestamp correct?** `release_time_et` is quoted from an official page; `tz_abbrev` (EST/EDT) is computed from the date with `America/New_York`, and the build fails if it disagrees with the EST/EDT printed on the page. `release_utc` is the same moment in UTC.
2. **Was it available before the prediction?** `available_utc = release_utc + processing_lag_min` (default 5 minutes, conservative). A release may be used for a prediction only if `available_utc < prediction time`. Use the script:

```bash
python scripts/build_fomc_releases.py check-before --prediction-et "2020-03-16 09:30"
python scripts/build_fomc_releases.py check-before --predictions-csv my_forecasts.csv   # needs a prediction_utc column
```

`market_session_et` is `pre_market`, `regular` (9:30-16:00 ET), `after_hours`, `weekend` or `unknown`. Exchange holidays and early closes are **not** handled here: Person 2 should map each `available_utc` to the first SPX 5-minute bar that starts at or after it.

## Text fingerprint

`text_sha256` is the SHA-256 of the cleaned text: the `<p>` paragraphs under `#article > div.col-sm-8:not(.heading)` (statements) or under `#article` / `#leftText` (minutes, 2008-2011 pages use `#leftText`), whitespace collapsed, empty paragraphs and "For media inquiries" lines dropped, joined with a newline. June 2008 minutes exist only as a PDF, so that row hashes the PDF bytes. All 311 hashes are unique (no duplicates), and the statement hashes gave the same result when the pages were fetched a second time.

## Re-check everything yourself

```bash
pip install beautifulsoup4 lxml requests          # html5lib optional (browser-identical parsing)
python scripts/build_fomc_releases.py verify --save-text
python -m pytest tests/test_fomc_releases.py -q
```

`verify` downloads each page again, recomputes every hash and checks that each time quote still appears on its evidence page. It writes `downloads/fomc_releases_v1/verify_report.json`.

`build` refuses to overwrite an existing CSV. A future version must be written to a new file (for example `--out analysis/macro_releases/fomc_releases_v2.csv`), so every version stays traceable.

## What we found about the old data (`old_c1_vs_official_fomc_v1.csv`)

- Old folder **4** is FOMC minutes: all 122 of its dates match official minutes **release** dates. It has dates only (no times), and 6 releases inside its date range are missing: 2011-04-05, 2014-07-09, 2020-04-08, 2021-01-06, 2023-07-05 and 2023-11-21.
- None of the other old folders lines up with official FOMC dates (at most about 10% of dates match by chance), so FOMC statements are not a separate folder in the old data.

## Checklist mapping (`analysis/text_value/REPLACEMENT_NEWS_AUDIT.md`)

| Check | Status for Person 1 |
|---|---|
| 1. Relevant text with source ID, version/hash, no duplicates | Done: `event_id`, `source_url`, `text_sha256`; duplicate test passes. |
| 2. Text available before forecast; verified timestamp with timezone; conservative availability plus lag | Done: `release_utc`, `tz_abbrev`, `available_utc` (+5 min), `check-before`. |
| 3. Matched SPX 5-minute bars | Person 2 (Joye): join on `available_utc`. |
| 4. Enough independent events; chronological splits | Persons 3/4. About 150 scheduled statements is small: consider statements + minutes, and note that 2020 has many unscheduled releases. |
| 5. Permission | The Fed site says its information is in the public domain and may be copied and distributed without permission; cite the Board as the source ([disclaimer](https://www.federalreserve.gov/disclaimer.htm)). |

## Known limits

- Only FOMC releases. Other macro news (CPI, jobs report, speeches) is not included.
- `stmt_20080311a` has no verified time; leave it out of intraday labels.
- The 5-minute lag is a choice; change it with `build --lag-min N --out <new file>`.
- Statements released "for immediate release" in 2008-2015 rely on the time the Committee voted to release them (from the minutes). This is the Fed's own record, but it is not a press-release timestamp.
