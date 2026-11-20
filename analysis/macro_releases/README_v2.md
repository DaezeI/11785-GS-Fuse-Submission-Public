# Federal Reserve announcements with verified public times (Person 1, v2)

Owner: Jessica (Person 1). Built 2026-11-20.

v2 completes the Person 1 task, "official Federal Reserve announcements", which is wider than the FOMC-only v1.
**v1 files are not changed.** The v2 CSV includes the v1 rows exactly as they are (`source_version = v1`), so teammates need only `fed_announcements_v2.csv`.

## Files

| File | What it is |
|---|---|
| `fed_announcements_v2.csv` | Deliverable: 2,551 rows (311 from v1 + 2,240 new) |
| `fed_announcements_evidence_v2.json` | Reviewed evidence for the 2,240 new rows: source URL, time evidence, text SHA-256, plus 6 excluded feed entries with the reason |
| `../../scripts/build_fed_announcements_v2.py` | `collect` / `evidence` / `build` / `verify` / `fetch-text` / `check-before` |
| `../../tests/test_fed_announcements_v2.py` | Offline tests |

No source text is committed. To get the texts, run `fetch-text` (see below).

## What is in it (2008-01 to 2026-11, all from www.federalreserve.gov)

| event_type | rows | usable intraday | notes |
|---|---|---|---|
| fomc_statement / fomc_minutes / fomc_other_policy_release | 311 | 310 | v1, unchanged |
| fed_board_monetary_release | 695 | 671 | discount rate, lending facilities, swap lines, economic projections, auctions, schedules (`subcategory`) |
| fed_speech | 1,210 | 1,138 | Chair, Vice Chairs and Governors (`speaker`, `speaker_role`) |
| fed_testimony | 240 | 216 | Congressional testimony, including the semiannual Monetary Policy Report |
| fomc_press_conference | 95 | 95 | Chair's post-meeting press conference, transcript PDF |

**Usable rows inside the CAMEF price window (2008-01 to 2024-06): 2,085.**
By type: 895 speeches, 636 Board releases, 206 testimonies, 137 statements, 131 minutes, 77 press conferences, 3 other.

`usable_intraday = True` means the row has a verified market time, has non-empty text, and is not a duplicate of an earlier text.

## Which time to use: when the market learned the content

The goal is to line up each announcement with the moment prices could react, so every row is dated by **when its content first became public**, not by when a text file was posted online.

- `event_time_et`: when the event happened (release time, speech start, hearing start).
- `market_time_et`: when the market first had the content.
- `available_utc` = `market_time_et` + 5 minutes. Use it both to align price bars and for Nicole's check 2, "was it available before the prediction?":

```bash
python scripts/build_fed_announcements_v2.py check-before --prediction-et "2020-03-16 09:30"
```

**Why this is not leakage.** Leakage means using information that was not yet public at prediction time. A speech is public the moment it is delivered: it is webcast, reported live, and the prepared text is released at delivery. Using its text in a prediction made after delivery uses only public information, even if the web page was updated later.

**Alternative considered and rejected.** One draft dated speeches and testimony by when the full text could be shown to be online. That pushed 1,360 speeches and testimonies to the next day or later, long after prices had reacted, so the model would have been trained on the wrong windows.

## How times were verified (`evidence_level`)

- **A_release_line** (600 Board releases, plus the v1 FOMC rows): the page prints "For release at 10:00 a.m. EDT".
- **F_official_feed_time** (speeches, testimony, 72 Board releases): the timestamp in the Fed's own news feeds (`/json/ne-press.json`, `ne-speeches.json`, `ne-testimony.json`).
  - Feed accuracy check: Board pages that print a time agree with the feed in 598 of 600 cases. The 2 that differ print "12:00 a.m." while the feed has a date only.
  - Calendar check (`/json/calendar.json`, covers 2017-2022 and 2025-2026): speeches agree in 475 of 481 matches. **The 6 that disagree are marked not usable** (`calendar_check = differs` or `ambiguous_multiple_events`).
  - Testimony agrees in 37 of 50 matches. Most differences are the semiannual testimony, where the prepared text is released at 8:30 a.m. and the hearing starts at 10:00 a.m. **The earlier time is used**, because that is when the market learned the content.
  - Spot check: for 2017 onward, the speech PDF on the Fed site is usually last modified on the delivery day, often within seconds of the delivery time (e.g. Governor Cook, 2026-11-20: speech 12:15 p.m., PDF 12:15:12 p.m.).
- **P_after_live_event_day** (95 press conferences): the press conference is live on the meeting-day afternoon, but the transcript covers the whole event and its end time is not verified. The transcript therefore counts as known only after the meeting day ends (midnight ET). Use the FOMC statement row for the 2:00 p.m. reaction itself.
- **C_date_only** (66 speeches and 24 testimonies from 2008, 23 Board releases): no time is known. These count as available only from midnight ET after the release day, and `usable_intraday = False`.

The transcript PDF's posting time (HTTP Last-Modified) is recorded in `time_evidence` for reference only. It is not used as a timestamp: a file can be uploaded before it is made public, so it is not a reliable bound.

## Other columns

- `market_session_et` (pre_market / regular / after_hours / weekend) is computed at `market_time_et`. Of the usable rows: 1,694 regular hours, 405 pre-market, 276 after hours, 55 weekend. Exchange holidays are left to Person 2.
- `duplicate_of`: the same text published twice. Only the earliest copy stays usable.
- `concurrent_with`: other events at the same minute (212 rows, e.g. economic projections released with the statement). Persons 3/4 should not count these as independent events.
- Short texts: 116 Board releases are cover notes such as "The minutes are attached" (under 200 characters). They are correct but carry little information; filter by text length or `subcategory` if needed.

## Excluded

- 5 feed entries with no page of their own (for example "identical remarks" delivered elsewhere), and 1 press release already in v1 (the June 2008 minutes release). All 6 are listed in the evidence JSON under `excluded`.
- Other Board press releases (enforcement actions, bank applications, banking regulation): not monetary-policy announcements.
- Reserve Bank presidents' speeches: not in the Board's speech feed (they are on each Reserve Bank's own site).

## Getting the text: `fetch-text`

**Why it is needed.** The team rule is to never commit source data, so the CSV holds only each announcement's link and text fingerprint (`text_sha256`), not the text itself. The model needs the text, so every teammate has to download it on their own machine. Before this command, that meant running two scripts (v1 and v2) whose saved files were named differently and did not match `event_id`, and the press-conference PDFs were not converted to text.

**What it does.** One command downloads the text of every row in `fed_announcements_v2.csv`, v1 rows included:

```bash
pip install beautifulsoup4 lxml requests pypdf
python scripts/build_fed_announcements_v2.py fetch-text --usable-only
```

1. Downloads each row's `source_url` from federalreserve.gov.
2. Extracts the text with the same container recorded in `text_selector`, converts PDFs to text, and saves it as `downloads/fed_announcements_v2/text/<event_id>.txt`. The folder is git-ignored and is never committed.
3. Recomputes the fingerprint and compares it with `text_sha256`. A match means the downloaded text is exactly the text we verified. Mismatches and errors are listed in `downloads/fed_announcements_v2/fetch_text_report.json`.

Options: drop `--usable-only` to fetch every row, and add `--skip-existing` to resume an interrupted run.

**Using it.** Join on `event_id`:

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("analysis/macro_releases/fed_announcements_v2.csv")
df = df[df.usable_intraday]
text_dir = Path("downloads/fed_announcements_v2/text")
df["text"] = [(text_dir / f"{e}.txt").read_text() for e in df.event_id]
```

**PDF rows** (95 press conferences and the June 2008 minutes): for these, the check confirms that the PDF file is the same one we verified. The `.txt` is converted from that exact file, but the converted wording itself was not fingerprinted beforehand.

## Re-check

```bash
python scripts/build_fed_announcements_v2.py verify      # re-downloads ~2,300 pages, takes a while
python -m pytest tests/test_fed_announcements_v2.py -q
```

`build` and `evidence` refuse to overwrite existing files.
