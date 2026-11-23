# Fed announcements to S&P 500 price windows (Person 2) - draft, November 22, 2026

**Status: code and offline tests ready; conventions below are PENDING confirmation by Person 1
(Jessica). The numbers are a preliminary run against her unmerged PR #41. No model was trained.**

Owner: Person 2. Depends on Person 1's `analysis/macro_releases/fed_announcements_v2.csv`
(PR #41, branch `jessica/fomc-releases-v1`, commit `12e3e51`, file sha256 `56157fe0…c9d939`).
Price file: the team's frozen `SP500.csv` (sha256 `6a873b1d…8de2a`, checked by the script).

## Time contract (needs Jessica's confirmation)

1. The price file's `date` is the **start** of each 5-minute bar, New York wall time. The source does
   not document this; it is supported by three checks on the data (November 22):
   - FOMC statements public at 14:00 (88 events, available time 14:05): the mean high-low range is
     5.5 bp in the bars labelled 13:35 to 13:50, **44.7 bp in the bar labelled 14:00**, then 23.6 bp at
     14:05 and falling. A statement released at 14:00 can only move the bar that covers 14:00 to 14:05.
     The 14:15 statements (33 events) peak at labels 14:15 and 14:20 (44 and 45 bp).
   - The bar labelled 09:30 has the largest median range of the day (23.8 bp versus 12.9 bp at 09:35),
     and on 2008-01-02 its Open is 1467.97 (recalled, not checked against a source: the official
     open).
   - The bar labelled 16:00 has a median range of 0.7 bp, versus about 10 bp at 15:55: it covers the
     time after the close.
2. `available_utc` (market time + 5 min) is converted to New York time. No further buffer is added.
3. **Target window** starts at the first bar whose start is >= the available time.
4. **Input window** ends at the last bar that had ended by the available time (start <= available - 5 min).
   A bar that straddles the available time is an embargo bar, used as neither input nor target.
5. Only bars starting 09:30 to 16:00 are real prices. Later bars are filler (about 61% flat at 16:05,
   97% at 16:10, 100% at 16:15) and are dropped.

The script asserts `input_last_bar_end <= available_et <= target_first_bar_start` for every output row.

## What the price file needed

| Finding | Handling |
|---|---|
| 2011-11-07 and 2012-02-02 appear twice (162 rows removed, identical prices); file not sorted | drop duplicates, sort; conflicting duplicates would raise |
| 8,311 bars after 16:00, 80% flat | dropped |
| 2016-10-27 opens at 10:00 | windows touching it flagged `has_gap` |
| 28 within-day gaps over 5 minutes; 35 short or incomplete days | gaps flagged; early closes kept |
| 152 weekdays without bars | holidays or data gaps; no exchange calendar applied, so they are reported not classified |
| Last bar 2024-04-17 | 383 usable events after it have no target |

## Preliminary alignment (Jessica's CSV, 2,551 rows; input 35 bars, target 140 bars)

| Outcome | Rows |
|---|---:|
| aligned | 2,045 |
| after price coverage | 383 |
| not usable intraday in the events file | 121 |
| before price coverage / too few target bars | 1 / 1 |

Aligned rows by session at the available time: regular 1,461, pre-market 307, after-hours 227,
weekend 43, closed weekday or data gap 7. Reaction lag (available time to first target bar):
1,465 within 5 minutes, 88 within 30, 106 within 2 hours, 216 up to 16 hours, 170 up to 72 hours.
`main_candidate` (regular session, lag <= 5 minutes, no gap in either window) is true for 1,450 rows.
All 2,045 target windows cross a trading day (140 bars exceed one 79-bar session); 589 input windows do.
1,952 distinct available instants, so 176 aligned rows share an instant with another event
(`concurrent_group_id`); 116 rows are short cover notes (`short_text`).

These counts were computed from a CSV that may still change before merge; rerun and replace them
after the merge. They are not a result about models.

## Design issue found by the event check

With `available = market time + 5 min`, the input window ends exactly at the available time. For an
announcement made on the 5-minute grid (most FOMC statements), the **last input bar is the bar in
which the news broke** (14:00 to 14:05) and already contains the largest part of the reaction
(44.7 bp mean range against about 5.5 bp normally). This is not leakage under "only data before
the prediction time" (the forecast is made at 14:05), but the text then has to explain what happens
after the first reaction. 2,039 of 2,045 aligned rows have no embargo bar. An alternative is to end
the input before the announcement (`market_time_et`), embargo the reaction bar, and start the target
at the available time. This choice is for the team; the script currently follows the
available-time rule.

## Questions that change the dataset

- Jessica: items 1 to 4 above.
- Persons 3 and 4: keep 35 input / 140 target bars even though 140 bars is almost two sessions? Allow
  windows across the overnight break? Use only `main_candidate` rows for the main benchmark and report
  the rest separately? The price file ends 2024-04-17; extend it to 2024-06?
- Splits are not assigned here. Concurrent events share a price window and are not independent; any
  chronological split must purge windows that overlap across splits (the C1 audit already does this).

## Reproduce

```bash
python scripts/build_fed_price_alignment.py \
    --events analysis/macro_releases/fed_announcements_v2.csv \
    --prices /path/to/SP500.csv --out-dir downloads/fed_price_alignment_v1
python -m pytest tests/test_fed_price_alignment.py -q
```

Outputs (`fed_price_alignment_v1.csv`, `price_alignment_audit_v1.json`) hold timestamps and flags
only: no prices and no article text. They are not committed in this change.
