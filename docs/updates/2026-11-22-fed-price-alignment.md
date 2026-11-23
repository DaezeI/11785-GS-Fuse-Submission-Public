# Fed announcements to S&P 500 price windows (Person 2) - November 22, 2026

**Status: code and offline tests ready; time rules agreed with Person 1 (Jessica) on November 22.
No model was trained. Splits are not assigned here.**

Owner: Person 2. Input from Person 1: `analysis/macro_releases/fed_announcements_v2.csv`
(PR #41, merged; commit `12e3e51`, file sha256 `56157fe0…c9d939`). Price file: the team's frozen
`SP500.csv` (sha256 `6a873b1d…8de2a`, checked by the script).

## Time rules (agreed with Person 1)

1. The price file's `date` is the **start** of each 5-minute bar, New York wall time. The source does
   not document this; three checks on the data support it:
   - **Official close.** For 1,846 days (2016-11 to 2024-04) the Close of the bar labelled 16:00
     equals the S&P 500 daily close from FRED in 92.5% of days (median difference 0.00). The 15:55
     bar differs by a median 0.24 points and the 15:50 bar by 1.94. Consecutive 5-minute closes
     earlier in the afternoon differ by 1.8 to 2.0 points, but 15:55 to 16:00 differs by only
     0.24, so the 16:00 bar is not a full trading interval: it holds the closing print after 16:00.
     If labels marked bar ends, 15:55 would be 5 minutes before the close and differ by a full
     5-minute move.
   - **FOMC statements** public at 14:00 (88 events): mean high-low range is about 5.5 bp in the
     bars labelled 13:35 to 13:50, 8.8 bp at 13:55, **44.7 bp at 14:00**, 23.6 bp at 14:05. A
     14:00 release can only move the bar that covers 14:00 to 14:05. The 14:15 statements (33
     events) peak at labels 14:15 and 14:20 (44 and 45 bp).
   - The 09:30 bar has the largest median range of the day (23.8 bp versus 12.9 bp at 09:35); the
     16:00 bar has a median range of 0.7 bp, versus about 10 bp at 15:55.
2. `available_utc` = market time + 5 min processing lag, converted to New York time.
3. **Target window** starts at the first bar whose start is >= `available_et`, no extra buffer.
4. **Input window** ends at the last bar that had ended by the **market time** (`market_time_et`, the
   announcement time): start <= market time - 5 min. For a 14:00 statement the last input bar is
   13:55-14:00, and the 14:00-14:05 bar (the first reaction) is in neither input nor target.
   The 5-minute lag only sets where the target starts. `--input-cutoff available` keeps the
   older rule (cut at the available time) for comparison.
5. Only bars starting 09:30 to 16:00 are real prices. Later bars are filler (about 61% flat at
   16:05, 97% at 16:10, 100% at 16:15) and are dropped.
6. Regular-session events (classified at the market time) form the main dataset (`main_candidate`:
   regular session, target starts within 5 minutes of the available time, no missing bars in either
   window). Pre-market, after-hours, weekend and closed-day events stay as a flagged subset.
   Press-conference transcripts count as public after the meeting day, so all 75 are in the subset;
   the statement row covers the 2:00 p.m. reaction.

Every output row is asserted to satisfy
`input_last_bar_end <= market_et <= available_et <= target_first_bar_start`. All 88 14:00
statements have input ending 13:55 and target starting 14:05. The market time text
(`market_time_et`) agrees with `available_utc - 5 min` on every usable row.

## What the price file needed

| Finding | Handling |
|---|---|
| 2011-11-07 and 2012-02-02 appear twice (162 rows removed, identical prices); file not sorted | drop duplicates, sort; conflicting duplicates would raise |
| 8,311 bars after 16:00, 80% flat | dropped |
| 2016-10-27 opens at 10:00 | windows touching it flagged `has_gap` |
| 28 within-day gaps over 5 minutes; 35 short or incomplete days | gaps flagged; early closes kept |
| 152 weekdays without bars | holidays or data gaps; no exchange calendar applied, so reported not classified |
| Last bar 2024-04-17 | 383 usable events after it have no target |

## Alignment result (Jessica's CSV, 2,551 rows; input 35 bars, target 140 bars, market cutoff)

| Outcome | Rows |
|---|---:|
| aligned | 2,045 |
| after price coverage | 383 |
| not usable intraday in the events file | 121 |
| before price coverage / too few target bars | 1 / 1 |

Aligned rows by session at the market time: regular 1,483, pre-market 309, after-hours 203, weekend
43, closed weekday or data gap 7. **`main_candidate`: 1,448 rows** (545 Board releases, 503 speeches,
145 testimonies, 130 statements, 124 minutes, 1 other policy release), 23 to 149 per year, 1,365
distinct announcement instants. Reaction lag (available time to first target bar): 1,465 within 5
minutes, 88 within 30, 106 within 2 hours, 216 up to 16 hours, 170 up to 72 hours.
Embargo bars: none for 562 rows, one for 1,477, two for 6. All 2,045 target windows cross a trading
day (140 bars exceed one 79-bar session); 556 input windows do. 176 aligned rows share an instant
with another event (`concurrent_group_id`); 116 are short cover notes (`short_text`).

These are data-preparation counts, not model results.

## Still open (Persons 3 and 4)

- Keep 35 input / 140 target bars although 140 bars is almost two sessions? Allow windows across
  the overnight break? The audit file reports how many do.
- The price file ends 2024-04-17; extend it to 2024-06 to recover part of the 383?
- Splits are not assigned here. Concurrent events share a price window and are not independent;
  a chronological split must purge windows that overlap across splits (the C1 audit already does).
- Press conferences are all in the flagged subset; the FOMC statement row stands in for them.

## Reproduce

```bash
python scripts/build_fed_price_alignment.py \
    --events analysis/macro_releases/fed_announcements_v2.csv \
    --prices /path/to/SP500.csv --out-dir downloads/fed_price_alignment_v1
python -m pytest tests/test_fed_price_alignment.py -q
```

Outputs (`fed_price_alignment_v1.csv`, `price_alignment_audit_v1.json`) hold timestamps and flags
only: no prices and no article text. They are not committed in this change. The official-close
check used FRED series `SP500`, downloaded for the comparison and not stored in the repository.
