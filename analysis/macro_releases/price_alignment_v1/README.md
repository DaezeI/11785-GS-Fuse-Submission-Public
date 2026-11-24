# Fed announcement price windows v1 - how to use

One row per Fed announcement in `fed_announcements_v2.csv`, aligned to the team's frozen 5-minute
S&P 500 prices. Built by `scripts/build_fed_price_alignment.py` (Person 2). The files here hold
timestamps and flags only: **no prices and no article text**. Details, evidence and the rules are in
[docs/updates/2026-11-22-fed-price-alignment.md](../../../docs/updates/2026-11-22-fed-price-alignment.md).

| File | What it is |
|---|---|
| `fed_price_alignment_v1.csv` | 2,551 rows; `exclude_reason` empty = usable example; `split` = train / val / test / purged |
| `price_alignment_audit_v1.json` | counts, price-file checks, leakage check, split audit, input hashes |

Inputs it was built from: `fed_announcements_v2.csv` sha256 `56157fe0…c9d939` and the frozen
`SP500.csv` sha256 `6a873b1d…8de2a` (the loader and script refuse a different price file).

## What one example is

- `x`: the 35 five-minute OHLC bars that had **ended before the announcement** (for a 14:00
  statement, the last one is 13:55-14:00).
- the 14:00-14:05 bar (first reaction) is in **neither** `x` nor `y`.
- `y`: the 140 bars starting at the first bar at or after `announcement + 5 min` (14:05 here).
- Price time is the bar start, New York time. Only 09:30-16:00 bars exist (later filler removed).
- 140 bars is almost two sessions, so `y` always crosses the overnight break.

## Use it

```python
from data.fed_price_dataset import FedPriceDataset

kw = dict(alignment_csv="analysis/macro_releases/price_alignment_v1/fed_price_alignment_v1.csv",
          price_csv="/path/to/SP500.csv",            # the frozen team file
          x_mode="return_bp", y_mode="return_bp")    # or "price"
train = FedPriceDataset(split="train", **kw)
scaler = FedPriceDataset.fit_scaler(train)           # fit on train only
val = FedPriceDataset(split="val", scaler=scaler, **kw)
test = FedPriceDataset(split="test", scaler=scaler, **kw)
ex = train[0]   # x (35,4) float32, y (140,4) float32, last_close, event_id, event_type, text ...
```

Channels are Open, High, Low, Close. `return_bp` = 10,000 * log(price / last input Close).
It needs only numpy and pandas; wrap it in a torch `Dataset` if you need one (`len` and `[]` work).

**Text.** Source text is not in Git. Get it with
`python scripts/build_fed_announcements_v2.py fetch-text --usable-only` (writes
`downloads/fed_announcements_v2/text/<event_id>.txt`) or from the team Drive zip
`fed_announcements_v2_text.zip`, then pass `text_dir=...` (add `require_text=True` to fail on a
missing file). The file name is the `event_id` column.

## Splits (chronological, purged)

| | Main rows | All aligned rows |
|---|---:|---:|
| train (2008-01 to 2019-05) | 1,013 | 1,409 |
| val (2019-05 to 2021-08) | 215 | 300 |
| test (2021-08 to 2024-04) | 218 | 334 |
| purged | 2 | 2 |

Cut dates are the 70% / 85% announcement-time quantiles of the main rows. Rows of a later split
whose windows would share a price bar with an earlier split are `purged`; the audit file verifies
that no bar is shared and that events at the same instant stay in one split. Train contains the
2008-09 crisis, so train inputs are more volatile than val/test (scaled val/test std is about 0.77
and 0.63). Test is the 2022 rate-hike period, so expect a regime shift.

## Rules for using it

- **Main benchmark = `subset="main"`** (1,448 rows): announcement in the regular session, target
  starts within 5 minutes of the available time, no missing bars in either window. Pre-market,
  after-hours, weekend, closed-day rows and all 75 press-conference transcripts are `flagged`;
  report them separately (`subset="flagged"`), do not mix them into the main number.
- **Never refit anything on val or test** (scaler, normalisation, thresholds).
- **Do not shuffle across splits and do not re-split.** The purge is what keeps windows apart.
- Events at the same instant (`n_concurrent` > 1, e.g. statement plus projections) share one price
  window. Count them once for effective sample size, or weight by 1 / `n_concurrent`.
- 116 rows are short cover notes (`short_text`); drop them if the text branch should see content.
- Use the first 35 / 70 / 140 steps of `y` for the three horizons used elsewhere in the project.
- Overlapping windows inside a split are normal (announcements a few hours apart); use date-block
  bootstrap intervals, as the team does elsewhere.
- Nothing here is a model result. Price data ends 2024-04-17; 383 usable later announcements have
  no price window. Exchange holidays versus data gaps are not separated.

## Rebuild

```bash
python scripts/build_fed_price_alignment.py \
    --events analysis/macro_releases/fed_announcements_v2.csv \
    --prices /path/to/SP500.csv --out-dir analysis/macro_releases/price_alignment_v1 --force
python -m pytest tests/test_fed_price_alignment.py tests/test_fed_price_dataset.py -q
```
