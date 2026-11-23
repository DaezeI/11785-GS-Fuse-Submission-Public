"""Count FOMC statement rows lost at each filter, per horizon, then text proof.

Starts from every catalog row whose event_id starts with ``stmt_`` (FOMC
statements plus the three FOMC "other policy" releases filed with them) and
applies, in order:

  1. release time verified and usable intraday (catalog)
  2. inside frozen cash-SPX coverage (alignment v1)
  3. chronological split (alignment v1); test is counted and then left reserved
  4. release inside the regular XNYS session (cash bars exist at the origin)
  5. 35 complete five-minute OHLC input bars ending by the release time
  6. origin bar and horizon target bars present and ending by the session close,
     counted separately for 20 and 30 minutes
  7. no price bar shared with another split (per horizon)
  8. exact-version text evidence by the text cutoff (ledger from
     verify_fomc_statement_versions.py)

It also counts rows that a shorter processing buffer or a shorter input window
would recover. Prices are read only for train and validation rows.

    python scripts/fomc_statement_funnel.py --price-csv ../fed-person3/downloads/SP500.csv \
        --ledger downloads/fomc-statement-evidence-pilot/official-v2-*.json \
                 downloads/fomc-statement-evidence-pilot/unresolved-wayback.json \
        --out downloads/fomc-statement-evidence-pilot/funnel.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import exchange_calendars as calendars
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import INPUT_SHA256  # noqa: E402
from scripts.build_fed_price_alignment import load_prices, sha256_file  # noqa: E402

CATALOG = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
ALIGNMENT = CATALOG.parent / "price_alignment_v1/fed_price_alignment_v1.csv"
BAR = pd.Timedelta(minutes=5)
ET = "America/New_York"
HORIZONS = (20, 30)
N_INPUT = 35


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-csv", required=True)
    ap.add_argument("--ledger", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer-min", type=int, default=5)
    a = ap.parse_args(argv)
    for name, path in {"prices": a.price_csv, "catalog": CATALOG, "alignment": ALIGNMENT}.items():
        if sha256_file(path) != INPUT_SHA256[name]:
            raise SystemExit(f"Frozen {name} input hash mismatch")

    evidence = {}
    for p in a.ledger:
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            evidence[r["event_id"]] = r  # later files (targeted re-checks) override
    cat = pd.read_csv(CATALOG)
    cat = cat[cat.event_id.str.startswith("stmt_")]
    al = pd.read_csv(ALIGNMENT).set_index("event_id")

    funnel, lost = [], defaultdict(list)

    def step(name, rows, dropped):
        by = lambda ids: dict(Counter(al.split.get(i, "none") if i in al.index else "none" for i in ids))
        funnel.append({"step": name, "remaining": len(rows), "lost": len(dropped),
                       "remaining_by_split": by(rows) if name != "catalog statements" else None,
                       "lost_ids": sorted(dropped)})
        lost[name] = sorted(dropped)

    ids = list(cat.event_id)
    step("catalog statements", ids, [])
    ok = set(cat.loc[cat.usable_intraday.astype(str) == "True", "event_id"])
    step("1 release time verified", [i for i in ids if i in ok], [i for i in ids if i not in ok]); ids = [i for i in ids if i in ok]
    cov = [i for i in ids if i in al.index and pd.isna(al.at[i, "exclude_reason"])]
    step("2 inside price coverage", cov, [i for i in ids if i not in cov]); ids = cov
    purged = [i for i in ids if al.at[i, "split"] == "purged"]
    step("3a not purged at split boundary", [i for i in ids if i not in purged], purged); ids = [i for i in ids if i not in purged]
    test = [i for i in ids if al.at[i, "split"] == "test"]
    step("3b test reserved (not evaluated)", [i for i in ids if i not in test], test); ids = [i for i in ids if i not in test]
    reg = [i for i in ids if al.at[i, "session_at_market"] == "regular"]
    step("4 release in regular session", reg, [i for i in ids if i not in reg]); ids = reg

    bars, _ = load_prices(a.price_csv)
    times = pd.DatetimeIndex(bars.date)
    tset = set(times)
    days = pd.to_datetime([al.at[i, "market_et"] for i in ids]).normalize()
    sched = calendars.get_calendar("XNYS", start=days.min(), end=days.max()).schedule
    close = {d.date(): r.close.tz_convert(ET).tz_localize(None) for d, r in sched.iterrows()}

    inp = {}
    bad_inp = []
    for i in ids:
        market = pd.Timestamp(al.at[i, "market_et"])
        before = times[times + BAR <= market]
        win = before[-N_INPUT:]
        complete = len(win) == N_INPUT and pd.Timestamp(al.at[i, "input_last_bar"]) == win[-1]
        inp[i] = win
        if not complete:
            bad_inp.append(i)
    step(f"5 {N_INPUT} input bars by release", [i for i in ids if i not in bad_inp], bad_inp)
    ids = [i for i in ids if i not in bad_inp]

    def target_ok(i, h, buffer_min):
        market = pd.Timestamp(al.at[i, "market_et"])
        origin = (market + pd.Timedelta(minutes=buffer_min)).ceil("5min")
        need = pd.date_range(origin - BAR, origin + pd.Timedelta(minutes=h) - BAR, freq=BAR)
        cl = close.get(market.date())
        if cl is None or origin + pd.Timedelta(minutes=h) > cl:
            return False, "target_after_session_close", need
        if not all(t in tset for t in need):
            return False, "missing_origin_or_target_bar", need
        return True, "", need

    per_h = {}
    for h in HORIZONS:
        keep, why = [], Counter()
        for i in ids:
            good, reason, _ = target_ok(i, h, a.buffer_min)
            (keep.append(i) if good else why.update([reason]))
        # cross-split shared bars among retained rows at this horizon
        used = defaultdict(set)
        for i in keep:
            _, _, need = target_ok(i, h, a.buffer_min)
            used[al.at[i, "split"]].update(set(inp[i]) | set(need))
        shared = len(used["train"] & used["val"])
        proof = Counter(evidence.get(i, {}).get("eligibility", "not_checked") for i in keep)
        verified = [i for i in keep if str(evidence.get(i, {}).get("eligibility", "")).startswith("eligible")]
        by_split = lambda rows: {s: {"rows": sum(al.at[i, "split"] == s for i in rows),
                                     "release_instants": len({al.at[i, "market_et"] for i in rows if al.at[i, "split"] == s}),
                                     "dates": len({al.at[i, "market_et"][:10] for i in rows if al.at[i, "split"] == s})}
                                 for s in ("train", "val")}
        per_h[f"{h}m"] = {
            "6_target_window_complete": {"remaining": len(keep), "lost": dict(why), "by_split": by_split(keep)},
            "7_cross_split_shared_bars": shared,
            "8_text_evidence_status": dict(proof),
            "usable_text_and_price": {"total": len(verified), "by_split": by_split(verified)},
            "lacking_text_proof": sorted(set(keep) - set(verified)),
        }

    # buffer and input-length sensitivity on rows lost after step 4
    recover_buffer = {b: sum(1 for i in ids for h in HORIZONS if target_ok(i, h, b)[0] and not target_ok(i, h, a.buffer_min)[0])
                      for b in range(0, a.buffer_min)}
    shorter_input = {n: sum(1 for i in bad_inp if len(times[times + BAR <= pd.Timestamp(al.at[i, "market_et"])]) >= n)
                     for n in (12, 24)}
    lost_before_session = {s["step"]: s["lost_ids"] for s in funnel if s["lost"]}
    out = {"funnel": funnel, "per_horizon": per_h,
           "buffer_sensitivity_rows_recovered_by_shorter_buffer": recover_buffer,
           "input_length_sensitivity_rows_recovered": shorter_input,
           "lost_ids_by_step": lost_before_session,
           "conventions": {"bars": "five-minute cash SPX OHLC, bar-start timestamps, America/New_York, 09:30-16:00",
                           "session": "XNYS calendar incl. early closes", "buffer_min": a.buffer_min,
                           "text_cutoff": "origin minus buffer; origin = release + buffer rounded up to the bar grid"},
           "input_sha256": INPUT_SHA256}
    Path(a.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for s in funnel:
        print(f"{s['step']:<38} remaining {s['remaining']:>4}  lost {s['lost']:>3}  {s['remaining_by_split'] or ''}")
    for h, v in per_h.items():
        print(h, json.dumps({k: v[k] for k in ("6_target_window_complete", "7_cross_split_shared_bars",
                                               "8_text_evidence_status", "usable_text_and_price")}))
    print("buffer recovery:", recover_buffer, "| shorter input recovery:", shorter_input)


if __name__ == "__main__":
    main()
