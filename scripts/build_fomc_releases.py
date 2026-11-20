"""Person 1 (news detective): verified FOMC release list with exact public times.

This is a NEW, separate pipeline. It never reads or changes the old CAMEF /
GS-Fuse text folders (data/event/...) except in the read-only `compare-old`
report, so the original data stays untouched and traceable.

Inputs (committed, small, reviewed):
    analysis/macro_releases/fomc_release_evidence_v1.json
        One record per official Federal Reserve release: source URL, the exact
        release-time sentence quoted from an official page, and a SHA-256
        fingerprint of the cleaned text. No source text is stored.

Commands:
    build          evidence JSON -> fomc_releases_v1.csv (UTC times, conservative
                   availability time, market session). Refuses to overwrite.
    verify         re-download every source page from federalreserve.gov, re-extract
                   the clean text, compare SHA-256 and re-check the time evidence.
                   Texts are saved under downloads/ (git-ignored, never committed).
    check-before   Nicole's check 2: is each release available before a given
                   prediction time? Works for one time or a CSV of prediction times.
    compare-old    read-only report: old event-folder dates vs official dates.

Examples:
    python scripts/build_fomc_releases.py build
    python scripts/build_fomc_releases.py verify --save-text
    python scripts/build_fomc_releases.py check-before --prediction-et "2020-03-16 09:30"
    python scripts/build_fomc_releases.py compare-old \
        --manifest splits/manifests/purged_split_c1_date_screen_v1.json
"""

import argparse
import csv
from datetime import datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "analysis" / "macro_releases" / "fomc_release_evidence_v1.json"
OUT_CSV = ROOT / "analysis" / "macro_releases" / "fomc_releases_v1.csv"
TEXT_DIR = ROOT / "downloads" / "fomc_releases_v1"  # git-ignored (downloads/)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DEFAULT_LAG_MIN = 5
OPEN, CLOSE = time(9, 30), time(16, 0)

COLUMNS = [
    "event_id", "event_type", "scheduled", "meeting_date", "meeting_label",
    "release_date", "release_time_et", "tz_abbrev", "release_datetime_et",
    "release_utc", "processing_lag_min", "available_utc", "available_et",
    "market_session_et", "time_verified", "usable_intraday", "evidence_level",
    "time_evidence", "time_evidence_url", "source_url", "title", "text_format",
    "text_selector", "text_paragraphs", "text_chars", "text_sha256", "notes",
]

NOTES = {
    "stmt_20080122b": "Unscheduled intermeeting cut (Jan 21 call); 8:30 a.m. from Jan 29-30, 2008 minutes.",
    "stmt_20080311a": "Release date verified; exact time not stated on any official page found. Date-level only: do not use for intraday labels.",
    "stmt_20080625a": "HTML minutes not available; time from official PDF minutes.",
    "min_20080625": "Minutes published only as PDF; fingerprint is SHA-256 of the PDF bytes.",
    "min_20130320": "Unusual 9:00 a.m. release (press release line).",
    "stmt_20081008a": "Coordinated central-bank rate cut (7:00 a.m.).",
    "stmt_20100509a": "Sunday-evening swap-line statement (9:15 p.m.).",
    "stmt_20200315a": "Sunday emergency statement (5:00 p.m.).",
}


# ----------------------------------------------------------------- timing ---
def parse_clock(text):
    """'2:15 p.m.' -> time(14, 15)."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2}) ([ap])\.m\.", text.strip())
    if not m:
        raise ValueError(f"Bad clock text: {text!r}")
    hour, minute = int(m.group(1)) % 12, int(m.group(2))
    if m.group(3) == "p":
        hour += 12
    return time(hour, minute)


def market_session(dt_et):
    """Session of an ET time. Weekends only; exchange holidays are Person 2's call."""
    if dt_et.weekday() >= 5:
        return "weekend"
    t = dt_et.time()
    if t < OPEN:
        return "pre_market"
    if t < CLOSE:
        return "regular"
    return "after_hours"


def derive_row(ev, lag_min=DEFAULT_LAG_MIN):
    day = datetime.strptime(ev["release_date"], "%Y%m%d").date()
    verified = ev.get("release_time_et") is not None
    if verified:
        rel_et = datetime.combine(day, parse_clock(ev["release_time_et"]), ET)
        avail_et = rel_et + timedelta(minutes=lag_min)
        abbrev = rel_et.tzname()
        stated = re.search(r"\b(E[SD]T)\b", ev.get("time_evidence") or "")
        if stated and stated.group(1) != abbrev:
            raise ValueError(f"{ev['event_id']}: page says {stated.group(1)}, calendar gives {abbrev}")
        session = market_session(rel_et)
    else:
        # Time unknown: the only safe assumption is "not known until the day is over".
        rel_et = None
        avail_et = datetime.combine(day + timedelta(days=1), time(0, 0), ET)
        abbrev = avail_et.tzname()
        session = "unknown"
    row = {k: ev.get(k) for k in COLUMNS if k in ev}
    row.update(
        tz_abbrev=abbrev,
        release_datetime_et=rel_et.isoformat() if rel_et else "",
        release_utc=rel_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if rel_et else "",
        processing_lag_min=lag_min if verified else "",
        available_utc=avail_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        available_et=avail_et.isoformat(),
        market_session_et=session,
        time_verified=verified,
        usable_intraday=verified,
        meeting_date=ev.get("meeting_date") or (ev["release_date"] if ev["event_type"] != "fomc_minutes" else ""),
        notes=NOTES.get(ev["event_id"], ""),
    )
    return {k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS}


def load_evidence(path=EVIDENCE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_rows(evidence, lag_min=DEFAULT_LAG_MIN):
    rows = [derive_row(ev, lag_min) for ev in evidence["events"]]
    rows.sort(key=lambda r: (r["available_utc"], r["event_id"]))
    return rows


def integrity_problems(rows):
    """Checks a downstream user can rerun at any time. Empty list = OK."""
    problems = []
    ids = [r["event_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate event_id")
    hashes = [r["text_sha256"] for r in rows]
    if len(hashes) != len(set(hashes)):
        problems.append("duplicate text_sha256 (same text listed twice)")
    for r in rows:
        if not re.fullmatch(r"[0-9a-f]{64}", str(r["text_sha256"])):
            problems.append(f"{r['event_id']}: bad text hash")
        if not str(r["source_url"]).startswith("https://www.federalreserve.gov/"):
            problems.append(f"{r['event_id']}: source is not federalreserve.gov")
        if str(r["time_verified"]) in ("True", "true", True) and not r["time_evidence"]:
            problems.append(f"{r['event_id']}: verified time without evidence quote")
        if r["release_utc"] and r["available_utc"] < r["release_utc"]:
            problems.append(f"{r['event_id']}: available before release")
        if r["event_type"] == "fomc_minutes" and r["meeting_date"] >= r["release_date"]:
            problems.append(f"{r['event_id']}: minutes released on/before meeting")
    return problems


def write_csv(rows, path, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Write a new version (e.g. --out ..._v2.csv) instead of replacing it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path=OUT_CSV):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------- availability check ---
def to_utc(text, tz):
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(UTC)


def available_before(row, prediction_utc):
    """True if the release was public (plus lag) strictly before the prediction time."""
    return to_utc(row["available_utc"], UTC) < prediction_utc


def releases_known_at(rows, prediction_utc):
    return [r for r in rows if available_before(r, prediction_utc)]


# ----------------------------------------------------------- web verify ---
def clean_text(html, text_selector):
    from bs4 import BeautifulSoup  # only needed for `verify`

    parser = "lxml"
    try:
        import html5lib  # noqa: F401  (browser-identical parsing when installed)
        parser = "html5lib"
    except ImportError:
        pass
    soup = BeautifulSoup(html, parser)
    if text_selector.startswith("#article > div"):
        box = soup.select_one(text_selector)
    else:
        box = soup.select_one("#article") or soup.select_one("#leftText")
    if box is None:
        raise ValueError("text container not found")
    paragraphs = []
    for p in box.select("p"):
        for br in p.find_all("br"):
            br.replace_with(" ")
        t = re.sub(r"\s+", " ", p.get_text()).strip()
        if t and not re.match(r"For media inquiries", t, re.I):
            paragraphs.append(t)
    return "\n".join(paragraphs)


def page_text(html):
    from bs4 import BeautifulSoup

    return re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text())


def verify(evidence, save_text=False, timeout=30):
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "CMU-11785-GS-Fuse-research (FOMC release audit)"
    cache = {}

    def get(url):
        if url not in cache:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            cache[url] = r.content
        return cache[url]

    report = []
    for ev in evidence["events"]:
        item = {"event_id": ev["event_id"], "hash_ok": False, "time_ok": None, "error": ""}
        try:
            raw = get(ev["source_url"])
            if ev["text_format"] == "pdf":
                digest, text = hashlib.sha256(raw).hexdigest(), None
            else:
                text = clean_text(raw.decode("utf-8", "replace"), ev["text_selector"])
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            item["hash_ok"] = digest == ev["text_sha256"]
            if save_text:
                TEXT_DIR.mkdir(parents=True, exist_ok=True)
                if text is None:
                    (TEXT_DIR / f"{ev['event_id']}.pdf").write_bytes(raw)
                else:
                    (TEXT_DIR / f"{ev['event_id']}.txt").write_text(text, encoding="utf-8")
            if ev.get("release_time_et") and not ev["time_evidence_url"].endswith(".pdf"):
                quote = re.sub(r"\s+", " ", ev["time_evidence"]).strip()
                item["time_ok"] = quote in page_text(get(ev["time_evidence_url"]).decode("utf-8", "replace"))
        except Exception as e:  # report, keep going
            item["error"] = f"{type(e).__name__}: {e}"
        report.append(item)
    return report


# ------------------------------------------------------------ old compare ---
def compare_old(manifest_path, rows):
    manifest = json.loads(Path(manifest_path).read_text())
    entries = [e for k in ("train_events", "vali_events", "test_events") for e in manifest[k]]
    entries += manifest.get("article_date_screen", {}).get("excluded_event_ids", [])
    by_cat = {}
    for e in entries:
        m = re.search(r"(\d+)[\\/](\d{8})\.txt$", e)
        if m:
            by_cat.setdefault(m.group(1), set()).add(m.group(2))
    minutes = {r["release_date"] for r in rows if r["event_type"] == "fomc_minutes"}
    meetings = {r["meeting_date"] for r in rows if r["event_type"] == "fomc_minutes"}
    stmts = {r["release_date"] for r in rows if r["event_type"] != "fomc_minutes"}
    out = []
    for cat, dates in sorted(by_cat.items(), key=lambda x: int(x[0])):
        lo, hi = min(dates), max(dates)
        window_minutes = {d for d in minutes if lo <= d <= hi}
        is_minutes_folder = len(dates & minutes) > 0.5 * len(dates)
        out.append({
            "old_category": cat, "n_dates": len(dates), "first": lo, "last": hi,
            "match_minutes_release_date": len(dates & minutes),
            "match_minutes_meeting_date": len(dates & meetings),
            "match_statement_date": len(dates & stmts),
            "looks_like_fomc_minutes_folder": is_minutes_folder,
            "official_minutes_missing_in_old": " ".join(sorted(window_minutes - dates)) if is_minutes_folder else "",
            "old_dates_not_official_minutes": " ".join(sorted(dates - minutes)) if is_minutes_folder else "",
        })
    return out


# --------------------------------------------------------------------- cli ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--evidence", default=str(EVIDENCE))
    b.add_argument("--out", default=str(OUT_CSV))
    b.add_argument("--lag-min", type=int, default=DEFAULT_LAG_MIN)
    v = sub.add_parser("verify")
    v.add_argument("--evidence", default=str(EVIDENCE))
    v.add_argument("--save-text", action="store_true", help="save clean texts under downloads/ (git-ignored)")
    v.add_argument("--report", default=str(TEXT_DIR / "verify_report.json"))
    c = sub.add_parser("check-before")
    c.add_argument("--csv", default=str(OUT_CSV))
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--prediction-et", help='e.g. "2020-03-16 09:30" (New York time)')
    g.add_argument("--prediction-utc", help='e.g. "2020-03-16T13:30:00Z"')
    g.add_argument("--predictions-csv", help="CSV with a prediction_utc column; adds n_releases_known")
    o = sub.add_parser("compare-old")
    o.add_argument("--manifest", required=True)
    o.add_argument("--csv", default=str(OUT_CSV))
    o.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        rows = build_rows(load_evidence(args.evidence), args.lag_min)
        problems = integrity_problems(rows)
        if problems:
            sys.exit("Integrity problems:\n" + "\n".join(problems))
        write_csv(rows, args.out)
        n_t = sum(r["time_verified"] for r in rows)
        print(f"wrote {args.out}: {len(rows)} releases, {n_t} with verified time")
    elif args.cmd == "verify":
        report = verify(load_evidence(args.evidence), args.save_text)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1))
        bad = [r for r in report if not r["hash_ok"] or r["time_ok"] is False or r["error"]]
        print(f"checked {len(report)}; problems {len(bad)}; report {args.report}")
        for r in bad:
            print(r)
    elif args.cmd == "check-before":
        rows = read_csv(args.csv)
        if args.predictions_csv:
            preds = read_csv(args.predictions_csv)
            for p in preds:
                p["n_releases_known"] = len(releases_known_at(rows, to_utc(p["prediction_utc"], UTC)))
            out = Path(args.predictions_csv).with_name(Path(args.predictions_csv).stem + "_fomc_checked.csv")
            write_csv_generic(preds, out)
            print(f"wrote {out}")
        else:
            t = to_utc(args.prediction_et, ET) if args.prediction_et else to_utc(args.prediction_utc, UTC)
            known = releases_known_at(rows, t)
            latest = known[-1] if known else None
            print(f"prediction {t.isoformat()}: {len(known)} releases public before it")
            if latest:
                print(f"latest usable: {latest['event_id']} available {latest['available_utc']}")
    elif args.cmd == "compare-old":
        report = compare_old(args.manifest, read_csv(args.csv))
        if args.out:
            write_csv_generic(report, args.out)
            print(f"wrote {args.out}")
        for r in report:
            print(r)


def write_csv_generic(rows, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new name")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
