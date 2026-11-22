"""Person 1, v2: all official Federal Reserve announcements with verified public times.

v2 ADDS to v1 (it never edits v1 files):
    v1  analysis/macro_releases/fomc_releases_v1.csv       FOMC statements + minutes
    v2  analysis/macro_releases/fed_announcements_v2.csv   v1 rows + the rest below
        - Board monetary-policy press releases (discount rate, lending facilities,
          swap lines, economic projections, auctions, ...)
        - speeches by the Chair, Vice Chairs and Governors
        - Congressional testimony
        - FOMC press conferences (transcript PDF)

Each row has two times:
    event_time_et        when the event happened (release time, speech start, hearing start)
    market_time_et       when the market first had the CONTENT; available_utc is this time
                         plus a 5-minute lag. Use available_utc both to align prices and for
                         "was it known before the prediction?". The time the text file was
                         posted online does not matter: content spoken live is public.

Time sources (`evidence_level`):
    A_release_line            "For release at 2:00 p.m. EDT" printed on the page
    F_official_feed_time      timestamp in the Fed's own news feed
                              (/json/ne-press.json, ne-speeches.json, ne-testimony.json),
                              cross-checked against the Fed calendar (/json/calendar.json).
                              Speeches: delivery time; if feed and calendar disagree the row
                              is not usable. Testimony: the earlier of prepared-text release
                              and hearing start.
    P_after_live_event_day    press conferences: live on the meeting day, but the transcript
                              covers the whole event and its end time is not verified, so the
                              transcript counts as known only after that day ends
    C_date_only               date known, time not: available only from midnight ET after
                              that day; usable_intraday = False

Commands:
    collect   download feeds/pages from federalreserve.gov -> raw JSON (downloads/, git-ignored)
    evidence  raw JSON -> analysis/macro_releases/fed_announcements_evidence_v2.json
    build     evidence v2 + v1 CSV -> analysis/macro_releases/fed_announcements_v2.csv
    verify    re-download every v2 source and compare text hashes and times
    fetch-text  download the text of EVERY row in the v2 CSV (v1 rows included) to
              downloads/fed_announcements_v2/text/<event_id>.txt and check each hash.
              Why: source text may not be committed (team rule), so the CSV only holds
              links and fingerprints. Each teammate runs this once to get the texts
              locally, named by event_id so they join directly to the CSV.
    check-before   same as v1, on the v2 CSV

Typical:
    python scripts/build_fed_announcements_v2.py build
    python scripts/build_fed_announcements_v2.py verify
    python scripts/build_fed_announcements_v2.py fetch-text --usable-only
    python scripts/build_fed_announcements_v2.py check-before --prediction-et "2020-03-16 09:30"
"""

import argparse
import csv
from datetime import date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_fomc_releases import (  # noqa: E402  (reuse v1 helpers, v1 unchanged)
    ET, UTC, DEFAULT_LAG_MIN, market_session, parse_clock, read_csv, releases_known_at, to_utc,
)

BASE = "https://www.federalreserve.gov"
MACRO = ROOT / "analysis" / "macro_releases"
V1_CSV = MACRO / "fomc_releases_v1.csv"
EVIDENCE = MACRO / "fed_announcements_evidence_v2.json"
OUT_CSV = MACRO / "fed_announcements_v2.csv"
DL = ROOT / "downloads" / "fed_announcements_v2"   # git-ignored
RAW = DL / "raw_collect.json"
START_YEAR = 2008

COLUMNS = [
    "event_id", "source_version", "event_type", "subcategory", "speaker", "speaker_role",
    "title", "release_date", "event_time_et", "market_time_et", "tz_abbrev",
    "available_utc", "available_et", "processing_lag_min", "market_session_et",
    "time_verified", "usable_intraday", "evidence_level", "time_evidence",
    "time_evidence_url", "calendar_time_et", "calendar_check", "source_url",
    "text_format", "text_selector", "text_paragraphs", "text_chars", "text_sha256",
    "duplicate_of", "concurrent_with", "notes",
]


# ------------------------------------------------------------- parsing ---
def feed_dt(d):
    """'11/20/2026 12:15:00 PM' -> (date, time|None)."""
    parts = d.split()
    m, dd, y = (int(x) for x in parts[0].split("/"))
    day = date(y, m, dd)
    if len(parts) < 3:
        return day, None
    return day, datetime.strptime(parts[1] + " " + parts[2], "%I:%M:%S %p").time()


def clock_text(t):
    """time(14, 5) -> '2:05 p.m.'"""
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'a' if t.hour < 12 else 'p'}.m."


def cal_clock(text):
    m = re.search(r"(\d{1,2}):(\d{2})\s*([ap])\.?m", text or "", re.I)
    if not m:
        return None
    return parse_clock(f"{int(m.group(1))}:{m.group(2)} {m.group(3).lower()}.m.")


def speaker_role(s):
    s = s or ""
    for key, role in [("Vice Chair for Supervision", "vice_chair_supervision"),
                      ("Vice Chairman for Supervision", "vice_chair_supervision"),
                      ("Vice Chair", "vice_chair"), ("Vice Chairman", "vice_chair"),
                      ("Chair", "chair"), ("Chairman", "chair"), ("Governor", "governor")]:
        if s.startswith(key):
            return role
    return "staff_or_other"


def last_name(s):
    return re.sub(r",.*$", "", s or "").strip().split(" ")[-1].lower()


def board_subcategory(title):
    t = title.lower()
    rules = [
        (r"economic projections", "economic_projections"),
        (r"discount rate|primary credit|discount window", "discount_rate"),
        (r"swap", "swap_lines"),
        (r"term auction|\btaf\b|auction|term deposit|offering of", "auction_or_operation"),
        (r"facility|facilities|lending|credit|liquidity", "facility_or_lending"),
        (r"schedule", "meeting_schedule"),
        (r"open market|fomc|monetary policy", "other_monetary_policy"),
    ]
    for pat, name in rules:
        if re.search(pat, t):
            return name
    return "other"


# -------------------------------------------------------- calendar check ---
def calendar_index(cal):
    idx = {}
    for e in cal:
        if not e.get("m") or not e.get("dd"):
            continue
        key = f"{e['m']}-{int(e['dd']):02d}"
        idx.setdefault(key, []).append(e)
    return idx


def calendar_match(idx, day, speaker, feed_time):
    """Return (calendar time | None, check label)."""
    cands = [e for e in idx.get(day.isoformat(), [])
             if e.get("ty") in ("Speeches", "Testimony") and last_name(speaker) in (e.get("ti") or "").lower()]
    times = [cal_clock(e.get("tm")) for e in cands]
    times = [t for t in times if t]
    if not times:
        return None, "not_in_calendar"
    if feed_time in times:
        return feed_time, "agrees"
    if len(times) == 1:
        return times[0], "differs"
    return None, "ambiguous_multiple_events"


# ------------------------------------------------------------- evidence ---
def v1_urls(v1_rows):
    """Pages already covered by v1 (statement pages and the minutes press releases)."""
    return {r["source_url"] for r in v1_rows} | {r["time_evidence_url"] for r in v1_rows}


def evidence_from_raw(raw, v1_rows=()):
    """Turn collected raw metadata into reviewed evidence records (no source text)."""
    idx = calendar_index(raw.get("cal", []))
    events, excluded = [], []
    covered = v1_urls(v1_rows)
    for kind in ("mp", "sp", "te"):
        keep = []
        for x in raw[kind]:
            if BASE + x.get("l", "") in covered:
                excluded.append({"feed": kind, "d": x["d"], "title": x["t"], "reason": "already in v1"})
            elif not x.get("l", "").startswith("/"):
                # Feed entries with no page of their own (e.g. "identical remarks" given elsewhere)
                excluded.append({"feed": kind, "d": x["d"], "title": x["t"], "reason": "no own page link in feed"})
            else:
                keep.append(x)
        raw = {**raw, kind: keep}

    for x in raw["mp"]:
        day, ftime = feed_dt(x["d"])
        slug = x["l"].rsplit("/", 1)[1].replace(".htm", "")
        ev = dict(event_id="board_" + slug, event_type="fed_board_monetary_release",
                  subcategory=board_subcategory(x["t"]), speaker="", title=x["t"],
                  release_date=day.strftime("%Y%m%d"), source_url=BASE + x["l"],
                  text_format="html", text_selector=x.get("sel"), text_paragraphs=x.get("n_par"),
                  text_chars=x.get("chars"), text_sha256=x.get("sha256"), release_line=x.get("release_line"))
        m = re.search(r"at (\d{1,2}:\d{2} [ap]\.m\.)", x.get("release_line") or "")
        if m:
            ev.update(event_time_et=m.group(1), evidence_level="A_release_line",
                      time_evidence=x["release_line"], time_evidence_url=BASE + x["l"])
        elif ftime:
            ev.update(event_time_et=clock_text(ftime), evidence_level="F_official_feed_time",
                      time_evidence=f"ne-press.json d={x['d']} (page says '{x.get('release_line')}')",
                      time_evidence_url=BASE + "/json/ne-press.json")
        else:
            ev.update(event_time_et=None, evidence_level="C_date_only",
                      time_evidence=f"ne-press.json d={x['d']} (date only); page says '{x.get('release_line')}'",
                      time_evidence_url=BASE + "/json/ne-press.json")
        ev["market_time_et"] = ev["event_time_et"]
        events.append(ev)

    for kind, feed_name in (("sp", "ne-speeches.json"), ("te", "ne-testimony.json")):
        for x in raw[kind]:
            day, ftime = feed_dt(x["d"])
            slug = x["l"].rsplit("/", 1)[1].replace(".htm", "")
            ctime, check = calendar_match(idx, day, x.get("s"), ftime)
            ev = dict(event_id=("speech_" if kind == "sp" else "testimony_") + slug,
                      event_type="fed_speech" if kind == "sp" else "fed_testimony",
                      subcategory="", speaker=x.get("s", ""), title=x["t"],
                      release_date=day.strftime("%Y%m%d"), source_url=BASE + x["l"],
                      text_format="html", text_selector=x.get("sel"), text_paragraphs=x.get("n_par"),
                      text_chars=x.get("chars"), text_sha256=x.get("sha256"),
                      calendar_time_et=clock_text(ctime) if ctime else "", calendar_check=check,
                      location=x.get("lo", ""))
            if ftime:
                # Market time = when the content first became public.
                # Speech: the feed time (delivery). If the calendar shows a different time
                #   for the same speaker and day, we cannot tell which is right -> not usable.
                # Testimony: the prepared text is often released before the hearing
                #   (e.g. 8:30 a.m. text, 10:00 a.m. hearing), so the EARLIER time is when
                #   the market learned it.
                if kind == "te" and ctime:
                    text_t = min(ftime, ctime)
                else:
                    text_t = ftime
                if kind == "sp" and check in ("differs", "ambiguous_multiple_events"):
                    ev["timing_conflict"] = True
                ev.update(event_time_et=clock_text(ctime or ftime), market_time_et=clock_text(text_t),
                          evidence_level="F_official_feed_time",
                          time_evidence=f"{feed_name} d={x['d']}" + (f"; calendar.json {clock_text(ctime)}" if ctime else ""),
                          time_evidence_url=BASE + "/json/" + feed_name)
            else:
                ev.update(event_time_et=None, market_time_et=None, evidence_level="C_date_only",
                          time_evidence=f"{feed_name} d={x['d']} (date only)",
                          time_evidence_url=BASE + "/json/" + feed_name)
            events.append(ev)

    for x in raw["pc"]:
        day = datetime.strptime(x["id"], "%Y%m%d").date()
        cal = [e for e in idx.get(day.isoformat(), []) if "press conference" in (e.get("ti") or "").lower()]
        ctime = cal_clock(cal[0]["tm"]) if cal else None
        lm = parsedate_to_datetime(x["last_modified"]).astimezone(ET)
        # The press conference is live on the meeting-day afternoon, so its content is public
        # that day, but the transcript covers the whole event. Without a verified end time,
        # the text counts as known only after the meeting day ends (midnight ET).
        after_day = datetime.combine(day + timedelta(days=1), time(0, 0), ET)
        events.append(dict(
            event_id="presconf_" + x["id"], event_type="fomc_press_conference", subcategory="transcript",
            speaker="Chair", title=f"FOMC press conference transcript, {day.isoformat()}",
            release_date=day.strftime("%Y%m%d"), source_url=BASE + x["pdf"],
            text_format="pdf", text_selector="pdf_bytes", text_paragraphs=None, text_chars=None,
            text_sha256=x["sha256"], event_time_et=clock_text(ctime) if ctime else None,
            calendar_time_et=clock_text(ctime) if ctime else "", calendar_check="agrees" if ctime else "not_in_calendar",
            market_datetime_et=after_day.isoformat(), evidence_level="P_after_live_event_day",
            time_evidence=(f"Live press conference on {day.isoformat()}"
                           + (f" at {clock_text(ctime)} (calendar.json)" if ctime else "")
                           + "; transcript content used only after that day ends. "
                           + f"Transcript PDF posted later (HTTP Last-Modified: {x['last_modified']})."),
            time_evidence_url=BASE + x["page"], page_url=BASE + x["page"]))

    return {"schema_version": 2, "name": "fed_announcements_evidence_v2",
            "captured_at_utc": raw.get("captured_at"), "excluded": excluded, "events": events}


# ---------------------------------------------------------------- build ---
def _avail(day, text_time, lag, text_dt=None):
    if text_dt is not None:
        rel = text_dt
    elif text_time is not None:
        rel = datetime.combine(day, text_time, ET)
    else:
        return None, datetime.combine(day + timedelta(days=1), time(0, 0), ET)
    return rel, rel + timedelta(minutes=lag)


def derive_v2_row(ev, lag=DEFAULT_LAG_MIN):
    day = datetime.strptime(ev["release_date"], "%Y%m%d").date()
    text_dt = datetime.fromisoformat(ev["market_datetime_et"]) if ev.get("market_datetime_et") else None
    text_t = parse_clock(ev["market_time_et"]) if ev.get("market_time_et") else None
    rel, avail = _avail(day, text_t, lag, text_dt)
    verified = rel is not None
    text_ok = bool(ev.get("text_sha256")) and ev.get("text_chars") != 0
    conflict = bool(ev.get("timing_conflict"))
    notes = []
    if ev["event_type"] == "fomc_press_conference":
        notes.append("Live event on the meeting day; the full transcript is used only after that day ends.")
    if conflict:
        notes.append("Feed and calendar give different times for this speech; not usable until resolved.")
    elif ev["event_type"] == "fed_testimony" and ev.get("calendar_check") == "differs":
        notes.append("Prepared text and hearing times differ; the earlier one is when the market learned it.")
    if not text_ok:
        notes.append("No paragraph text on the page; do not use as a text sample.")
    row = dict(
        event_id=ev["event_id"], source_version="v2", event_type=ev["event_type"],
        subcategory=ev.get("subcategory", ""), speaker=ev.get("speaker", ""),
        speaker_role=speaker_role(ev.get("speaker")) if ev.get("speaker") else "",
        title=ev["title"], release_date=ev["release_date"],
        event_time_et=ev.get("event_time_et") or "",
        market_time_et=(rel.strftime("%Y-%m-%d ") + clock_text(rel.time())) if rel else "",
        tz_abbrev=(rel or avail).tzname(),
        available_utc=avail.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        available_et=avail.isoformat(),
        processing_lag_min=lag if verified else "",
        market_session_et=market_session(rel) if verified else "unknown",
        time_verified=verified, usable_intraday=verified and text_ok and not conflict,
        evidence_level=ev["evidence_level"], time_evidence=ev["time_evidence"],
        time_evidence_url=ev["time_evidence_url"], calendar_time_et=ev.get("calendar_time_et", ""),
        calendar_check=ev.get("calendar_check", ""), source_url=ev["source_url"],
        text_format=ev["text_format"], text_selector=ev.get("text_selector") or "",
        text_paragraphs="" if ev.get("text_paragraphs") is None else ev["text_paragraphs"],
        text_chars="" if ev.get("text_chars") is None else ev["text_chars"],
        text_sha256=ev.get("text_sha256") or "", duplicate_of="", concurrent_with="",
        notes=" ".join(notes))
    return row


def v1_as_v2(r):
    """Map a v1 FOMC row into the v2 columns without changing its values."""
    return dict(
        event_id=r["event_id"], source_version="v1", event_type=r["event_type"], subcategory="",
        speaker="", speaker_role="", title=r["title"], release_date=r["release_date"],
        event_time_et=r["release_time_et"],
        market_time_et=(r["release_datetime_et"][:10] + " " + r["release_time_et"]) if r["release_time_et"] else "",
        tz_abbrev=r["tz_abbrev"], available_utc=r["available_utc"], available_et=r["available_et"],
        processing_lag_min=r["processing_lag_min"], market_session_et=r["market_session_et"],
        time_verified=r["time_verified"] == "True", usable_intraday=r["usable_intraday"] == "True",
        evidence_level=r["evidence_level"], time_evidence=r["time_evidence"],
        time_evidence_url=r["time_evidence_url"], calendar_time_et="", calendar_check="",
        source_url=r["source_url"], text_format=r["text_format"], text_selector=r["text_selector"],
        text_paragraphs=r["text_paragraphs"], text_chars=r["text_chars"], text_sha256=r["text_sha256"],
        duplicate_of="", concurrent_with="", notes=r["notes"])


def build_v2_rows(evidence, v1_rows, lag=DEFAULT_LAG_MIN):
    rows = [v1_as_v2(r) for r in v1_rows] + [derive_v2_row(ev, lag) for ev in evidence["events"]]
    rows.sort(key=lambda r: (r["available_utc"], r["event_id"]))
    first = {}
    for r in rows:   # same text published twice -> keep the earliest, flag the rest
        h = r["text_sha256"]
        if not h or r["text_chars"] in (0, "0"):
            continue
        if h in first:
            r["duplicate_of"] = first[h]
            r["usable_intraday"] = False
        else:
            first[h] = r["event_id"]
    by_minute = {}
    for r in rows:
        if r["time_verified"] in (True, "True"):
            by_minute.setdefault(r["available_utc"], []).append(r["event_id"])
    for r in rows:
        group = by_minute.get(r["available_utc"], [])
        if len(group) > 1 and r["time_verified"] in (True, "True"):
            r["concurrent_with"] = " ".join(g for g in group if g != r["event_id"])
    return rows


def v2_problems(rows):
    problems = []
    ids = [r["event_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate event_id")
    for r in rows:
        if not str(r["source_url"]).startswith(BASE + "/"):
            problems.append(f"{r['event_id']}: not a federalreserve.gov source")
        if r["text_sha256"] and not re.fullmatch(r"[0-9a-f]{64}", r["text_sha256"]):
            problems.append(f"{r['event_id']}: bad hash")
        if not r["time_evidence"]:
            problems.append(f"{r['event_id']}: no time evidence")
        if r["evidence_level"].startswith("C_") and str(r["usable_intraday"]) == "True":
            problems.append(f"{r['event_id']}: date-only row marked usable")
        if r["duplicate_of"] and str(r["usable_intraday"]) == "True":
            problems.append(f"{r['event_id']}: duplicate marked usable")
    seen = {}
    for r in rows:
        if str(r["usable_intraday"]) == "True" and r["text_sha256"]:
            if r["text_sha256"] in seen:
                problems.append(f"{r['event_id']}: same text as {seen[r['text_sha256']]} and both usable")
            seen[r["text_sha256"]] = r["event_id"]
    return problems


def write_new_csv(rows, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. Write a new version instead of replacing it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_new_json(obj, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. Write a new version instead of replacing it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------- web: collect ---
def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "CMU-11785-GS-Fuse-research (Fed announcement audit)"
    return s


def _soup(html):
    from bs4 import BeautifulSoup

    try:
        import html5lib  # noqa: F401
        return BeautifulSoup(html, "html5lib")
    except ImportError:
        return BeautifulSoup(html, "lxml")


def clean_doc(soup):
    """Same rule as v1: <p> paragraphs of the article column, whitespace collapsed."""
    sel, box = "#article > div.col-sm-8:not(.heading)", None
    box = soup.select_one(sel)
    if box is None:
        for s in ("#article", "#leftText"):
            box = soup.select_one(s)
            if box is not None:
                sel = s
                break
    if box is None:
        return {"sel": None}
    paras = []
    for p in box.select("p"):
        for br in p.find_all("br"):
            br.replace_with(" ")
        t = re.sub(r"\s+", " ", p.get_text()).strip()
        if t and not re.match(r"For media inquiries", t, re.I):
            paras.append(t)
    text = "\n".join(paras)
    return {"sel": sel, "n_par": len(paras), "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "_text": text}


def _get_text(s, url):
    """Decode as UTF-8 ourselves: the Fed's JSON feeds start with a BOM, and requests may
    guess a wrong charset for pages without one (that would change the text hashes)."""
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return r.content.decode("utf-8-sig", "replace")


def _feed(s, name):
    return [x for x in json.loads(_get_text(s, f"{BASE}/json/{name}").strip()) if x and x.get("d")]


def collect(v1_rows, save_text=False):
    s = _session()
    year = lambda x: int(x["d"].split()[0].split("/")[2])  # noqa: E731
    in_v1 = v1_urls(v1_rows)
    raw = {"captured_at": datetime.now(UTC).isoformat()}

    def page(x, extra):
        if not x.get("l", "").startswith("/"):
            return {k: x.get(k) for k in extra} | dict(l=x.get("l", ""), d=x["d"], t=x["t"], status=None)
        try:
            soup = _soup(_get_text(s, BASE + x["l"]))
        except Exception as e:  # keep going; verify will report this page
            return {k: x.get(k) for k in extra} | dict(l=x["l"], d=x["d"], t=x["t"], status=f"error: {e}")
        info = clean_doc(soup)
        text = info.pop("_text", None)
        if save_text and text is not None:
            (DL / "text").mkdir(parents=True, exist_ok=True)
            (DL / "text" / (x["l"].rsplit("/", 1)[1].replace(".htm", ".txt"))).write_text(text, encoding="utf-8")
        out = {k: x.get(k) for k in extra}
        out.update(l=x["l"], d=x["d"], t=x["t"], status=200, **info)
        if "release_line" in extra:
            head = soup.select_one("#article .heading")
            m = re.search(r"For (immediate )?release( at [0-9:]+ [ap]\.m\.( E[SD]T)?)?", head.get_text(" ") if head else "", re.I)
            out["release_line"] = re.sub(r"\s+", " ", m.group(0)) if m else None
        return out

    press = [x for x in _feed(s, "ne-press.json") if x.get("pt") == "Monetary Policy" and year(x) >= START_YEAR
             and BASE + x["l"] not in in_v1]
    raw["mp"] = [page(x, ["release_line"]) for x in press]
    raw["sp"] = [page(x, ["s", "lo", "a"]) for x in _feed(s, "ne-speeches.json") if year(x) >= START_YEAR]
    raw["te"] = [page(x, ["s", "lo", "a"]) for x in _feed(s, "ne-testimony.json") if year(x) >= START_YEAR]
    cal = json.loads(_get_text(s, f"{BASE}/json/calendar.json"))["events"]
    raw["cal"] = [dict(m=e.get("month"), dd=e.get("days"), ty=e.get("type"), ti=e.get("title"), tm=e.get("time"))
                  for e in cal if e.get("month") and re.search(r"Speeches|Testimony|FOMC", e.get("type") or "")]
    ids = set()
    for p in ["/monetarypolicy/fomccalendars.htm"] + [f"/monetarypolicy/fomchistorical{y}.htm" for y in range(START_YEAR, 2021)]:
        r = s.get(BASE + p, timeout=60)
        if r.status_code == 200:
            ids |= set(re.findall(r"fomcpresconf(\d{8})\.htm", r.text))
    raw["pc"] = []
    for i in sorted(ids):
        url = f"/mediacenter/files/FOMCpresconf{i}.pdf"
        r = s.get(BASE + url, timeout=60)
        if r.status_code != 200:
            continue
        raw["pc"].append(dict(id=i, page=f"/monetarypolicy/fomcpresconf{i}.htm", pdf=url, status=200,
                              last_modified=r.headers.get("Last-Modified"), bytes=len(r.content),
                              sha256=hashlib.sha256(r.content).hexdigest()))
    return raw


TEXT_DIR = DL / "text"


def clean_by_selector(soup, selector):
    """Extract text with the exact container recorded in the CSV (`text_selector`),
    so the hash matches the one computed when the row was built."""
    box = soup.select_one(selector)
    if box is None:
        return None
    paras = []
    for p in box.select("p"):
        for br in p.find_all("br"):
            br.replace_with(" ")
        t = re.sub(r"\s+", " ", p.get_text()).strip()
        if t and not re.match(r"For media inquiries", t, re.I):
            paras.append(t)
    return "\n".join(paras)


def pdf_to_text(data):
    """Plain text of a PDF (press-conference transcripts, June 2008 minutes)."""
    from io import BytesIO

    from pypdf import PdfReader  # pip install pypdf

    reader = PdfReader(BytesIO(data))
    return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def fetch_texts(rows, usable_only=False, skip_existing=False, out_dir=TEXT_DIR):
    """Download and save the text of each row; return a per-row report.

    hash_ok compares against `text_sha256`:
      html rows: SHA-256 of the extracted text (identical to what we verified)
      pdf rows:  SHA-256 of the PDF file itself; the .txt is converted from that exact file
    """
    s = _session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for r in rows:
        if usable_only and str(r["usable_intraday"]) != "True":
            continue
        if not r["text_sha256"] or str(r["text_chars"]) == "0":
            continue
        path = out_dir / f"{r['event_id']}.txt"
        item = {"event_id": r["event_id"], "file": str(path), "hash_ok": False, "error": ""}
        if skip_existing and path.exists():
            item.update(hash_ok=None, error="skipped (exists)")
            report.append(item)
            continue
        try:
            resp = s.get(r["source_url"], timeout=60)
            resp.raise_for_status()
            if r["text_format"] == "pdf":
                item["hash_ok"] = hashlib.sha256(resp.content).hexdigest() == r["text_sha256"]
                text = pdf_to_text(resp.content)
            else:
                text = clean_by_selector(_soup(resp.content.decode("utf-8-sig", "replace")), r["text_selector"])
                if text is None:
                    raise ValueError(f"container {r['text_selector']!r} not found")
                item["hash_ok"] = hashlib.sha256(text.encode("utf-8")).hexdigest() == r["text_sha256"]
            path.write_text(text, encoding="utf-8")
        except Exception as e:  # keep going; the report lists failures
            item["error"] = f"{type(e).__name__}: {e}"
        report.append(item)
    return report


def verify(evidence, raw_now):
    """Compare a fresh collect against the committed evidence."""
    fresh = {e["event_id"]: e for e in evidence_from_raw(raw_now, read_csv(V1_CSV))["events"]}
    report = []
    for ev in evidence["events"]:
        f = fresh.get(ev["event_id"])
        item = {"event_id": ev["event_id"], "found": f is not None}
        if f:
            item["hash_ok"] = f["text_sha256"] == ev["text_sha256"]
            item["time_ok"] = (f.get("market_time_et") == ev.get("market_time_et")
                               and f.get("market_datetime_et") == ev.get("market_datetime_et"))
        report.append(item)
    return report


# ------------------------------------------------------------------ cli ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default=str(RAW))
    c.add_argument("--save-text", action="store_true")
    e = sub.add_parser("evidence")
    e.add_argument("--raw", default=str(RAW))
    e.add_argument("--out", default=str(EVIDENCE))
    b = sub.add_parser("build")
    b.add_argument("--evidence", default=str(EVIDENCE))
    b.add_argument("--v1", default=str(V1_CSV))
    b.add_argument("--out", default=str(OUT_CSV))
    b.add_argument("--lag-min", type=int, default=DEFAULT_LAG_MIN)
    v = sub.add_parser("verify")
    v.add_argument("--evidence", default=str(EVIDENCE))
    v.add_argument("--save-text", action="store_true")
    t = sub.add_parser("fetch-text", help="download the text of every CSV row, named by event_id")
    t.add_argument("--csv", default=str(OUT_CSV))
    t.add_argument("--out-dir", default=str(TEXT_DIR))
    t.add_argument("--usable-only", action="store_true", help="only rows with usable_intraday=True")
    t.add_argument("--skip-existing", action="store_true", help="resume: keep files already downloaded")
    k = sub.add_parser("check-before")
    k.add_argument("--csv", default=str(OUT_CSV))
    k.add_argument("--prediction-et")
    k.add_argument("--prediction-utc")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        raw = collect(read_csv(V1_CSV), args.save_text)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(raw))
        print(f"wrote {args.out}: " + ", ".join(f"{k}={len(v)}" for k, v in raw.items() if isinstance(v, list)))
    elif args.cmd == "evidence":
        ev = evidence_from_raw(json.loads(Path(args.raw).read_text()), read_csv(V1_CSV))
        write_new_json(ev, args.out)
        print(f"wrote {args.out}: {len(ev['events'])} events")
    elif args.cmd == "build":
        rows = build_v2_rows(json.loads(Path(args.evidence).read_text(encoding="utf-8")), read_csv(args.v1), args.lag_min)
        problems = v2_problems(rows)
        if problems:
            sys.exit("Integrity problems:\n" + "\n".join(problems[:50]))
        write_new_csv(rows, args.out)
        print(f"wrote {args.out}: {len(rows)} rows, {sum(str(r['usable_intraday']) == 'True' for r in rows)} usable intraday")
    elif args.cmd == "verify":
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        raw_now = collect(read_csv(V1_CSV), args.save_text)
        report = verify(evidence, raw_now)
        DL.mkdir(parents=True, exist_ok=True)
        (DL / "verify_report.json").write_text(json.dumps(report, indent=1))
        bad = [r for r in report if not r["found"] or not r.get("hash_ok") or not r.get("time_ok")]
        print(f"checked {len(report)}; problems {len(bad)}; report {DL / 'verify_report.json'}")
        for r in bad[:30]:
            print(r)
    elif args.cmd == "fetch-text":
        report = fetch_texts(read_csv(args.csv), args.usable_only, args.skip_existing, args.out_dir)
        rep = Path(args.out_dir).parent / "fetch_text_report.json"
        rep.write_text(json.dumps(report, indent=1))
        bad = [r for r in report if r["hash_ok"] is False]
        print(f"saved {sum(1 for r in report if not r['error'])} texts to {args.out_dir}; "
              f"hash mismatches or errors: {len(bad)}; report {rep}")
        for r in bad[:30]:
            print(r)
    elif args.cmd == "check-before":
        rows = read_csv(args.csv)
        t = to_utc(args.prediction_et, ET) if args.prediction_et else to_utc(args.prediction_utc, UTC)
        known = [r for r in releases_known_at(rows, t) if r["usable_intraday"] == "True"]
        print(f"prediction {t.isoformat()}: {len(known)} usable announcements public before it")
        if known:
            print(f"latest: {known[-1]['event_id']} available {known[-1]['available_utc']}")


if __name__ == "__main__":
    main()
