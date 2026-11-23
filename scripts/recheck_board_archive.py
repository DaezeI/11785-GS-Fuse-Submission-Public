"""Re-run archive lookups for Board release rows left 'archive lookup incomplete'.

Resumable: appends one JSON line per event to --out and skips events already there.
Stops cleanly after --budget seconds. Shard with --shard i/n for parallel runs.

Per event: today's catalog page (body + "For release at" line); Wayback CDX for the
current and legacy URL from 7 days before the cutoff to 24 hours after; replay of the
latest capture at or before the cutoff, else the earliest capture after it; letters-only
comparison of the replayed body with today's body.
"""
import argparse, csv, datetime as dt, json, re, sys, time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_fomc_statement_versions import norm_para, paragraphs, sha, soup  # noqa: E402

UTC = dt.timezone.utc
FURN = re.compile(r"^(For release at|Release Date|Last Update|Share|Related|For media inquiries|Return to top|"
                  r"Accessible Version|Board of Governors of the Federal Reserve System|Home \|)", re.I)
S = requests.Session(); S.headers["User-Agent"] = "CMU-11785-GS-Fuse-research (Board release version audit)"


def get(url, delay):
    for k in range(3):
        try:
            r = S.get(url, timeout=60)
        except Exception as e:
            last = type(e).__name__; time.sleep(5); continue
        if r.status_code == 429:
            last = 429; time.sleep(60); continue
        time.sleep(delay); return r
    return last


def body_letters(box):
    return re.sub(r"[^a-z]", "", " ".join(norm_para(p) for p in paragraphs(box) if not FURN.match(norm_para(p))).lower())


def line_ok(line, cutoff):
    """True when the page's 'For release at' clock, on the cutoff's ET date, is no later than the cutoff."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    m = re.match(r"(\d{1,2}):(\d{2})\s*([ap])", line)
    if not m:
        return False
    h, mi = int(m.group(1)), int(m.group(2))
    h += 12 if m.group(3) == "p" and h != 12 else 0
    h = 0 if m.group(3) == "a" and h == 12 else h
    c = cutoff.astimezone(et)
    return c.replace(hour=h, minute=mi, second=0) <= c


def find_box(s, selector):
    for sel in (selector, "#article > div.col-sm-8:not(.heading)", "#leftText", "#article", "#content"):
        if sel and (b := s.select_one(sel)) is not None:
            return b, sel
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--delay", type=float, default=2.5)
    a = ap.parse_args()
    t_start = time.time()
    i, n = map(int, a.shard.split("/"))
    cat = {r["event_id"]: r for r in csv.DictReader(open(a.catalog, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(a.ledger, encoding="utf-8"))
            if r["event_type"] == "fed_board_monetary_release" and "incomplete" in r["eligibility"]]
    rows = [r for k, r in enumerate(rows) if k % n == i]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["event_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in rows:
            if r["event_id"] in done:
                continue
            if time.time() - t_start > a.budget:
                break
            c, eid = cat[r["event_id"]], r["event_id"]
            cutoff = dt.datetime.fromisoformat(r["text_cutoff_utc"]).astimezone(UTC)
            rec = {"event_id": eid, "split": r["split"], "text_cutoff_utc": cutoff.isoformat(), "source_url": c["source_url"]}
            cur = get(c["source_url"], 1.0)
            today = None
            if hasattr(cur, "ok") and cur.ok:
                s = soup(cur.content)
                box, _ = find_box(s, c["text_selector"])
                today = body_letters(box) if box is not None else None
                m = re.search(r"For release at\s*([0-9:]+\s*[ap]\.m\.\s*E[SD]T)", s.get_text(" "))
                rec["release_line"] = m.group(1) if m else None
            else:
                rec["blocked"] = f"current page {getattr(cur, 'status_code', cur)}"
            bare = re.sub(r"^https?://", "", c["source_url"])
            urls = [bare]
            m = re.search(r"/newsevents/pressreleases/monetary(\d{8}[a-z]\d?)\.htm$", bare)
            if m:
                urls.append(f"www.federalreserve.gov/newsevents/press/monetary/{m.group(1)}.htm")
            caps, errs = [], []
            for u in urls:
                q = (f"https://web.archive.org/cdx/search/cdx?url={u}&from={(cutoff - dt.timedelta(days=7)):%Y%m%d%H%M%S}"
                     f"&to={(cutoff + dt.timedelta(hours=24)):%Y%m%d%H%M%S}&output=json&fl=timestamp,statuscode,original&limit=200")
                rr = get(q, a.delay)
                if not (hasattr(rr, "ok") and rr.ok):
                    errs.append(str(getattr(rr, "status_code", rr))); continue
                for x in (rr.json()[1:] if rr.text.strip() else []):
                    if x[1] == "200":
                        caps.append((dt.datetime.strptime(x[0], "%Y%m%d%H%M%S").replace(tzinfo=UTC), x[0], x[2]))
            rec["captures_in_window"] = len(caps)
            if errs and len(errs) == len(urls):
                rec["disposition"] = "blocked_lookup"; rec["reason"] = "archive index unavailable: " + ",".join(errs)
            elif today is None:
                rec["disposition"] = "blocked_lookup"; rec["reason"] = rec.get("blocked", "today's body not extracted")
            else:
                pre = sorted(x for x in caps if x[0] <= cutoff)
                post = sorted(x for x in caps if x[0] > cutoff)
                pick = pre[-1] if pre else (post[0] if post else None)
                if pick is None:
                    rec["disposition"] = "insufficient_evidence"
                    rec["reason"] = "no archive copy from 7 days before the cutoff to 24 hours after"
                else:
                    rec["capture_utc"] = pick[0].isoformat(); rec["capture_minutes_after_cutoff"] = round((pick[0] - cutoff).total_seconds() / 60, 1)
                    rec["replay_url"] = f"https://web.archive.org/web/{pick[1]}id_/{pick[2]}"
                    rp = get(rec["replay_url"], a.delay)
                    if not (hasattr(rp, "ok") and rp.ok):
                        rec["disposition"] = "blocked_lookup"; rec["reason"] = f"replay {getattr(rp, 'status_code', rp)}"
                    else:
                        box, sel = find_box(soup(rp.content), c["text_selector"])
                        old = body_letters(box) if box is not None else None
                        rec["replay_selector"] = sel
                        if old is None:
                            rec["disposition"] = "blocked_lookup"; rec["reason"] = "archived page body not found"
                        elif old != today:
                            rec["disposition"] = "version_conflict_review"
                            rec["reason"] = "archived body differs from today's body"
                            rec["letters_today_vs_archive"] = [len(today), len(old)]
                        elif pick[0] <= cutoff:
                            rec["disposition"] = "eligible_archived_copy_before_cutoff"
                            rec["reason"] = "archive copy at or before the cutoff has today's text"
                        elif rec.get("release_line") and line_ok(rec["release_line"], cutoff):
                            rec["disposition"] = "eligible_document_release_line_release_day_capture"
                            rec["reason"] = (f"page says 'For release at {rec['release_line']}'; archive copy "
                                             f"{rec['capture_minutes_after_cutoff']} min after the cutoff has today's text")
                        else:
                            rec["disposition"] = "insufficient_evidence"
                            rec["reason"] = "release-day copy matches but the page's release time is missing or after the cutoff"
                        rec["today_body_letters_sha256"] = sha(today)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(eid, rec["disposition"], flush=True)
    print("REMAINING_IN_SHARD", len({r["event_id"] for r in rows} - {json.loads(l)["event_id"] for l in open(a.out, encoding="utf-8") if l.strip()}))


if __name__ == "__main__":
    main()
