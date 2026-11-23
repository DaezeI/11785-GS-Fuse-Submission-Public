"""Pilot: tie a specific FOMC statement text version to its public release time.

For each requested FOMC statement in the frozen Fed catalog this script

1. fetches today's press-release page, extracts it with the catalog's recorded
   selector and checks the catalog SHA-256 (current version identity);
2. fetches the FOMC minutes of the same meeting and extracts the statement that
   the Committee approved "for release at H:MM p.m." (official record of the
   released text and its release time);
3. optionally asks the Wayback CDX index for same-day captures of the press
   release URL and replays the earliest capture at or after the release time
   (contemporaneous corroboration).

It writes one JSON ledger row per event. Raw HTML stays in --raw-dir, which
must be outside Git. Nothing here reads prices or test-split data.

    python scripts/verify_fomc_statement_versions.py \
        --catalog analysis/macro_releases/fed_announcements_v2.csv \
        --population downloads/fed-spx-endpoints-price-v1/population.csv \
        --events stmt_20130731a stmt_20210728a \
        --raw-dir downloads/fomc-statement-evidence-pilot/raw \
        --out downloads/fomc-statement-evidence-pilot/ledger.json --wayback
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UA = "CMU-11785-GS-Fuse-research (FOMC statement version audit)"
FED = "https://www.federalreserve.gov"
DROP_PREFIX = re.compile(
    r"^(Voting for|Voting against|Votes for|Votes against|For media inquiries|Implementation Note|"
    r"Implementation note|Related Information|For release at|Release Date|Last Update|Share)",
    re.I,
)
RELEASE_RE = re.compile(
    r"(?:statement below|following statement|text below)[^.:]{0,40}?"
    r"(?:to be released|for release|was released|to be issued)\s+at\s+"
    r"(\d{1,2}:\d{2})\s*([ap])\.m\.", re.I)
FRACTIONS = {"\u00bc": "1/4", "\u00bd": "1/2", "\u00be": "3/4"}


TRANS = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                       "\u2013": "-", "\u2010": "-", "\u2011": "-", "\u00a0": " "})


def sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def soup(content: bytes, declared_charset: bool = False):
    """UTF-8 decoding matches the catalog builder; older minutes pages declare
    iso-8859-1, so those are parsed from bytes with their declared charset."""
    from bs4 import BeautifulSoup
    if declared_charset:
        return BeautifulSoup(content, "html5lib")
    return BeautifulSoup(content.decode("utf-8-sig", "replace"), "html5lib")


def paragraphs(box) -> list[str]:
    out = []
    for p in box.select("p"):
        for br in p.find_all("br"):
            br.replace_with(" ")
        t = re.sub(r"\s+", " ", p.get_text()).strip()
        if t and not re.match(r"For media inquiries", t, re.I):
            out.append(t)
    return out


def norm_para(t: str) -> str:
    t = t.translate(TRANS).replace("\u2014", "--").replace("\u00ad", "")
    for k, val in FRACTIONS.items():
        t = re.sub(rf"(\d)\s*{k}", rf"\1-{val}", t).replace(k, val)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\d) (\d/\d)", r"\1-\2", t)
    t = re.sub(r"\s*Return to text$", "", t).strip()  # footnote back-link
    return t.strip('"').strip()


def policy_body(paras: list[str]) -> str:
    """Statement body without voting roster, release line or page furniture."""
    keep = [norm_para(p) for p in paras]
    return "\n".join(p for p in keep if p and not DROP_PREFIX.match(p))


class Client:
    def __init__(self, raw_dir: Path, delay: float):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.raw = raw_dir
        self.delay = delay
        raw_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, name: str, tries: int = 4):
        last = None
        for k in range(tries):
            try:
                r = self.s.get(url, timeout=60)
                last = r.status_code
                if r.status_code in (429, 503):
                    # Wayback sends no Retry-After; wait a full minute after 429.
                    time.sleep(60 if r.status_code == 429 else self.delay * (2 ** (k + 1)))
                    continue
                if r.ok:
                    (self.raw / name).write_bytes(r.content)
                time.sleep(self.delay)
                return r
            except Exception as e:  # network failure is recorded, not hidden
                last = repr(e)
                time.sleep(self.delay * (2 ** (k + 1)))
        return last


def _blockquote_paras(bq) -> list[str]:
    """Paragraphs of a blockquote that may mix bare text, <br> breaks and <p> tags.
    Source-line wrapping is whitespace; only <br> and <p> boundaries split paragraphs."""
    mark = "\u2029"
    for br in bq.find_all("br"):
        br.replace_with(mark)
    for p in bq.find_all("p"):
        p.insert_before(mark)
        p.insert_after(mark)
    lines = [re.sub(r"\s+", " ", x).strip() for x in bq.get_text().split(mark)]
    lines = [x for x in lines if x and not re.match(r"For media inquiries", x, re.I)]
    merged = []
    for x in lines:  # a <br> inside a sentence is a line break, not a paragraph break
        if merged and not re.search(r"[.:;?!\"\u201d)]$", merged[-1]) and re.match(r"[a-z0-9(]", x):
            merged[-1] += " " + x
        else:
            merged.append(x)
    return merged


def minutes_statement(content: bytes, which: int = 0):
    """Return (release clock, paragraphs, layout) of the statement approved in the minutes.

    Minutes print the approved statement either as quoted <p> paragraphs after
    the release-time sentence or as a <blockquote> with <br> breaks."""
    s = soup(content, declared_charset=True)
    root = s.select_one("#article") or s
    nodes = root.find_all(string=lambda x: x and RELEASE_RE.search(re.sub(r"\s+", " ", x)))
    if len(nodes) <= which:
        return None, [], None
    node = nodes[which]
    m = RELEASE_RE.search(re.sub(r"\s+", " ", node))
    clock = f"{m.group(1)} {m.group(2).lower()}.m."
    nxt = node.find_next(lambda t: getattr(t, "name", None) in ("blockquote", "p"))
    if nxt is not None and nxt.name == "blockquote":
        return clock, _blockquote_paras(nxt), "blockquote"
    block = []
    for q in node.find_all_next("p"):
        for br in q.find_all("br"):
            br.replace_with(" ")
        t = re.sub(r"\s+", " ", q.get_text()).strip()
        if not t:
            continue
        if re.match(r"(Voting|Votes) (for|against) this action", t, re.I):
            break
        block.append(t)
        if len(block) >= 15:  # no voting line: leave the mismatch for review
            break
    return clock, block, "paragraphs"


def alnum(t: str) -> str:
    return re.sub(r"[^0-9a-z]", "", norm_para(t).lower())


def pdf_minutes_statement(content: bytes):
    """Older minutes exist only as PDFs, whose extracted text has line-break
    hyphenation and split words. Return (clock, alphanumeric stream of the
    approved statement); comparison is therefore letters and digits only."""
    import io
    import pypdf
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(content)).pages)
    flat = re.sub(r"-\s*\n\s*", "", text)
    flat = re.sub(r"\s+", " ", flat)
    m = RELEASE_RE.search(flat)
    if not m:
        return None, None
    end = re.search(r"Votes? (for|against) this action|Voting (for|against) this action", flat[m.end():])
    body = flat[m.end():m.end() + end.start()] if end else ""
    return f"{m.group(1)} {m.group(2).lower()}.m.", alnum(body.lstrip(":"))


def clock_to_et(day: str, clock: str) -> dt.datetime:
    hm, ap = clock.split()
    h, mi = map(int, hm.split(":"))
    if ap.startswith("p") and h != 12:
        h += 12
    return dt.datetime.strptime(day, "%Y%m%d").replace(hour=h, minute=mi, tzinfo=ET)


def first_diff(a: str, b: str) -> str | None:
    if a == b:
        return None
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            return f"{tag}: {a[max(0,i1-40):i2+40]!r} -> {b[max(0,j1-40):j2+40]!r}"
    return None


def alt_urls(url: str) -> list[str]:
    """Current and pre-redesign press-release URLs for the same release."""
    bare = re.sub(r"^https?://", "", url)
    m = re.search(r"/newsevents/pressreleases/monetary(\d{8}[a-z]\d?)\.htm$", bare)
    out = [bare]
    if m:
        out.append(f"www.federalreserve.gov/newsevents/press/monetary/{m.group(1)}.htm")
    return out


def wayback(cli: Client, url: str, day: str, release_utc: dt.datetime, selector: str, eid: str):
    nxt = (dt.datetime.strptime(day, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
    rows, queries, failures = [], [], []
    for k, bare in enumerate(alt_urls(url)):
        q = (f"https://web.archive.org/cdx/search/cdx?url={bare}&from={day}&to={nxt}"
             f"&output=json&fl=timestamp,statuscode,digest,original&limit=50")
        queries.append(q)
        r = cli.get(q, f"{eid}.cdx{k}.json")
        if not hasattr(r, "ok") or not r.ok:
            failures.append(getattr(r, "status_code", r))
            continue
        rows += r.json()[1:] if r.text.strip() else []
    if len(failures) == len(queries):
        return {"status": "index_unavailable", "detail": failures, "query": queries}
    q = queries
    caps = [x for x in rows if x[1] == "200"]
    if not caps:
        return {"status": "no_same_day_capture", "query": q, "n_index_rows": len(rows)}
    stamps = [dt.datetime.strptime(x[0], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc) for x in caps]
    before = [s for s in stamps if s < release_utc]
    after = sorted((s, c) for s, c in zip(stamps, caps) if s >= release_utc)
    out = {"query": q, "n_captures": len(caps),
           "n_captures_before_release": len(before),
           "earliest_capture_utc": min(stamps).isoformat()}
    if not after:
        out["status"] = "no_capture_at_or_after_release"
        return out
    ts, cap = after[0]
    replay = f"https://web.archive.org/web/{cap[0]}id_/{cap[3]}"
    rr = cli.get(replay, f"{eid}.wayback-{cap[0]}.html")
    out.update(capture_utc=ts.isoformat(), minutes_after_release=round((ts - release_utc).total_seconds() / 60, 1),
               replay_url=replay, cdx_digest=cap[2])
    if not hasattr(rr, "ok") or not rr.ok:
        out["status"] = "replay_failed"
        out["detail"] = getattr(rr, "status_code", rr)
        return out
    s = soup(rr.content)
    box = None
    for sel in (selector, "#article > div.col-sm-8:not(.heading)", "#leftText", "#article", "#content"):
        if sel and (box := s.select_one(sel)) is not None:
            out["replay_selector"] = sel
            break
    if box is None:
        out["status"] = "replay_container_not_found"
        return out
    out["replay_body"] = policy_body(paragraphs(box))
    out["replay_body_sha256"] = sha(out["replay_body"])
    out["status"] = "replayed"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer-min", type=int, default=5)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--wayback", action="store_true")
    ap.add_argument("--minutes-overrides", help="JSON {event_id: {url, which, reason}} for statements "
                    "recorded in another meeting's minutes (e.g. intermeeting actions)")
    a = ap.parse_args(argv)

    cat = {r["event_id"]: r for r in csv.DictReader(open(a.catalog, encoding="utf-8"))}
    pop = {r["event_id"]: r for r in csv.DictReader(open(a.population, encoding="utf-8"))}
    cli = Client(Path(a.raw_dir), a.delay)
    overrides = json.load(open(a.minutes_overrides, encoding="utf-8")) if a.minutes_overrides else {}
    ledger = []
    for eid in a.events:
        c, p = cat[eid], pop.get(eid)
        if p is None:
            sys.exit(f"{eid} is not in the price-eligible population")
        if p["split"] not in ("train", "val"):
            sys.exit(f"{eid}: held-out split must stay reserved")
        day = c["release_date"]
        origin = dt.datetime.fromisoformat(p["origin_et"]).replace(tzinfo=ET)
        cutoff = origin - dt.timedelta(minutes=a.buffer_min)
        row = {"event_id": eid, "split": p["split"], "catalog_release_et": c["market_time_et"],
               "forecast_origin_et": origin.isoformat(), "text_cutoff_et": cutoff.isoformat(),
               "text_cutoff_utc": cutoff.astimezone(dt.timezone.utc).isoformat(),
               "catalog_text_sha256": c["text_sha256"], "source_url": c["source_url"]}

        r = cli.get(c["source_url"], f"{eid}.current.html")
        cur_body = None
        if hasattr(r, "ok") and r.ok:
            s = soup(r.content)
            box = s.select_one(c["text_selector"])
            if box is not None:
                paras = paragraphs(box)
                row["current_sha256"] = sha("\n".join(paras))
                row["current_matches_catalog"] = row["current_sha256"] == c["text_sha256"]
                cur_body = policy_body(paras)
                row["current_body_sha256"] = sha(cur_body)
        else:
            row["current_status"] = getattr(r, "status_code", r)

        ov = overrides.get(eid, {})
        murl = ov.get("url", f"{FED}/monetarypolicy/fomcminutes{day}.htm")
        if ov:
            row["minutes_override_reason"] = ov.get("reason")
        r = cli.get(murl, f"{eid}.minutes.html")
        row["minutes_url"] = murl
        min_body = None
        if hasattr(r, "ok") and r.ok:
            row["minutes_page_sha256"] = sha(r.content)
            if murl.lower().endswith(".pdf"):
                clock, stream = pdf_minutes_statement(r.content)
                block, layout = [], "pdf_alnum"
                if clock and stream and cur_body:
                    # keep the leading posted paragraphs whose letters/digits equal the PDF record
                    acc, keep = "", []
                    for para in cur_body.split("\n"):
                        if acc == stream:
                            break
                        acc += alnum(para)
                        keep.append(para)
                    block = keep if acc == stream else []
                    row["pdf_alnum_match"] = acc == stream
            else:
                clock, block, layout = minutes_statement(r.content, ov.get("which", 0))
            row["minutes_layout"] = layout
            if clock:
                rel = clock_to_et(day, clock)
                min_body = policy_body(block)
                row.update(minutes_release_clock=clock, minutes_release_et=rel.isoformat(),
                           minutes_body_sha256=sha(min_body), minutes_body_chars=len(min_body),
                           minutes_release_le_cutoff=rel <= cutoff)
            else:
                row["minutes_status"] = "release_phrase_not_found"
        else:
            row["minutes_status"] = getattr(r, "status_code", r)

        if cur_body and min_body:
            row["minutes_matches_current_body"] = cur_body == min_body
            mp, cp = min_body.split("\n"), cur_body.split("\n")
            row["minutes_is_leading_part_of_current"] = cp[:len(mp)] == mp
            row["current_extra_paragraphs"] = cp[len(mp):] if cp[:len(mp)] == mp else None
            row["minutes_vs_current_ratio"] = round(difflib.SequenceMatcher(None, min_body, cur_body, autojunk=False).ratio(), 5)
            row["minutes_vs_current_first_diff"] = first_diff(min_body, cur_body)

        if a.wayback:
            rel_utc = (clock_to_et(day, row["minutes_release_clock"]) if "minutes_release_clock" in row
                       else cutoff).astimezone(dt.timezone.utc)
            w = wayback(cli, c["source_url"], day, rel_utc, c["text_selector"], eid)
            body = w.pop("replay_body", None)
            if body is not None:
                w["matches_minutes_body"] = (body.split("\n")[:len(min_body.split("\n"))] == min_body.split("\n")
                                             if min_body else None)
                w["exactly_equals_minutes_body"] = body == min_body if min_body else None
                w["matches_current_body"] = body == cur_body if cur_body else None
                if min_body and body != min_body:
                    w["first_diff_vs_minutes"] = first_diff(min_body, body)
            row["wayback"] = w

        ok_official = (bool(min_body) and row.get("minutes_release_le_cutoff") is True
                       and row.get("minutes_is_leading_part_of_current") is True)
        w = row.get("wayback") or {}
        if ok_official and w.get("matches_minutes_body"):
            dec = "eligible_A_official_record_plus_near_time_capture"
        elif ok_official:
            dec = "eligible_B_official_record_only"
        elif min_body and row.get("minutes_release_le_cutoff") is False:
            dec = "ineligible_release_after_cutoff"
        elif min_body:
            dec = "review_minutes_text_differs_from_posted_page"
        else:
            dec = "not_verified_no_official_record_found"
        row["fingerprint_sha256"] = row.get("minutes_body_sha256")
        row["fingerprint_source"] = "minutes-approved statement body (normalized)" if min_body else None
        row["eligibility"] = dec
        ledger.append(row)
        print(eid, dec, row.get("minutes_release_clock"), row.get("minutes_matches_current_body"),
              w.get("status"), w.get("minutes_after_release"), w.get("matches_minutes_body"), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "buffer_min": a.buffer_min, "rows": ledger}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
