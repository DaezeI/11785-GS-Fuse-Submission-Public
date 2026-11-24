"""Audit U.S. macro release files against the macro document rule (decided November 23, 2026).

A release is eligible (``eligible_macro_document_rule``) when all hold:

1. the release file states its own embargo/release time, no later than the text cutoff;
2. the PDF's internal modification time is no later than that release time;
3. no archive copy of the file differs from the copy used (absence of copies alone does not fail).

Otherwise: ``version_conflict_review`` (an archive copy differs, or the PDF was modified after
release), ``insufficient_evidence`` (no file, no release line, or no modification time) or
``blocked_lookup`` (a required request failed; rerun later). Resumable: one JSON line per event
in ``--out``; events already there are skipped. Raw files go to ``--raw-dir`` (outside Git).

    python scripts/audit_macro_versions.py --population downloads/macro-spx-v1/population.csv \\
        --source BLS --out downloads/macro-spx-v1/audit-bls.jsonl --raw-dir downloads/macro-spx-v1/raw
"""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import requests

ET, UTC = ZoneInfo("America/New_York"), dt.timezone.utc
AGENT = "CMU-11785-GS-Fuse-research (macro release version audit)"
S = requests.Session()
S.headers["User-Agent"] = AGENT
# "embargoed ... until": 2008-09 BLS releases print the line in two columns, so text extraction puts the
# media-contact phone number between "EMBARGOED" and "UNTIL 8:30 A.M." (up to 80 characters allowed).
RELEASE_LINE = re.compile(
    r"(?:embargoed(?:\s+until|.{0,80}?\buntil\b)|for\s+release(?:\s+at)?|release\s+time|released\s+at|"
    r"not\s+for\s+release\s+until)"
    r".{0,40}?\b(\d{1,2})[:.](\d{2})\s*([ap])\.?\s*m", re.I)


def get(url, delay):
    for attempt in range(3):
        try:
            r = S.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(90)
            continue
        time.sleep(delay)
        return r
    return None


def b32(data):
    return base64.b32encode(hashlib.sha1(data).digest()).decode()


def cdx(url, delay):
    bare = re.sub(r"^https?://", "", url)
    r = get(f"https://web.archive.org/cdx/search/cdx?url={bare}&output=json"
            "&fl=timestamp,statuscode,digest,original&limit=500", delay)
    if r is None or not r.ok:
        return None
    rows = r.json()[1:] if r.text.strip() else []
    return [x for x in rows if x[1] == "200"]


def pdf_facts(data, release_day):
    import pypdf
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        meta = reader.metadata or {}
        text = " ".join((reader.pages[i].extract_text() or "") for i in range(min(2, len(reader.pages))))
    except Exception as exc:  # malformed PDF
        return {"pdf_error": repr(exc)[:120]}
    text = re.sub(r"\s+", " ", text)
    facts = {"pdf_modified_raw": str(meta.get("/ModDate")), "pdf_created_raw": str(meta.get("/CreationDate"))}
    m = RELEASE_LINE.search(text)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        facts["release_line"] = text[max(0, m.start() - 5):m.end() + 40][:140]
        facts["release_line_utc"] = dt.datetime.combine(release_day, dt.time(hour, int(m.group(2))),
                                                        ET).astimezone(UTC).isoformat()
    facts["pdf_modified_utc"] = pdf_time(meta.get("/ModDate"))
    return facts


def pdf_time(raw):
    if not raw:
        return None
    m = re.match(r"D:(\d{14})(?:([+-])(\d\d)'?(\d\d)'?|Z)?", str(raw))
    if not m:
        return None
    clock = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if m.group(2):
        offset = dt.timedelta(hours=int(m.group(3)), minutes=int(m.group(4)))
        clock = (clock - offset) if m.group(2) == "+" else (clock + offset)
        return clock.replace(tzinfo=UTC).isoformat()
    if str(raw).rstrip("'").endswith("Z"):
        return clock.replace(tzinfo=UTC).isoformat()
    return None  # no zone: not comparable


def fetch_bea(row, raw, rec):
    page = get(row["document_hint"], 1.0)
    if page is None or not page.ok:
        rec["blocked"] = f"release page {getattr(page, 'status_code', 'error')}"
        return None, []
    links = re.findall(r'href="([^"]+\.pdf)"', page.text)
    kind = row["product"]
    pdfs = [h for h in links if re.search(rf"(?:^|/)({kind}|gdp|pi)\w*\d\d\w*\.pdf$", h, re.I)
            and not re.search(r"fax|tech|hlt|glance|summary", h, re.I)]
    if not pdfs:
        rec["reason_detail"] = "no release PDF linked from the release page"
        return None, []
    url = "https://www.bea.gov" + pdfs[0] if pdfs[0].startswith("/") else pdfs[0]
    rec["document_url"] = url
    r = get(url, 1.0)
    if r is None or not r.ok or r.content[:4] != b"%PDF":
        rec["blocked"] = f"release PDF {getattr(r, 'status_code', 'error')}"
        return None, []
    fn, year = url.rsplit("/", 1)[1], row["market_et"][:4]
    variants = [url, f"www.bea.gov/newsreleases/national/{kind}/{year}/pdf/{fn}",
                f"www.bea.gov/national/{kind}/{year}/pdf/{fn}"]
    return r.content, variants


def fetch_direct(url, rec):
    r = get(url, 1.0)
    if r is None:
        rec["blocked"] = "download failed"
        return None
    if r.status_code == 404 or (r.ok and r.content[:4] != b"%PDF"):
        rec["reason_detail"] = f"no file at the scheduled address ({r.status_code})"
        return None
    if not r.ok:
        rec["blocked"] = f"download {r.status_code}"
        return None
    rec["server_last_modified"] = r.headers.get("Last-Modified")
    return r.content


def audit(row, raw, wb_delay):
    rec = {k: row[k] for k in ("event_id", "source", "product", "split", "market_et", "text_cutoff_utc",
                               "clock_rule", "document_hint")}
    release_day = dt.date.fromisoformat(row["market_et"][:10])
    release_utc = dt.datetime.fromisoformat(row["market_et"]).replace(tzinfo=ET).astimezone(UTC)
    cutoff = dt.datetime.fromisoformat(row["text_cutoff_utc"])
    data, variants, captures = None, [], []
    if row["source"] == "BLS":
        rec["document_url"] = "https://" + row["document_hint"]
        captures = cdx(row["document_hint"], wb_delay)
        if captures is None:
            rec["blocked"] = "archive index unavailable"
        elif captures:
            r = get(f"https://web.archive.org/web/{captures[-1][0]}id_/{captures[-1][3]}", wb_delay)
            if r is not None and r.ok and r.content[:4] == b"%PDF":
                data = r.content
                rec["copy_used"] = f"archive capture {captures[-1][0]}"
            else:
                rec["blocked"] = f"archive replay {getattr(r, 'status_code', 'error')}"
        else:
            rec["reason_detail"] = "no archive copy of the BLS archive file (bls.gov refuses scripts)"
        variants = []
    elif row["source"] == "BEA":
        data, variants = fetch_bea(row, raw, rec)
    else:
        rec["document_url"] = row["document_hint"]
        data = fetch_direct(row["document_hint"], rec)
        variants = [row["document_hint"]] if data else []
    if data is not None:
        rec["copy_used"] = rec.get("copy_used", "direct download")
        rec["fingerprint_sha256"] = hashlib.sha256(data).hexdigest()
        (raw / f"{row['event_id']}.pdf").write_bytes(data)
        rec.update(pdf_facts(data, release_day))
        digest = b32(data)
        for url in variants:
            found = cdx(url, wb_delay)
            if found is None:
                rec["blocked"] = "archive index unavailable"
                break
            captures += found
        rec["archive_copies"] = len(captures)
        rec["archive_distinct_digests"] = len({c[2] for c in captures} | {digest})
        rec["archive_differs"] = any(c[2] != digest for c in captures)
        if captures:
            rec["archive_first_capture"] = min(c[0] for c in captures)
    rec["disposition"], rec["reason"] = decide(rec, release_utc, cutoff)
    return rec


def decide(rec, release_utc, cutoff):
    if rec.get("blocked"):
        return "blocked_lookup", rec["blocked"]
    if "fingerprint_sha256" not in rec:
        return "insufficient_evidence", rec.get("reason_detail", "no release file")
    if rec.get("pdf_error"):
        return "insufficient_evidence", "release PDF could not be read"
    line = rec.get("release_line_utc")
    if not line:
        return "insufficient_evidence", "release file states no release time"
    if dt.datetime.fromisoformat(line) > cutoff:
        return "insufficient_evidence", "stated release time is after the text cutoff"
    modified = rec.get("pdf_modified_utc")
    if not modified:
        return "insufficient_evidence", "PDF has no zoned modification time"
    if dt.datetime.fromisoformat(modified) > dt.datetime.fromisoformat(line):
        return "version_conflict_review", "PDF modified after its stated release time"
    if rec.get("archive_differs"):
        return "version_conflict_review", "an archive copy differs from the file used"
    return "eligible_macro_document_rule", (
        f"release line at or before cutoff; PDF modified before release; {rec.get('archive_copies', 0)} archive "
        "copies, none different")


def redecide(args):
    """Recompute PDF facts and dispositions from saved files; archive lookups are kept as recorded."""
    lines = [json.loads(x) for x in args.out.read_text(encoding="utf-8").splitlines() if x.strip()]
    latest = {}
    for rec in lines:
        latest[rec["event_id"]] = rec
    for rec in latest.values():
        path = args.raw_dir / f"{rec['event_id']}.pdf"
        if "fingerprint_sha256" in rec and path.exists():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != rec["fingerprint_sha256"]:
                raise ValueError(f"Saved file changed for {rec['event_id']}")
            for key in ("release_line", "release_line_utc", "pdf_error"):
                rec.pop(key, None)
            rec.update(pdf_facts(data, dt.date.fromisoformat(rec["market_et"][:10])))
        release_utc = dt.datetime.fromisoformat(rec["market_et"]).replace(tzinfo=ET).astimezone(UTC)
        rec["disposition"], rec["reason"] = decide(rec, release_utc, dt.datetime.fromisoformat(rec["text_cutoff_utc"]))
    args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in latest.values()), encoding="utf-8")
    from collections import Counter
    print(len(latest), Counter(r["disposition"] for r in latest.values()).most_common())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--source", choices=("BLS", "BEA", "DOL"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--wayback-delay", type=float, default=2.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--redecide", action="store_true",
                        help="re-read the saved PDFs and recompute every decision in --out without network requests")
    args = parser.parse_args()
    if args.redecide:
        redecide(args)
        return
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(args.population.open(encoding="utf-8")) if r["source"] == args.source]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["event_id"] for line in args.out.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("disposition") != "blocked_lookup"}
    todo = [r for r in rows if r["event_id"] not in done][:args.limit]
    with args.out.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(todo, 1):
            rec = audit(row, args.raw_dir, args.wayback_delay)
            handle.write(json.dumps(rec, default=str) + "\n")
            handle.flush()
            print(f"{i}/{len(todo)} {rec['event_id']} {rec['disposition']} | {rec['reason'][:90]}", flush=True)


if __name__ == "__main__":
    main()
