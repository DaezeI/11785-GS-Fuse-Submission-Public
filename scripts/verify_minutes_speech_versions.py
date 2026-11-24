"""Exact-version text evidence for FOMC minutes and Fed speech events.

One ledger row per event in the frozen endpoint population. For each event:

1. today's catalog page: extract with the catalog selector, check the catalog
   SHA-256 and build the normalized body;
2. the official document file (PDF) linked from that page or the release
   record: SHA-256 and SHA-1 (base32, comparable with Wayback CDX digests) of
   its bytes, HTTP Last-Modified, internal CreationDate/ModDate, and a word-level
   comparison of the PDF text with the HTML body;
3. minutes only: the official release record (press release named in the
   catalog's time_evidence_url) must say "For release at H:MM" and link this
   minutes document;
4. Wayback CDX for the HTML page and the PDF from seven days before the cutoff
   through the day after: any capture at or before the cutoff, the earliest after
   it, and whether any capture's digest equals today's PDF.

Dispositions are assigned by ``decide`` (see the docstring there). Raw pages,
PDFs and CDX responses go to --raw-dir, which must stay outside Git.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_fomc_statement_versions import Client, norm_para, paragraphs, sha, soup  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
FED = "https://www.federalreserve.gov"
FILE_WINDOW = dt.timedelta(minutes=10)  # server write within this window after release = not rewritten since


HEADER_LINE = re.compile(
    r"^\s*(Page\s+\d+(\s+of\s+\d+)?(\s+Federal Open Market Committee)?|Minutes of the Meeting of .*|"
    r"\d{1,3}|Federal Open Market Committee)\s*$", re.I)


WEB_ONLY = re.compile(r"^\s*(Watch (live|the|video)|Accessible Keys for Video|View (the )?(video|webcast)|"
                      r"Video\b|Return to top|Related (Information|Content)|Last Update|Back to Top|\[[^\]]+\]|"
                      r"Accessible Version|.*\$\$|"
                      r".*(toggles play/pause|seeks the video|toggles captions|volume (up|down)))", re.I)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", norm_para(text).lower())


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,/]\d+)*", norm_para(text))


def clean_pdf_text(text: str) -> str:
    lines = [x for x in text.split("\n") if not HEADER_LINE.match(x)]
    t = "\n".join(lines)
    return re.sub(r"-\s*\n\s*", "", t)


def pdf_info(content: bytes):
    import pypdf
    r = pypdf.PdfReader(io.BytesIO(content))
    meta = r.metadata or {}
    text = "\n".join(p.extract_text() or "" for p in r.pages)

    def pdate(v):
        m = re.match(r"D:(\d{14})(Z|[+-]\d{2}'?\d{2}'?)?", str(v or ""))
        if not m:
            return None
        base = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        tz = m.group(2)
        if not tz:
            return None  # no zone: cannot place on the clock
        if tz == "Z":
            return base.replace(tzinfo=UTC)
        sign = 1 if tz[0] == "+" else -1
        hh, mm = int(tz[1:3]), int(tz.replace("'", "")[3:5] or 0)
        return base.replace(tzinfo=dt.timezone(sign * dt.timedelta(hours=hh, minutes=mm)))

    first = re.sub(r"\s+", " ", (r.pages[0].extract_text() or "") if r.pages else "")
    m = re.search(r"For release on delivery\s*([0-9]{1,2}:[0-9]{2})\s*([ap])\.\s*m\.\s*(E[SD]T)\s*"
                  r"([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", first)
    release_line = None
    if m:
        try:
            when = dt.datetime.strptime(f"{m.group(4).replace(' ', ' ')} {m.group(1)} {m.group(2).upper()}M",
                                        "%B %d, %Y %I:%M %p").replace(tzinfo=ET)
            release_line = {"text": m.group(0)[:80], "utc": when.astimezone(UTC)}
        except ValueError:
            release_line = {"text": m.group(0)[:80], "utc": None}
    return {"pdf_creation": pdate(meta.get("/CreationDate")), "pdf_mod": pdate(meta.get("/ModDate")),
            "pdf_producer": str(meta.get("/Producer") or ""), "text": text, "pages": len(r.pages),
            "release_line": release_line}


def compare_words(html_paras: list[str], pdf_text: str):
    """Wording check that tolerates PDF extraction artefacts.

    Letters: every HTML paragraph's letter stream must appear in the PDF letter
    stream, either verbatim or inside a nearby window with only PDF-side
    insertions (page furniture, footnotes placed at page bottoms). Numbers: every
    number in the HTML (footnote markers removed) must occur at least as often in
    the PDF. Spacing, hyphenation and footnote-marker digits are ignored."""
    from collections import Counter
    pdf = clean_pdf_text(pdf_text)
    stream = letters(pdf)
    bad, checked, skipped = [], 0, []
    for para in html_paras:
        para = re.sub(r"Return to (text|top)|\((PDF|HTML)\)", " ", para, flags=re.I)
        if WEB_ONLY.match(para):
            skipped.append(para[:80])
            continue
        pl = letters(para)
        if len(pl) < 8:
            continue
        checked += 1
        if pl in stream:
            continue
        pos = stream.find(pl[:60])
        if pos < 0:
            pos = stream.find(pl[-60:])
            pos = max(0, pos - len(pl)) if pos >= 0 else -1
        if pos < 0:
            bad.append(("not_found", para[:120]))
            continue
        win = stream[pos:pos + int(len(pl) * 1.6) + 4000]
        sm = difflib.SequenceMatcher(None, pl, win, autojunk=False)
        miss = len(pl) - sum(b.size for b in sm.get_matching_blocks())
        if miss:
            bad.append((f"{miss}_letters_unmatched", para[:120]))
    html_text = "\n".join(x for x in html_paras if not WEB_ONLY.match(x))
    pdf_nums = []
    for n in numbers(pdf):  # "2013.7" can be 2013 followed by footnote marker 7
        pdf_nums.append(n)
        if "." in n:
            pdf_nums.append(n.split(".")[0])
    hn, pn = Counter(numbers(html_text)), Counter(pdf_nums)
    num_missing = {k: v - pn.get(k, 0) for k, v in hn.items() if pn.get(k, 0) < v}
    return {"html_paragraphs_checked": checked, "paragraphs_not_matched": len(bad),
            "html_web_only_paragraphs_skipped": skipped,
            "numbers_missing_in_pdf": num_missing, "examples": [f"{a}: {b}" for a, b in bad[:5]]}


def b32sha1(content: bytes) -> str:
    return base64.b32encode(hashlib.sha1(content).digest()).decode()


def cdx(cli: Client, url: str, start: dt.datetime, end: dt.datetime, name: str):
    bare = re.sub(r"^https?://", "", url)
    q = (f"https://web.archive.org/cdx/search/cdx?url={bare}&from={start:%Y%m%d}&to={end:%Y%m%d}"
         f"&output=json&fl=timestamp,statuscode,digest,original&limit=200")
    r = cli.get(q, name)
    if not hasattr(r, "ok") or not r.ok:
        return None, q, getattr(r, "status_code", r)
    rows = r.json()[1:] if r.text.strip() else []
    caps = [(dt.datetime.strptime(x[0], "%Y%m%d%H%M%S").replace(tzinfo=UTC), x[2], x[3]) for x in rows if x[1] == "200"]
    return caps, q, None


def legacy(url: str) -> list[str]:
    out = [url]
    m = re.search(r"/newsevents/pressreleases/monetary(\d{8}[a-z]\d?)\.htm$", url)
    if m:
        out.append(f"{FED}/newsevents/press/monetary/{m.group(1)}.htm")
    return out


def decide(row: dict) -> tuple[str, str]:
    """Disposition and reason.

    eligible_official_release_record_unmodified_file (minutes): the release record
      says "For release at" a time no later than the cutoff and links this document;
      the PDF's internal ModDate precedes the release; the PDF was written to the
      server within ten minutes after the release (not rewritten since) or its bytes
      equal a Wayback capture made within 24 hours after release; every HTML word
      appears in order in the PDF.
    eligible_archived_copy_before_cutoff: a Wayback capture taken at or before the
      cutoff whose digest equals today's PDF, or whose replayed HTML body equals
      today's body.
    version_conflict_review: verified historical document wording differs from
      today's HTML wording, or the PDF was modified after release and no
      pre-release copy matches.
    insufficient_evidence: no record ties a specific text version to availability
      by the cutoff (e.g. speeches with only a scheduled delivery time).
    blocked_lookup: a required lookup failed; the event was not fully checked.
    """
    if row.get("blocked"):
        return "blocked_lookup", "; ".join(row["blocked"])
    if row.get("pre_cutoff_capture_matches"):
        if row.get("words_match") is False:
            return "version_conflict_review", "pre-cutoff archived file differs in wording from today's HTML"
        return "eligible_archived_copy_before_cutoff", f"capture {row['pre_cutoff_capture_utc']} <= cutoff"
    if row["event_type"] == "fomc_minutes":
        if not row.get("release_record_ok"):
            return "insufficient_evidence", "official release record missing, has no time, or does not link this document"
        if not row.get("pdf_sha256"):
            return "insufficient_evidence", "no official document file to tie to the release"
        mod, rel = row.get("pdf_mod_utc"), row.get("release_utc")
        if not mod:
            return "insufficient_evidence", "PDF has no zoned internal modification time"
        if mod > rel:
            return "version_conflict_review", "PDF internal ModDate is after the release; released version not established"
        unmodified = row.get("pdf_written_minutes_after_release") is not None and \
            0 <= row["pdf_written_minutes_after_release"] <= FILE_WINDOW.total_seconds() / 60
        if not (unmodified or row.get("pdf_digest_seen_within_24h")):
            return "insufficient_evidence", "PDF rewritten after release and no archived copy within 24 h shows the same bytes"
        if row.get("words_match") is False:
            return "version_conflict_review", "released PDF wording differs from today's HTML wording"
        if row.get("words_match") is not True:
            return "insufficient_evidence", "released PDF found but its wording could not be compared with the HTML"
        return "eligible_official_release_record_unmodified_file", (
            "release record time and link; PDF ModDate before release; "
            + ("server write at release" if unmodified else "same bytes archived within 24 h"))
    if row.get("pdf_release_line_utc") is not None:
        line, mod = row["pdf_release_line_utc"], row.get("pdf_mod_utc")
        unmodified = row.get("pdf_written_minutes_after_release") is not None and \
            0 <= row["pdf_written_minutes_after_release"] <= FILE_WINDOW.total_seconds() / 60
        if line > row["cutoff_utc"]:
            return "insufficient_evidence", "document release line is later than the text cutoff"
        if mod and mod <= line and (unmodified or row.get("pdf_digest_seen_within_24h")):
            if row.get("words_match") is False:
                return "version_conflict_review", "released PDF wording differs from today's HTML wording"
            if row.get("words_match") is not True:
                return "insufficient_evidence", "released PDF found but its wording could not be compared with the HTML"
            return "eligible_document_release_line_unmodified_file", (
                "PDF says 'For release on delivery' at a time no later than the cutoff; ModDate before it; "
                + ("server write at release" if unmodified else "same bytes archived within 24 h"))
        if mod and mod > line:
            return "version_conflict_review", "PDF internal ModDate is after its release line; released version not established"
        return "insufficient_evidence", ("PDF carries a release line and pre-release ModDate, but it was rewritten "
                                         "after release and no archived copy within 24 h shows the same bytes")
    if row.get("pdf_sha256") and row.get("pdf_mod_utc") and row["pdf_mod_utc"] <= row["cutoff_utc"]:
        return "insufficient_evidence", ("only a scheduled delivery time; PDF prepared before the cutoff"
                                         + (" and written to the server at delivery" if row.get("pdf_written_minutes_after_release") is not None
                                            and abs(row["pdf_written_minutes_after_release"]) <= 10 else "")
                                         + ", but no record shows the text public by the cutoff")
    return "insufficient_evidence", "only a scheduled delivery time; no archived copy at or before the cutoff"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--types", nargs="+", default=["fomc_minutes", "fed_speech"])
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between Wayback requests")
    ap.add_argument("--fed-delay", type=float, default=1.0, help="seconds between federalreserve.gov requests")
    ap.add_argument("--no-wayback", action="store_true")
    ap.add_argument("--allow-test", action="store_true",
                    help="version-check held-out test text (reads no prices); use only with the test population")
    a = ap.parse_args(argv)

    cat = {r["event_id"]: r for r in csv.DictReader(open(a.catalog, encoding="utf-8"))}
    pop = [r for r in csv.DictReader(open(a.population, encoding="utf-8")) if r["event_type"] in a.types]
    if a.events:
        pop = [r for r in pop if r["event_id"] in set(a.events)]
    pop = pop[a.start:a.start + a.count if a.count else None]
    cli = Client(Path(a.raw_dir), a.fed_delay)
    wb = Client(Path(a.raw_dir), a.delay)
    raw = Path(a.raw_dir)
    out = []
    for p in pop:
        if p["split"] not in ("train", "val") + (("test",) if a.allow_test else ()):
            sys.exit(f"{p['event_id']}: held-out split must stay reserved")
        eid, c = p["event_id"], cat[p["event_id"]]
        cutoff = dt.datetime.fromisoformat(p["text_cutoff_utc"]).astimezone(UTC)
        release = dt.datetime.strptime(c["market_time_et"].replace("a.m.", "AM").replace("p.m.", "PM"),
                                       "%Y-%m-%d %I:%M %p").replace(tzinfo=ET).astimezone(UTC)
        row = {"event_id": eid, "event_type": p["event_type"], "split": p["split"], "cutoff_utc": cutoff,
               "release_utc": release, "source_url": c["source_url"], "catalog_text_sha256": c["text_sha256"],
               "time_evidence": c["time_evidence"], "time_evidence_url": c["time_evidence_url"], "blocked": []}

        body, pdf_links, compare_paras = None, [], []
        is_pdf_source = c["text_format"] == "pdf"
        r = None if is_pdf_source else cli.get(c["source_url"], f"{eid}.current.html")
        if is_pdf_source:
            pass  # the catalog text is the PDF itself; handled with the PDF below
        elif hasattr(r, "ok") and r.ok:
            s = soup(r.content)
            box = s.select_one(c["text_selector"])
            if box is not None:
                paras = paragraphs(box)
                row["current_sha256"] = sha("\n".join(paras))
                row["current_matches_catalog"] = row["current_sha256"] == c["text_sha256"]
                body = "\n".join(norm_para(x) for x in paras)
                row["current_body_sha256"] = sha(body)
                for sup in box.select("sup"):
                    sup.decompose()
                compare_paras = paragraphs(box)
            else:
                row["blocked"].append("catalog selector not found on current page")
            pdf_links = [x.get("href") for x in s.select("a[href]") if x.get("href", "").lower().endswith(".pdf")]
        else:
            row["blocked"].append(f"current page {getattr(r, 'status_code', r)}")

        stem = Path(c["source_url"]).stem
        pdf_url = c["source_url"] if is_pdf_source else next((h for h in pdf_links if stem in h), None)
        if pdf_url and pdf_url.startswith("/"):
            pdf_url = FED + pdf_url

        if p["event_type"] == "fomc_minutes":
            rr = cli.get(c["time_evidence_url"], f"{eid}.release.html")
            if hasattr(rr, "ok") and rr.ok:
                txt = rr.content.decode("utf-8-sig", "replace")
                m = re.search(r"For release at\s*([0-9:]+\s*[ap]\.m\.)\s*(E[SD]T)", txt)
                row["release_record_sha256"] = sha(rr.content)
                row["release_record_clock"] = " ".join(m.groups()) if m else None
                links_doc = stem in txt
                rel_pdf = re.findall(r'href="([^"]*' + re.escape(stem) + r'[^"]*\.pdf)"', txt)
                if not pdf_url and rel_pdf:
                    pdf_url = rel_pdf[0] if rel_pdf[0].startswith("http") else FED + rel_pdf[0]
                if m:
                    hm, ap_ = m.group(1).replace(" ", "")[:-4], m.group(1).strip()[-4]
                    h, mi = map(int, hm.split(":"))
                    h += 12 if ap_ == "p" and h != 12 else 0
                    rec = release.astimezone(ET).replace(hour=h, minute=mi).astimezone(UTC)
                    row["release_record_utc"] = rec
                    row["release_record_ok"] = links_doc and rec <= cutoff
                else:
                    row["release_record_ok"] = False
            else:
                row["blocked"].append(f"release record {getattr(rr, 'status_code', rr)}")

        row["pdf_url"] = pdf_url
        pdf_bytes = None
        if pdf_url:
            rp = cli.get(pdf_url, f"{eid}.pdf")
            if hasattr(rp, "ok") and rp.ok and rp.content[:4] == b"%PDF":
                pdf_bytes = rp.content
                lm = rp.headers.get("Last-Modified")
                row["pdf_sha256"] = sha(pdf_bytes)
                row["pdf_sha1_b32"] = b32sha1(pdf_bytes)
                if is_pdf_source:
                    row["current_sha256"] = row["pdf_sha256"]
                    row["current_matches_catalog"] = row["pdf_sha256"] == c["text_sha256"]
                    row["words_match"] = True
                    row["word_compare"] = {"note": "catalog text is this PDF; no HTML version to compare"}
                row["pdf_last_modified_utc"] = (dt.datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=UTC)
                                                if lm else None)
                try:
                    info = pdf_info(pdf_bytes)
                    row["pdf_creation_utc"] = info["pdf_creation"].astimezone(UTC) if info["pdf_creation"] else None
                    row["pdf_mod_utc"] = info["pdf_mod"].astimezone(UTC) if info["pdf_mod"] else None
                    row["pdf_pages"] = info["pages"]
                    if info["release_line"]:
                        row["pdf_release_line"] = info["release_line"]["text"]
                        row["pdf_release_line_utc"] = info["release_line"]["utc"]
                    if body:
                        cmp_ = compare_words(compare_paras, info["text"])
                        row["word_compare"] = cmp_
                        row["words_match"] = cmp_["paragraphs_not_matched"] == 0 and not cmp_["numbers_missing_in_pdf"]
                except Exception as e:  # unreadable PDF is recorded
                    row["blocked"].append(f"pdf parse {e!r}"[:200])
                if row.get("pdf_last_modified_utc"):
                    row["pdf_written_minutes_after_release"] = round(
                        (row["pdf_last_modified_utc"] - release).total_seconds() / 60, 2)
            elif not (hasattr(rp, "ok") and rp.status_code == 404):
                row["blocked"].append(f"pdf {getattr(rp, 'status_code', rp)}")

        row["disposition"], row["reason"] = decide(row)
        # Archive lookups can only add evidence; skip them when the row is already eligible.
        row["wayback_checked"] = False
        if not a.no_wayback and not row["disposition"].startswith("eligible"):
            row["wayback_checked"] = True
            start, end = cutoff - dt.timedelta(days=7), cutoff + dt.timedelta(days=1)
            targets = legacy(c["source_url"]) + ([pdf_url] if pdf_url else [])
            pre, after, seen24 = [], [], False
            for k, u in enumerate(targets):
                caps, q, err = cdx(wb, u, start, end, f"{eid}.cdx{k}.json")
                if err is not None:
                    row["blocked"].append(f"wayback index {err}")
                    continue
                for ts, digest, orig in caps:
                    is_pdf = orig.lower().endswith(".pdf")
                    same = bool(pdf_bytes) and is_pdf and digest == row.get("pdf_sha1_b32")
                    if ts <= cutoff:
                        pre.append((ts, orig, digest, same))
                    else:
                        after.append((ts, orig, digest, same))
                        if same and ts - release <= dt.timedelta(hours=24):
                            seen24 = True
            row["wayback_pre_cutoff_captures"] = len(pre)
            row["pdf_digest_seen_within_24h"] = seen24
            if after:
                first = min(after)
                row["wayback_first_after_cutoff_utc"] = first[0]
                row["wayback_first_after_url"] = first[1]
            match = [x for x in pre if x[3]]
            if pre and not match:
                # replay HTML captures before the cutoff and compare bodies
                for ts, orig, digest, _ in sorted(pre, reverse=True)[:2]:
                    if orig.lower().endswith(".pdf"):
                        continue
                    rr = wb.get(f"https://web.archive.org/web/{ts:%Y%m%d%H%M%S}id_/{orig}", f"{eid}.pre-{ts:%Y%m%d%H%M%S}.html")
                    if hasattr(rr, "ok") and rr.ok:
                        bx = soup(rr.content).select_one(c["text_selector"]) or soup(rr.content).select_one("#leftText")
                        if bx is not None and body and "\n".join(norm_para(x) for x in paragraphs(bx)) == body:
                            match.append((ts, orig, digest, True))
                            break
            if match:
                row["pre_cutoff_capture_matches"] = True
                row["pre_cutoff_capture_utc"] = min(match)[0]
                row["pre_cutoff_capture_url"] = min(match)[1]

        row["disposition"], row["reason"] = decide(row)
        row["fingerprint_sha256"] = (row.get("pdf_sha256") if row["disposition"].startswith(("eligible", "version_conflict"))
                                     and row.get("pdf_sha256") else (row.get("current_body_sha256")
                                     if row["disposition"] == "eligible_archived_copy_before_cutoff" else None))
        out.append(row)
        print(eid, p["split"], row["disposition"], "|", row["reason"][:90], flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"generated_utc": dt.datetime.now(UTC), "rows": out}, indent=1, default=str),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
