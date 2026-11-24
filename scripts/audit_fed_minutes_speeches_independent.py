"""Audit the frozen minutes and speech population using official Fed records."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from pypdf import PdfReader
from pypdf.errors import PdfReadError
import requests

from audit_fed_board_testimony_versions import (
    AGENT, CATALOG_SHA, POPULATION_SHA, ROOT, _soup, archive_pilot, clean_by_selector, digest, fetch,
    read_csv, write_ledger,
)

TYPES = {"fomc_minutes", "fed_speech"}
INCIDENTS = {
    "min_20130320": "https://oig.federalreserve.gov/reports/board-executive-summary-20130827a.htm",
    "min_20160615": "https://www.federalreserve.gov/newsevents/pressreleases/other20160706a.htm",
}
PILOT_EVENTS = {"min_20080130", "min_20160615", "min_20210728", "speech_powell20210817a"}
CLOCK = re.compile(r"(?:for release(?: on delivery)?(?: at)?|released at)\s+\d{1,2}(?::\d{2})?\s*[ap]\.m\.", re.I)


def fed_pdf(url):
    return urlsplit(url).hostname == "www.federalreserve.gov" and urlsplit(url).path.lower().endswith(".pdf")


def publisher_pdf_link(soup, source):
    stem = Path(urlsplit(source).path).stem
    possible_stems = {stem, re.sub(r"(\d)[a-z]$", r"\1", stem)}
    pdfs = {urljoin(source, link["href"]) for link in soup.select("a[href]")}
    matching = sorted(url for url in pdfs if fed_pdf(url) and Path(urlsplit(url).path).stem in possible_stems)
    return matching[0] if len(matching) == 1 else ""


def pdf_details(url, content=None):
    if content is None:
        content = fetch(url).content
    if not content.startswith(b"%PDF-"):
        return {"publisher_pdf_status": "response_not_pdf"}
    reader = PdfReader(BytesIO(content))
    meta = reader.metadata or {}
    title = re.sub(r"\s+", " ", reader.pages[0].extract_text()[:1500])
    clock = CLOCK.search(title)
    return {
        "publisher_pdf_status": "retrieved",
        "publisher_pdf_url": url,
        "publisher_pdf_sha256": digest(content),
        "publisher_pdf_created_metadata": str(meta.get("/CreationDate", "")),
        "publisher_pdf_modified_metadata": str(meta.get("/ModDate", "")),
        "publisher_pdf_cover_clock": clock.group() if clock else "",
    }


def audit(row):
    source = row["source_url"]
    event = row["event_id"]
    record = {
        "event_id": event, "event_type": row["event_type"], "split": row["split"],
        "release_et": row["market_et"], "forecast_origin_et": row["origin_et"],
        "text_cutoff_utc": row["text_cutoff_utc"], "evidence_url": row["time_evidence_url"],
        "publisher_body_url": source, "eligibility": "not_verified_no_version_specific_publication_proof",
        "note": "The current publisher document, release notice or feed does not establish the released version by the cutoff.",
    }
    try:
        response = fetch(source)
        if urlsplit(response.url).hostname != "www.federalreserve.gov":
            raise ValueError("Unexpected publisher redirect")
        body_sha = ""
        pdf_url = ""
        pdf_bytes = None
        if row["text_format"] == "pdf":
            body_sha = digest(response.content)
            pdf_url, pdf_bytes = source, response.content
        else:
            soup = _soup(response.content.decode("utf-8-sig", "replace"))
            body = clean_by_selector(soup, row["text_selector"])
            if body:
                body_sha = digest(body.encode("utf-8"))
            pdf_url = publisher_pdf_link(soup, source)
        record.update(current_page_sha256=body_sha, current_matches_catalog=body_sha == row["text_sha256"])
        if pdf_url:
            try:
                record.update(pdf_details(pdf_url, pdf_bytes))
            except (requests.RequestException, PdfReadError, ValueError) as exc:
                record["publisher_pdf_status"] = f"unavailable:{type(exc).__name__}"
    except (requests.RequestException, ValueError) as exc:
        record["eligibility"] = "not_verified_publisher_lookup_incomplete"
        record["note"] = f"Publisher lookup failed: {type(exc).__name__}"
    if row["event_type"] == "fomc_minutes":
        try:
            response = fetch(row["time_evidence_url"])
            notice = _soup(response.content.decode("utf-8-sig", "replace"))
            main = notice.select_one("#article") or notice
            clock = CLOCK.search(main.get_text(" ", strip=True))
            record["publisher_release_notice_clock"] = clock.group() if clock else ""
            record["publisher_release_notice_links_body"] = any(
                urlsplit(urljoin(response.url, a["href"])).path == urlsplit(source).path
                for a in main.select("a[href]")
            )
        except requests.RequestException as exc:
            record["publisher_release_notice_status"] = f"unavailable:{type(exc).__name__}"
    if event in INCIDENTS:
        record.update(eligibility="review_official_early_release_incident", evidence_url=INCIDENTS[event],
                      note="An official record documents early disclosure. Do not infer first-public timing from the frozen clock.")
    return record


def recover_minutes_pdfs(records):
    missing = [row for row in records if row["event_type"] == "fomc_minutes" and not row.get("publisher_pdf_status")]
    by_year = {}
    for row in missing:
        year = row["event_id"][4:8]
        if year not in by_year:
            index = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
            try:
                soup = _soup(fetch(index).content)
                by_year[year] = {urljoin(index, a["href"]) for a in soup.select("a[href]")}
            except requests.RequestException:
                by_year[year] = None
        if by_year[year] is None:
            row["publisher_pdf_status"] = "meeting_index_unavailable"
            continue
        stem = Path(urlsplit(row["publisher_body_url"]).path).stem
        matching = [url for url in by_year[year] if fed_pdf(url) and Path(urlsplit(url).path).stem.lower() == stem.lower()]
        if len(matching) == 1:
            try:
                row.update(pdf_details(matching[0]))
            except (requests.RequestException, PdfReadError, ValueError) as exc:
                row["publisher_pdf_status"] = f"unavailable:{type(exc).__name__}"
        else:
            row["publisher_pdf_status"] = "no_unique_pdf_in_fed_meeting_index"
    return records


def modified_vs_cutoff(row):
    raw = row.get("publisher_pdf_modified_metadata", "")
    match = re.fullmatch(r"D:(\d{14})(Z|[+-]\d{2}'?\d{2}'?)", raw)
    if not match:
        return "no_offset_aware_metadata"
    clock, offset = match.groups()
    try:
        timestamp = datetime.strptime(clock + offset.replace("'", ""), "%Y%m%d%H%M%S%z")
    except ValueError:
        return "no_offset_aware_metadata"
    return "after_cutoff" if timestamp > datetime.fromisoformat(row["text_cutoff_utc"]) else "by_cutoff"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive-pilot", action="store_true")
    parser.add_argument("--recover-minutes-pdfs", action="store_true")
    args = parser.parse_args()
    population = ROOT / "downloads/fed-spx-endpoints-price-v1/population.csv"
    catalog = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
    for path, expected in ((population, POPULATION_SHA), (catalog, CATALOG_SHA)):
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    events = [row for row in read_csv(population) if row["event_type"] in TYPES]
    expected_counts = {("fomc_minutes", "train"): 87, ("fomc_minutes", "val"): 17,
                       ("fed_speech", "train"): 273, ("fed_speech", "val"): 95}
    if Counter((r["event_type"], r["split"]) for r in events) != expected_counts:
        raise ValueError("Unexpected frozen cohorts")
    by_id = {row["event_id"]: row for row in read_csv(catalog)}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.archive_pilot:
        pilot = {}
        with requests.Session() as session:
            session.headers["User-Agent"] = AGENT
            for row in events:
                if row["event_id"] not in PILOT_EVENTS:
                    continue
                pilot[row["event_id"]] = archive_pilot({**by_id[row["event_id"]], **row}, session)
                if any(x["status"] == "lookup_unavailable:429" for x in pilot[row["event_id"]]):
                    break
        (args.out / "archive-pilot.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
        return
    ledger = args.out / "codex_minutes_speeches_evidence_ledger.csv"
    if args.recover_minutes_pdfs:
        prior = read_csv(ledger)
        if len(prior) != len(events) or {r["event_id"] for r in prior} != {r["event_id"] for r in events}:
            raise ValueError("Ledger does not cover the frozen assignment")
        records = recover_minutes_pdfs(prior)
    else:
        records = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(audit, ({**by_id[event["event_id"]], **event} for event in events)):
                records.append(result)
                if len(records) % 100 == 0:
                    print(f"Independent publisher audit: {len(records)}/{len(events)}", flush=True)
    for record in records:
        record["publisher_pdf_modified_vs_cutoff"] = modified_vs_cutoff(record)
    write_ledger(ledger, records)
    summary = {"population_sha256": POPULATION_SHA, "catalog_sha256": CATALOG_SHA, "groups": {}}
    for kind, split in expected_counts:
        rows = [r for r in records if (r["event_type"], r["split"]) == (kind, split)]
        summary["groups"][f"{kind}/{split}"] = {
            key: dict(Counter(r.get(key) or "not_checked" for r in rows))
            for key in ("eligibility", "current_matches_catalog", "publisher_pdf_status", "publisher_pdf_modified_vs_cutoff")
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
