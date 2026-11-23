"""Audit Board release and testimony text against the frozen endpoint cutoff."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from time import sleep
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from pypdf import PdfReader
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_fed_announcements_v2 import _soup, clean_by_selector

POPULATION_SHA = "4061fe505098babf845b2bc0547bdbe11bd732377f6e2a5face383010b024a71"
CATALOG_SHA = "56157fe018beeba6f051f765e0f9d6763e839324b9d95ebbf49f15ec47c9d939"
KINDS = {"fed_board_monetary_release", "fed_testimony"}
AGENT = "CMU-11785-GS-Fuse-research (Fed announcement version audit)"
RELEASE_LINE = re.compile(
    r"For (?:immediate release|release (?:on delivery|at [0-9:.]+\s*[ap]\.?m\.?)(?:\s+E[SD]T)?)",
    re.I,
)
HOUSE_HEARINGS = {
    "testimony_quarles20180417a": ("108192", "Quarles"),
    "testimony_quarles20181114a": ("108762", "Quarles"),
    "testimony_lindo20191120a": ("110249", "Lindo"),
    "testimony_powell20200630a": ("110839", "Powell"),
    "testimony_clark20200908a": ("110976", "Clark"),
    "testimony_powell20200922a": ("111021", "Powell"),
    "testimony_vanderweide20210415a": ("111448", "Weide"),
    "testimony_quarles20210519a": ("112665", "Quarles"),
    "testimony_powell20210622a": ("112798", "Powell"),
}
HOUSE_TIMESTAMP = re.compile(r"(?:Added|Updated) (\d\d/\d\d/\d{4} at \d\d:\d\d [AP]M)")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch(url, **kwargs):
    response = requests.get(url, headers={"User-Agent": AGENT}, timeout=(5, 12), **kwargs)
    response.raise_for_status()
    return response


def official_page(row):
    try:
        response = fetch(row["source_url"])
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else type(exc).__name__
        return {"official_page_status": f"unavailable:{status}"}
    if urlparse(response.url).hostname != "www.federalreserve.gov":
        return {"official_page_status": "unexpected_redirect"}
    soup = _soup(response.content.decode("utf-8-sig", "replace"))
    body = clean_by_selector(soup, row["text_selector"])
    heading = soup.select_one("#article .heading")
    line = RELEASE_LINE.search(heading.get_text(" ", strip=True)) if heading else None
    if not body:
        return {"official_page_status": "no_catalog_body"}
    current_sha = digest(body.encode("utf-8"))
    return {
        "official_page_status": "catalog_match" if current_sha == row["text_sha256"] else "catalog_mismatch",
        "current_page_sha256": current_sha,
        "current_matches_catalog": current_sha == row["text_sha256"],
        "current_release_line": line.group() if line else "",
    }


def historical_record(row, prior):
    statuses = {x["status"] for x in prior}
    log = {
        "prior_archive_statuses": "|".join(sorted(statuses)),
        "eligibility": "not_verified_archive_lookup_incomplete",
    }
    cutoff = re.sub(r"\D", "", row["text_cutoff_utc"])[:14]
    captures = sorted(x["capture_utc"] for x in prior if re.fullmatch(r"\d{14}", x.get("capture_utc", "")))
    if captures:
        log["wayback_capture_utc"] = captures[0]
    if any(stamp <= cutoff for stamp in captures):
        log["eligibility"] = "review_timely_capture_requires_replay"
    elif statuses.intersection({"nearest_capture_after_cutoff", "no_eligible_capture_in_index"}):
        log["eligibility"] = "not_verified_no_timely_capture_in_checked_index"
    return log


def archive_pilot(row, session):
    """Check canonical and pre-redesign Fed URLs without retrying a rate limit."""
    url = row["source_url"]
    legacy = re.sub(r"/pressreleases/monetary(\d)", r"/press/monetary/\1", url)
    cutoff = re.sub(r"\D", "", row["text_cutoff_utc"])[:14]
    attempts = []
    for candidate in dict.fromkeys((url, legacy)):
        sleep(2.5)
        try:
            response = session.get(
                "https://archive.org/wayback/available",
                params={"url": candidate, "timestamp": cutoff},
                timeout=(5, 12),
            )
            response.raise_for_status()
            capture = response.json().get("archived_snapshots", {}).get("closest", {})
            stamp = capture.get("timestamp", "")
            if not capture.get("available") or not re.fullmatch(r"\d{14}", stamp):
                attempts.append({"url": candidate, "status": "no_capture_returned"})
            elif stamp > cutoff:
                attempts.append({"url": candidate, "status": "capture_after_cutoff", "capture_utc": stamp})
            else:
                attempts.append({"url": candidate, "status": "timely_capture_requires_replay", "capture_utc": stamp})
        except (requests.RequestException, ValueError) as exc:
            result = getattr(exc, "response", None)
            status = result.status_code if result is not None else type(exc).__name__
            attempts.append({"url": candidate, "status": f"lookup_unavailable:{status}"})
            if status == 429:
                break
    return attempts


def house_body(pdf_bytes):
    """Extract the body of a verified cover-page testimony PDF."""
    pages = list(PdfReader(BytesIO(pdf_bytes)).pages)
    if len(pages) < 2:
        raise ValueError("Testimony PDF has no body pages")
    cover = re.sub(r"\s+", " ", pages[0].extract_text()).strip()
    body = "\n\n".join(page.extract_text().strip() for page in pages[1:])
    if not body.strip():
        raise ValueError("Empty testimony body")
    return cover, body


def house_evidence(row, hearing, surname, text_dir):
    listing = f"https://docs.house.gov/Committee/Calendar/ByEvent.aspx?EventID={hearing}"
    soup = _soup(fetch(listing).content)
    witness_pattern = re.compile(rf"-Wstate-[^/]*{surname}[^/]*\.pdf$", re.I)
    candidates = [link for link in soup.select("a[href]") if witness_pattern.search(link["href"])]
    if len(candidates) != 1:
        return {"house_status": "ambiguous_or_missing_witness_pdf", "house_listing_url": listing}
    link = candidates[0]
    pdf_url = urljoin(listing, link["href"]).replace("http://docs.house.gov/", "https://docs.house.gov/", 1)
    label = re.sub(r"\s+", " ", link.parent.get_text(" ", strip=True))
    result = {"house_listing_url": listing, "house_document_url": pdf_url, "house_document_label": label}
    if urlparse(pdf_url).hostname != "docs.house.gov" or "/" + row["release_date"] + "/" not in pdf_url:
        return {**result, "house_status": "document_event_date_conflict"}
    timestamps = HOUSE_TIMESTAMP.findall(label)
    if not timestamps:
        return {**result, "house_status": "missing_document_timestamp"}
    latest = max(datetime.strptime(value, "%m/%d/%Y at %I:%M %p") for value in timestamps)
    latest = latest.replace(tzinfo=ZoneInfo("America/New_York"))
    result["house_document_timestamp_et"] = latest.isoformat()
    if latest > datetime.fromisoformat(row["text_cutoff_utc"]):
        return {**result, "house_status": "listed_document_after_cutoff"}
    response = fetch(pdf_url)
    if response.url != pdf_url or not response.content.startswith(b"%PDF-"):
        return {**result, "house_status": "unexpected_document_response"}
    cover, body = house_body(response.content)
    if not re.match(r"For release (?:at|on delivery)\b", cover, re.I) or surname.casefold() not in cover.casefold():
        return {**result, "house_status": "unconfirmed_cover_identity"}
    # Store the congressional copy, which can differ in bytes from the Fed attachment.
    encoded = body.encode("utf-8")
    (text_dir / f'{row["event_id"]}.txt').write_bytes(encoded)
    return {
        **result,
        "house_status": "verified_timestamped_testimony_pdf",
        "eligibility": "eligible_B_official_record_only",
        "evidence_url": listing,
        "evidence_sha256": digest(response.content),
        "evidence_release_clock_et": latest.isoformat(),
        "fingerprint_sha256": digest(encoded),
        "fingerprint_source": "timestamped U.S. House witness PDF body (pypdf, cover excluded)",
        "note": "Committee repository timestamps the specific witness PDF by the frozen text cutoff.",
    }


def write_ledger(path, rows):
    common = list(read_csv(ROOT / "results/fed-spx-text-evidence-v1/fomc_statement_evidence_ledger.csv")[0])
    fields = common + sorted({key for row in rows for key in row} - set(common))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def followup(rows, path, prior):
    by_id = {row["event_id"]: row for row in rows}
    text_dir = ROOT / "downloads/fed-spx-board-testimony-evidence-v1/verified-text"
    text_dir.mkdir(parents=True, exist_ok=True)
    results = read_csv(path)
    for record in results:
        event = record["event_id"]
        record.update(historical_record(by_id[event], prior.get(event, [])))
        if event not in HOUSE_HEARINGS:
            continue
        for key in ("evidence_sha256", "evidence_release_clock_et", "fingerprint_sha256", "fingerprint_source"):
            record[key] = ""
        record["evidence_url"] = by_id[event]["source_url"]
        record["note"] = "Current publisher page and prior archive index checked; no eligible historical body established."
        try:
            record.update(house_evidence(by_id[event], *HOUSE_HEARINGS[event], text_dir))
        except (requests.RequestException, ValueError) as exc:
            record.update(house_status=f"source_unavailable:{type(exc).__name__}")
    # The previous spot audit found this timely CDX candidate. Replay was rate limited.
    for record in results:
        if record["event_id"] == "testimony_powell20210622a":
            record.update(wayback_capture_utc="20210622155409", wayback_replay_url=(
                "https://web.archive.org/web/20210622155409id_/"
                "https://www.federalreserve.gov/newsevents/testimony/powell20210622a.htm"
            ))
    write_ledger(path, results)
    return results


def write_summary(path, results, archive_sha):
    summary = {
        "population_sha256": POPULATION_SHA,
        "catalog_sha256": CATALOG_SHA,
        "prior_archive_ledgers_sha256": archive_sha,
        "groups": {
            f"{kind}/{split}": {
                field: dict(Counter(r.get(field) or "not_applicable" for r in results
                                    if (r["event_type"], r["split"]) == (kind, split)))
                for field in ("eligibility", "official_page_status", "prior_archive_statuses", "house_status")
            }
            for kind in sorted(KINDS) for split in ("train", "val")
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-pilot", action="store_true")
    parser.add_argument("--official-document-pass", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    downloads = ROOT / "downloads"
    population = downloads / "fed-spx-endpoints-price-v1/population.csv"
    catalog = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
    for path, expected in ((population, POPULATION_SHA), (catalog, CATALOG_SHA)):
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    assigned = [r for r in read_csv(population) if r["event_type"] in KINDS]
    if len({r["event_id"] for r in assigned}) != 603 or {r["split"] for r in assigned} != {"train", "val"}:
        raise ValueError("Unexpected train/validation assignment")
    by_id = {r["event_id"]: r for r in read_csv(catalog)}
    prior = {}
    archive_sha = {}
    for directory in ("wayback-text-v1", "wayback-availability-v1"):
        path = downloads / directory / "ledger.jsonl"
        archive_sha[directory] = digest(path.read_bytes())
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prior.setdefault(row["event_id"], []).append(row)
    rows = [{**by_id[r["event_id"]], **r} for r in assigned]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "fed_board_testimony_evidence_ledger.csv"
    if args.official_document_pass:
        results = followup(rows, path, prior)
        write_summary(args.out / "summary.json", results, archive_sha)
        return
    if args.archive_pilot:
        sample = ["board_monetary20080115a", "testimony_powell20210622a"]
        with requests.Session() as session:
            session.headers["User-Agent"] = AGENT
            pilot = {}
            for row in rows:
                if row["event_id"] in sample:
                    pilot[row["event_id"]] = archive_pilot(row, session)
                    if any(x["status"] == "lookup_unavailable:429" for x in pilot[row["event_id"]]):
                        break
        (args.out / "archive-pilot.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
        return
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row, official in zip(rows, pool.map(official_page, rows), strict=True):
            history = historical_record(row, prior.get(row["event_id"], []))
            result = {
                "event_id": row["event_id"], "split": row["split"], "event_type": row["event_type"],
                "evidence_url": row["source_url"], "release_et": row["market_et"],
                "text_cutoff_utc": row["text_cutoff_utc"], "forecast_origin_et": row["origin_et"],
                "note": "Current publisher page and prior archive index checked; no eligible historical body established.",
                **official, **history,
            }
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Publisher pages checked: {len(results)}/{len(rows)}", flush=True)
    write_ledger(path, results)
    write_summary(args.out / "summary.json", results, archive_sha)


if __name__ == "__main__":
    main()
