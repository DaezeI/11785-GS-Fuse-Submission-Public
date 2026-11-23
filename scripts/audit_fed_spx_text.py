#!/usr/bin/env python3
"""Check exact Fed release versions against dated historical captures."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
from threading import Event

import pandas as pd
from pypdf.errors import PdfReadError
import requests

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from data.fed_spx_endpoints import CATALOG, INPUT_SHA256
from scripts.build_fed_announcements_v2 import _soup, clean_by_selector, pdf_to_text
from scripts.build_fed_price_alignment import sha256_file

ARCHIVE = "https://web.archive.org"
TIMEOUT = (5, 12)


def audit_event(row, text_dir, rate_limited, index):
    cutoff = pd.Timestamp(row.text_cutoff_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S")
    result = {"event_id": row.event_id, "split": row.split, "event_type": row.event_type,
              "source_url": row.source_url, "cutoff_utc": cutoff}
    if rate_limited.is_set():
        return {**result, "status": "not_queried_after_rate_limit"}
    try:
        if index == "cdx":
            response = requests.get(ARCHIVE + "/cdx/search/cdx", timeout=TIMEOUT, params={
                "url": row.source_url, "output": "json", "fl": "timestamp,original",
                "filter": "statuscode:200", "to": cutoff, "limit": "-1",
            })
            response.raise_for_status()
            captures = response.json() if response.content.strip() else []
            if len(captures) < 2:
                return {**result, "status": "no_eligible_capture_in_index"}
            stamp, original = captures[-1]
        else:
            response = requests.get("https://archive.org/wayback/available", timeout=TIMEOUT,
                                    params={"url": row.source_url, "timestamp": cutoff})
            response.raise_for_status()
            closest = response.json().get("archived_snapshots", {}).get("closest", {})
            if not closest.get("available") or closest.get("status") != "200":
                return {**result, "status": "no_nearest_capture_returned"}
            stamp, original = closest.get("timestamp", ""), row.source_url
            if stamp > cutoff:
                return {**result, "status": "nearest_capture_after_cutoff", "capture_utc": stamp}
        if not re.fullmatch(r"\d{14}", stamp) or stamp > cutoff:
            return {**result, "status": "capture_timestamp_invalid", "capture_utc": stamp}
        replay = f"{ARCHIVE}/web/{stamp}id_/{original}"
        result.update(capture_utc=stamp, replay_url=replay)
        response = requests.get(replay, timeout=TIMEOUT)
        response.raise_for_status()
        if not re.match(rf"{re.escape(ARCHIVE)}/web/{stamp}(?:id_)?/", response.url):
            return {**result, "status": "replay_redirected_to_different_version"}
        if row.text_format == "pdf":
            matches = hashlib.sha256(response.content).hexdigest() == row.text_sha256
            content = pdf_to_text(response.content) if matches else None
        else:
            content = clean_by_selector(_soup(response.content.decode("utf-8-sig", "replace")), row.text_selector)
            matches = content is not None and hashlib.sha256(content.encode("utf-8")).hexdigest() == row.text_sha256
        if not matches:
            return {**result, "status": "exact_version_hash_mismatch"}
        if not content or not content.strip():
            return {**result, "status": "empty_text_extraction"}
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        (text_dir / f"{row.event_id}.txt").write_bytes(content.encode("utf-8"))
        return {**result, "status": "verified_exact_version", "model_text_sha256": digest}
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            rate_limited.set()
        phase = "replay" if "capture_utc" in result else "index"
        return {**result, "status": f"{phase}_unavailable", "error_type": type(exc).__name__, "http_status": status}
    except (PdfReadError, TypeError, ValueError) as exc:
        return {**result, "status": "archive_content_unreadable", "error_type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--index", choices=("cdx", "availability"), default="cdx")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    if sha256_file(CATALOG) != INPUT_SHA256["catalog"]:
        raise ValueError("Catalog SHA-256 mismatch")
    population = pd.read_csv(args.population, dtype=str)
    if not set(population.split) <= {"train", "val"} or not population.event_id.is_unique:
        raise ValueError("Expected distinct train/validation event IDs only")
    if not population.event_id.str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Invalid event ID")
    catalog = pd.read_csv(CATALOG, usecols=["event_id", "source_url", "text_format", "text_selector", "text_sha256"])
    rows = population.merge(catalog, on="event_id", validate="one_to_one")
    if len(rows) != len(population) or not rows.text_sha256.str.fullmatch(r"[a-f0-9]{64}").all():
        raise ValueError("Missing catalog artifact identity")
    text_dir = args.out_dir / "verified-text"
    text_dir.mkdir(parents=True, exist_ok=False)
    rate_limited = Event()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.out_dir / "ledger.jsonl").open(
        "w", encoding="utf-8", buffering=1,
    ) as ledger:
        results = []
        for row in pool.map(lambda row: audit_event(row, text_dir, rate_limited, args.index), rows.itertuples(index=False)):
            results.append(row)
            ledger.write(json.dumps(row) + "\n")
            if len(results) % 100 == 0:
                print(f"Archived text rows checked: {len(results)}/{len(rows)}", flush=True)
    summary = {"population_sha256": sha256_file(args.population), "catalog_sha256": INPUT_SHA256["catalog"],
               "index": args.index,
               "method": "CDX as-of capture or nearest Availability capture; cutoff and exact replay hash required",
               "splits": {split: dict(Counter(r["status"] for r in results if r["split"] == split))
                          for split in ("train", "val")}}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
