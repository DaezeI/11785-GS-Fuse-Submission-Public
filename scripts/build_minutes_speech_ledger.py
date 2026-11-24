"""Merge per-slice audit outputs into one sanitized ledger row per assigned event.

Assigned events = population rows of type fomc_minutes and fed_speech (472).
Rows with no audit output are written as not_attempted. Later inputs override
earlier ones (targeted re-runs). The ledger holds hashes, timestamps, URLs and
reasons only; no document text.

    python scripts/build_minutes_speech_ledger.py \
        --population downloads/fed-spx-endpoints-price-v1/population.csv \
        --inputs downloads/minutes-speech-evidence-v1/slices/s*.json \
                 downloads/minutes-speech-evidence-v1/wording-rerun.json \
        --out-dir results/fed-spx-minutes-speeches-evidence-v1
"""
import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATEMENT_FIELDS = ["event_id", "split", "eligibility", "evidence_url", "evidence_sha256", "evidence_release_clock_et",
                    "release_et", "text_cutoff_et", "text_cutoff_utc", "forecast_origin_et", "fingerprint_sha256",
                    "fingerprint_source", "current_page_sha256", "current_matches_catalog",
                    "posted_page_extra_paragraphs", "wayback_capture_utc", "wayback_replay_url",
                    "wayback_matches_fingerprint", "note"]
EXTRA = ["event_type", "status_class", "reason", "evidence_timezone", "pdf_url", "pdf_internal_moddate_utc",
         "pdf_server_written_utc", "pdf_written_minutes_after_release", "pdf_release_line", "wayback_checked",
         "wayback_pre_cutoff_captures", "words_match", "input_file"]
CLASS = {"eligible": "eligible", "version_conflict_review": "version_conflict", "insufficient_evidence": "insufficient_evidence",
         "blocked_lookup": "blocked_lookup", "not_attempted": "not_attempted"}


def iso_et(v):
    return dt.datetime.fromisoformat(str(v)).astimezone(ET).isoformat() if v else ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pop = [r for r in csv.DictReader(open(a.population, encoding="utf-8"))
           if r["event_type"] in ("fomc_minutes", "fed_speech")]
    audited = {}
    for p in a.inputs:
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            r["_file"] = Path(p).name
            audited[r["event_id"]] = r

    out = []
    for p in pop:
        r = audited.get(p["event_id"])
        cutoff_utc = dt.datetime.fromisoformat(p["text_cutoff_utc"])
        base = {"event_id": p["event_id"], "event_type": p["event_type"], "split": p["split"],
                "text_cutoff_utc": cutoff_utc.isoformat(), "text_cutoff_et": cutoff_utc.astimezone(ET).isoformat(),
                "forecast_origin_et": dt.datetime.fromisoformat(p["origin_et"]).replace(tzinfo=ET).isoformat(),
                "evidence_timezone": "America/New_York (ET); UTC fields end +00:00"}
        if r is None:
            base.update(eligibility="not_attempted", status_class="not_attempted",
                        reason="audit stopped before this row; no lookup made")
            out.append(base)
            continue
        disp = r["disposition"]
        is_min = r["event_type"] == "fomc_minutes"
        base.update({
            "eligibility": disp, "status_class": "eligible" if disp.startswith("eligible") else CLASS[disp],
            "reason": r["reason"],
            "evidence_url": r.get("time_evidence_url") if is_min else r.get("pdf_url"),
            "evidence_sha256": r.get("release_record_sha256") if is_min else r.get("pdf_sha256"),
            "evidence_release_clock_et": r.get("release_record_clock") if is_min else (r.get("pdf_release_line") or ""),
            "release_et": iso_et(r.get("release_record_utc") or r.get("release_utc")),
            "fingerprint_sha256": r.get("fingerprint_sha256") or "",
            "fingerprint_source": ("SHA-256 of the released PDF bytes" if r.get("fingerprint_sha256") == r.get("pdf_sha256")
                                   and r.get("fingerprint_sha256") else ("normalized HTML body" if r.get("fingerprint_sha256") else "")),
            "current_page_sha256": r.get("current_sha256"), "current_matches_catalog": r.get("current_matches_catalog"),
            "posted_page_extra_paragraphs": "",
            "wayback_capture_utc": r.get("pre_cutoff_capture_utc") or r.get("wayback_first_after_cutoff_utc") or "",
            "wayback_replay_url": r.get("pre_cutoff_capture_url") or r.get("wayback_first_after_url") or "",
            "wayback_matches_fingerprint": r.get("pre_cutoff_capture_matches", "") or (r.get("pdf_digest_seen_within_24h") or ""),
            "note": "; ".join((r.get("word_compare") or {}).get("examples", [])[:2])[:300] if r.get("words_match") is False else "",
            "pdf_url": r.get("pdf_url"), "pdf_internal_moddate_utc": r.get("pdf_mod_utc"),
            "pdf_server_written_utc": r.get("pdf_last_modified_utc"),
            "pdf_written_minutes_after_release": r.get("pdf_written_minutes_after_release"),
            "pdf_release_line": r.get("pdf_release_line"), "wayback_checked": r.get("wayback_checked"),
            "wayback_pre_cutoff_captures": r.get("wayback_pre_cutoff_captures"), "words_match": r.get("words_match"),
            "input_file": r["_file"]})
        out.append(base)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "minutes_speeches_evidence_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATEMENT_FIELDS + EXTRA, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    counts = {"assigned": len(out),
              "by_status_class": dict(Counter(r["status_class"] for r in out)),
              "by_type_split_status": {f"{t}|{s}|{c}": n for (t, s, c), n in
                                       sorted(Counter((r["event_type"], r["split"], r["status_class"]) for r in out).items())},
              "by_disposition": dict(Counter(r["eligibility"] for r in out))}
    (od / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
