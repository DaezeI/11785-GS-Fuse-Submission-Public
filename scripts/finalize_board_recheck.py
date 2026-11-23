"""Finalize the Board release archive re-check into a sanitized ledger.

For version conflicts, compare today's paragraphs with the archived release-day
paragraphs. If the only differences are paragraphs added to today's page that are
link labels (term sheets, FAQs), the release-day body is the verified historical
version: fingerprint it and record the added lines. Otherwise keep the conflict.
"""
import csv, json, re, sys, time
from collections import Counter
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_fomc_statement_versions import norm_para, paragraphs, sha, soup  # noqa: E402

LINK = re.compile(r"^(Updated )?(Term Sheet|Terms? and Conditions|FAQs?|Frequently Asked Questions)\b", re.I)
S = requests.Session(); S.headers["User-Agent"] = "CMU-11785-GS-Fuse-research (Board release version audit)"


def paras(content, sel):
    box = soup(content).select_one(sel)
    return [norm_para(p) for p in paragraphs(box)] if box is not None else None


def main():
    shards, catalog, out_dir = sys.argv[1:3], sys.argv[3], Path(sys.argv[4])
    cat = {r["event_id"]: r for r in csv.DictReader(open(catalog, encoding="utf-8"))}
    rows = [json.loads(l) for f in shards for l in open(f, encoding="utf-8") if l.strip()]
    for r in rows:
        c = cat[r["event_id"]]
        if r["disposition"] == "version_conflict_review":
            sel = r.get("replay_selector") or c["text_selector"]
            now = paras(S.get(c["source_url"], timeout=60).content, c["text_selector"]); time.sleep(1)
            old = paras(S.get(r["replay_url"], timeout=60).content, sel); time.sleep(2)
            added = [p for p in now if p not in old] if now and old else None
            removed = [p for p in old if p not in now] if now and old else None
            r["added_since_release"] = added
            if added is not None and not removed and all(LINK.match(p) for p in added) and r.get("release_line"):
                r["disposition"] = "eligible_document_release_line_release_day_capture"
                r["reason"] = (f"page says 'For release at {r['release_line']}'; release-day copy verified; "
                               f"today's page adds {len(added)} link line(s) after release, so the release-day body is used")
                r["fingerprint_sha256"] = sha("\n".join(old)); r["fingerprint_source"] = "release-day archived body (normalized)"
            else:
                r["reason"] += "; removed or non-link differences present" if removed else ""
        elif r["disposition"].startswith("eligible"):
            sel = r.get("replay_selector") or c["text_selector"]
            old = paras(S.get(r["replay_url"], timeout=60).content, sel); time.sleep(2)
            r["fingerprint_sha256"] = sha("\n".join(old)) if old else None
            r["fingerprint_source"] = "release-day archived body (normalized)"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "split", "eligibility", "reason", "release_line", "text_cutoff_utc", "source_url",
              "capture_utc", "capture_minutes_after_cutoff", "replay_url", "fingerprint_sha256", "fingerprint_source",
              "captures_in_window", "added_since_release"]
    with open(out_dir / "board_recheck_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in sorted(rows, key=lambda x: x["event_id"]):
            r["eligibility"] = r["disposition"]
            r["added_since_release"] = " | ".join(r.get("added_since_release") or [])
            w.writerow(r)
    counts = Counter((r["split"], r["disposition"]) for r in rows)
    (out_dir / "counts.json").write_text(json.dumps({f"{a}|{b}": n for (a, b), n in sorted(counts.items())}, indent=2))
    print(json.dumps({f"{a}|{b}": n for (a, b), n in sorted(counts.items())}, indent=1))


if __name__ == "__main__":
    main()
