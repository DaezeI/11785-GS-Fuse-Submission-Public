# Board release archive re-check

Codex's Board and testimony audit (`results/fed-spx-board-testimony-evidence-v1/`) left 370 Board
monetary releases as `not_verified_archive_lookup_incomplete`. Those rows carried earlier failed or
rate-limited lookups forward and were never re-queried. This pass re-queried all 370 on
November 23, 2026. Testimony rows left incomplete (110) were not re-checked. No training was run,
and the held-out test was not touched.

| Split | Eligible | Version conflict | Insufficient evidence | Blocked | Total |
|---|---:|---:|---:|---:|---:|
| Train | 10 | 0 | 305 | 0 | 315 |
| Validation | 33 | 2 | 20 | 0 | 55 |
| **Total** | **43** | **2** | **325** | **0** | **370** |

## Rule

Every eligible row is `eligible_document_release_line_release_day_capture`. This is the
document-based rule adopted for macro releases (see `results/macro-coverage-v1/README.md`). The
release page says "For release at H:MM ET" at a time no later than the text cutoff. A Wayback copy
taken after the cutoff, 0.5 to 1,301 minutes later (median 198), has the same body text. No copy
from before the cutoff was found for any row. The rule is weaker than an official record that
reproduces the text, so keep the class labeled in the cohort.

Eight eligible rows from 2020 have pages that gained "Updated Term Sheet" links after release. For
these, the release-day archived body is the verified version: `fingerprint_sha256` hashes it, and
the added lines are listed in `added_since_release`. Every eligible row is fingerprinted on the
release-day archived body.

Two rows remain conflicts with real body differences: `board_monetary20200430a` and
`board_monetary20210713a`. Insufficient rows had no archive copy from 7 days before the cutoff to
24 hours after; these are mostly 2008–2019 releases.

## Reproduce

```sh
python scripts/recheck_board_archive.py \
  --ledger results/fed-spx-board-testimony-evidence-v1/fed_board_testimony_evidence_ledger.csv \
  --catalog analysis/macro_releases/fed_announcements_v2.csv \
  --out downloads/board-recheck-v1/shard0.jsonl --shard 0/2     # repeat until done; also --shard 1/2
python scripts/finalize_board_recheck.py downloads/board-recheck-v1/shard0.jsonl \
  downloads/board-recheck-v1/shard1.jsonl analysis/macro_releases/fed_announcements_v2.csv \
  results/fed-spx-board-recheck-v1
```

Script hashes: `recheck_board_archive.py` `41837fa2f7f984076db04ff47b8f58b05a562915d0051e6566c84b3723346e04`,
`finalize_board_recheck.py` `438e5b364d8362b62041d5cdf221599ee4a54b37499f9872a94fe64a23ca1d13`.
The eligible cohort in `results/fed-spx-news-cohort-v1/` predates this pass and must be rebuilt to
include these rows.
