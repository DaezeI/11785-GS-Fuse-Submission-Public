# Verified text files for `FED-PXTX-ALL-REL5-v2`

Owner: Adaeze Ilodianya. November 23, 2026. Built by `scripts/build_fed_spx_text_files.py` from the
saved v1/v2 audit evidence. No prices, returns or labels were read.

## What is here

- `FED-PXTX-ALL-REL5-v2-manifest.csv`: one row per eligible release with its evidence, fingerprint,
  text source and hashes.
- `FED-PXTX-ALL-REL5-v2-summary.json`: counts and the SHA-256 of each JSONL file.

The JSONL files themselves (`train.jsonl`, `val.jsonl`, `test.jsonl`, about 9 MB) are not committed
(team rule: no source text in git). They are in Drive, `01 Results and Evidence`, as
`fed-spx-text-v2-FED-PXTX-ALL-REL5-v2.zip`. Check them against `jsonl_sha256` in the summary.

| Split | Releases |
|---|---:|
| train | 173 |
| val | 149 |
| test | 172 |

Every row's `text` is rebuilt from saved evidence and its SHA-256 equals the fingerprint the audit
recorded for the version proven public before the cutoff. All 494 rows reproduce.

## Fields

`event_id`, `event_type`, `split`, `market_et`, `text_cutoff_utc`, `evidence_class`,
`fingerprint_sha256`, `text_source`, `text_sha256`, `text`, `model_text_source`, `model_text_sha256`,
`model_text`.

- **`text`**: the proven version (proof field). For PDF-backed releases it is raw PDF extraction.
- **`model_text`**: use this as model input. For the 274 PDF-backed releases it is today's web page,
  used only where the audit's wording check passes against this exact PDF (every web paragraph is in
  the released PDF, no number missing); 8 House PDFs fall back to the cleaned PDF text. Other types
  are already clean, so `model_text` = `text`.

`model_text` leaves out PDF-only material: release-line headers, chart captions and tables, the
minutes' Summary of Economic Projections and standing-authorization appendices.

## Known issues

- **Drop `stmt_20081029a` and `stmt_20140430a` (train) for now.** The minutes-statement extractor
  captured only part of each statement (124 of ~1,649 and 736 of ~5,359 characters). The stored text
  is verified but incomplete. Train without them: 171.
- Four 2008 statements omit the Board's discount-rate paragraph ("In a related action…"); the minutes
  do not approve that paragraph, so this is intended. Statements also omit the vote roster.
- Reading steps (release lines, document structure) are script-based; several script bugs were found
  and fixed in this audit. A follow-up plan proposes an LLM page-image reader, validated on ~100
  documents including every known failure case.

## Reproduce

```sh
GSFUSE_DOWNLOADS=<checkout with the audit's downloads/> python scripts/build_fed_spx_text_files.py \
  --cohort FED-PXTX-ALL-REL5-v2 --ledger results/fed-spx-news-cohort-v2/eligible_cohort.csv \
  --ledger results/fed-spx-test-text-v2/test_text_ledger.csv \
  --out-dir downloads/fed-spx-text-v2 --manifest-dir results/fed-spx-text-files-v2
```

Set `PYTHONPATH` to `downloads/fed-spx-deps` and `downloads/codex-validation-deps`. Board pages are
fetched once from their recorded archive URLs and cached.
