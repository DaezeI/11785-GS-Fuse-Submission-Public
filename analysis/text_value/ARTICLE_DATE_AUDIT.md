# Article-date integrity audit

**Owner:** Nicole Lyu. **Started:** November 12, 2026. **Status:** confirmed
date conflicts; dataset-wide scope under audit; full C1 jobs held.

## Why this matters

C1's `data/c1_protocol.py` reads the event date from the article folder name.
It uses that date to select the input and target price bars. The manifest and
loader audit verified no **cross-partition price-bar overlap**, but they did
not verify that each article was public at the assumed forecast start. An
incorrect folder date is therefore a model-input and evaluation problem, not
just a cosmetic label.
This audit does not yet establish whether the wrong labels originated in the
upstream archive or a later transfer; compare hashes/paths with the upstream
archive before attributing the error.

For `*/20231216.txt`, the C1 emitted price window has input beginning
`2023-12-15 13:25` and target ending `2023-12-19 14:15` in the CSV's
unverified timestamp convention. Four article summaries in that group open
with a later Reuters dateline. The fifth opening matches a publisher page
dated December 21:

| Event ID | Folder date | Date evidence | Summary SHA-256 |
|---|---|---|---|
| `8/20231216.txt` | Dec 16 | Dec 21 publisher page | `ea708915c2c55e6b7830610e10d4f58dcf95113c94bdc145961515637a4fd34d` |
| `9/20231216.txt` | Dec 16 | Dec 19 | `a64929ee3f067dad286360ea44cdf0a25ef4511eadbe2796118f7ad72e76a6cc` |
| `10/20231216.txt` | Dec 16 | Dec 20 | `874eff8987dde9bcc61d3289bbce390e6587a50b54aadfd79b5d0a73ae31f4de` |
| `11/20231216.txt` | Dec 16 | Dec 19 | `9f6377c2aad46e95a1d037c6bab36ca767e4bea96792b2723e20beaf14ffa4e7` |
| `12/20231216.txt` | Dec 16 | Dec 19 | `25d37bbb277767ffa4e834180c9e7d9bc1e0dbb3faa90bad6971b8217d346d7e` |

All five are in the **C1 test** partition. The category-8 opening matches
[Zacks's Dec 21 article](https://www.zacks.com/stock/news/2201083/bring-on-santa-economic-prints-mostly-favorable?art_rec=blog-ahead_of_wall_street-up_next-ID01-img-2201083).
The category-9 and -10 article identities and dates also match published
Reuters stories: [Dec 19 rally](https://www.investing.com/news/economy/futures-inch-up-as-investors-pin-hopes-on-fed-rate-cuts-3259697)
and [Dec 20 sell-off](https://www.investing.com/news/economy/futures-take-a-breather-after-rally-fedex-slides-3260613).
The category-11 and -12 openings match a
[Dec 19 Reuters futures report](https://wtvbam.com/2023/12/19/futures-inch-up-as-investors-pin-hopes-on-fed-rate-cuts/)
and [Dec 19 Reuters natural-gas report](https://finance.yahoo.com/news/us-natgas-prices-ease-record-153026461.html).
The dateline is date evidence, not an exact time-of-day release timestamp.
The [machine-readable evidence record](article_date_confirmed_2026-11-12.json)
stores these five IDs, source links, dates, and file hashes without article text.

## Current execution decision

On November 12, training job `23812725` was still pending and was put on a
reversible Slurm user hold (`JobHeldUser`). Its dependent paired-evaluation
jobs `23812735` and `23812734` remain pending. No full C1 model result exists.
The five known bad records are test-only, so **those five alone would call for
test correction, not retraining**. We have not yet established whether the
same problem affects train or validation records, so the full training job
stays held while the scope is measured.

## Audit and correction rule

Run `scripts/audit_article_datelines.py` against the exact C1 manifest and
Delta article tree. It compares an opening Reuters dateline with the folder
date and writes only IDs, dates, and counts. Its mismatches are **candidates**:
verify source identity and publication date before changing C1. Summaries
without a recognized dateline remain unknown; the scan cannot certify them.

```sh
python scripts/audit_article_datelines.py \
  --manifest splits/manifests/purged_split_c1_macro_eras.json \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --out outputs/c1_article_dateline_candidates.json
```

For verified wrong-date articles, preserve the raw dataset. Record the event
ID, raw hash, source publication evidence, decision, and the corrected split
version. Exclude an affected record from a timing-safe baseline if its actual
release time or correct price anchor cannot be established. Reanchoring an
article requires recomputing its price window, C1 role/boundary checks,
training-only scaler coverage, Stage-1 coverage, and evaluation IDs. Do not
rename a directory in place or silently change the existing manifest.

After the scope and correction are reviewed, create a versioned corrected
manifest, rerun the emitted-window audit, and then release or replace the held
jobs with commands pointing to that version. The resulting comparison must
record the corrected manifest hash and exact included IDs.

## Relationship to Adaeze's notes

Adaeze's October 29 [Google Doc notes](https://docs.google.com/document/d/1FIIKD5c7RfsOzfk5lfy0GKHgZAddM-3slggN4DYq1j0/edit?tab=t.0)
raise the original split overlap and propose log returns, directional loss,
and operational metrics. The November 11 notes propose A–E gate/MoE tests.
Those are useful **separate modeling experiments**. C1 already addresses the
known cross-split overlap. Neither section supplies a verified publication
date for these article files, so the date audit is the immediate data repair.
Adaeze can send two specific additional data corrections for this record;
proposed architecture or target changes should be assessed after the corrected
baseline and should not silently change its definition.
