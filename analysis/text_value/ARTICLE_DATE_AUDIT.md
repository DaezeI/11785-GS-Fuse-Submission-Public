# Article-date integrity audit

**Owner:** Nicole Lyu. **Started:** November 12, 2026. **Status:** date scan
complete; versioned conservative screen prepared; full original C1 jobs held.

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
jobs `23812735` and `23812734` were pending. All three superseded jobs were
subsequently canceled to prevent the old manifest from running and to free
Slurm submission slots. No full C1 model result exists.
The five source-checked December records are test-only, so **those five alone
would call for test correction, not retraining**. The wider scan found affected
validation records, so the original queued job must not be released against
its old manifest. Checkpoint selection would use that validation set.

## Dataset-wide screen result

A read-only scan of all 12,267 `*_full_summary.txt` files in the shared Delta
article tree found 891 opening Reuters datelines. **887 differed from their
folder date**. Cross-referencing with the original C1 manifest gives 198
validation and 684 test candidates; no training entries were detected by
this particular Reuters rule. Of the 882 C1 candidates, 872 have a dateline
one to five days *later* than the folder date under a same-year comparison.
Ten have an earlier or anomalous date and need separate review. A spot check
of two validation candidates confirmed `Oct 23 (Reuters)` in
`11/20201022.txt` and `Feb 14 (Reuters)` in `12/20190213.txt`.

This is a **lower bound**, not a clean bill for the other 10,706 C1 records:
most summaries lack a parseable opening Reuters dateline. The source-verified
Zacks December 21 story is one example the rule misses. See the
[scan summary](c1_article_dateline_scan_summary_2026-11-12.json) and
[candidate IDs](c1_article_dateline_candidates_2026-11-12.csv). The scan
records IDs and dates only, not article text.

The scan does not produce 882 safe replacement folder names. The Reuters
opening usually supplies a calendar date without a release time or timezone,
and the summary may not establish when the complete event text was available.
For 541 of the 882 candidates, replacing the original date in the event ID
with the inferred dateline date would collide with an existing C1 event ID in
the same category. Preserve source IDs and files; record any verified
publication date and usable forecast anchor as separate, versioned metadata.

The [v1 exclusion ledger](c1_article_date_policy_v1.csv) gives each affected
original ID its evidence level and decision; its
[summary](c1_article_date_policy_v1_summary.json) checks the exact manifest
hashes and IDs. Five records have externally checked **calendar dates only**,
878 have only a Reuters-dateline candidate, and none has a verified release
time or replacement forecast anchor. The 10,705 retained C1 records also have
unknown release times. No raw folder or forecast window has been renamed or
reanchored; this is a screen, not a date-corrected baseline.

The [upstream CAMEF README](https://github.com/lakebodhi/CAMEF) calls the
folder an event date and describes generated summaries/counterfactual reports,
but supplies no per-article publication timestamp. Its
[loader](https://github.com/lakebodhi/CAMEF/blob/publish-camef-code/CAMEF/data/dataloader.py)
reads the summary filename date and appends a fixed clock time. A direct check
of the shared Delta article folders found generated report filename stems that
do not reliably match either the folder or the source date: for example,
`9/20231216.txt` has report stems `20231221` although the source-checked Reuters
date is Dec 19; `8/20231216.txt` has report stems `20240105` although the
source-checked Zacks date is Dec 21. Those stems cannot repair publication
times. The shared Drive data directory exposes event and series folders but no
date index at its top level; an original source archive with article metadata
has not been established.

All ten earlier/anomalous Reuters candidates were spot-checked in the Delta
summary openings. Nine are one or two days before weekend folder labels
(mostly Friday Reuters datelines); `12/20200803.txt` says `Feb 25 (Reuters)`
while discussing July activity, so even its dateline is suspect. The wider
candidate distribution includes +1 through +5 and negative offsets. There is
no justified global date shift. These ten remain excluded under v1.

The conservative [date-screened C1 manifest](../../splits/manifests/purged_split_c1_date_screen_v1.json)
excludes all 882 Reuters candidates and the one additional source-verified
Zacks record. It retains 6,167 train, 2,302 validation, and 2,236 test
events. This is a **sensitivity subset**, not a fully publication-time-safe
replacement for the agreed C1 benchmark. The original manifest and raw
dataset remain unchanged. The screened manifest still needs an emitted-window
audit on Delta before a new full run is submitted. That real-input check was
submitted as Slurm job `23835057` using
[`nicole_c1_date_screen_audit.sbatch`](../../scripts/delta/nicole_c1_date_screen_audit.sbatch);
it was pending at the last check. Replacement
[training](../../scripts/delta/nicole_c1_date_screen_full.sbatch) and
[paired-control](../../scripts/delta/nicole_c1_date_screen_paired.sbatch)
scripts point to this manifest and fresh output directories. Submit them only
after the audit passes.

Screened manifest SHA-256:
`1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`.
Regenerate it with:

```sh
python scripts/build_c1_date_screen.py \
  --base splits/manifests/purged_split_c1_macro_eras.json \
  --candidates analysis/text_value/c1_article_dateline_candidates_2026-11-12.csv \
  --confirmed analysis/text_value/article_date_confirmed_2026-11-12.json \
  --out splits/manifests/purged_split_c1_date_screen_v1.json
```

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

After the Delta emitted-window audit passes, submit the prepared replacement
training and paired jobs from the isolated checkout:

```sh
cd /work/hdd/bjjg/gs-fuse/nicole/c1-date-screen-v1/repo
training_job=$(sbatch --parsable scripts/delta/nicole_c1_date_screen_full.sbatch)
sbatch --dependency=afterok:"$training_job" \
  --export=ALL,MODE=zeroed_article_tokens \
  scripts/delta/nicole_c1_date_screen_paired.sbatch
sbatch --dependency=afterok:"$training_job" \
  --export=ALL,MODE=fusion_bypass_ts_inst \
  scripts/delta/nicole_c1_date_screen_paired.sbatch
```

Record the actual job IDs and source commit in this note; do not treat the
old unscreened jobs as equivalent. The resulting comparison must record the
screened manifest hash and exact included IDs. The prepared v1 screen is a
conservative exclusion policy; any later source-verified reanchoring needs a
new version.

## Relationship to Adaeze's notes

Adaeze's October 29 [Google Doc notes](https://docs.google.com/document/d/1FIIKD5c7RfsOzfk5lfy0GKHgZAddM-3slggN4DYq1j0/edit?tab=t.0)
raise the original split overlap and propose log returns, directional loss,
operational metrics, and volatility-conditioned gating. A later November 11
section proposes A–E gate/MoE tests; do not assume that section is Adaeze's
work. These are **modeling and evaluation experiments**, not requests for
article-date corrections. C1 already addresses the known cross-split overlap.
The date audit is a separate data-integrity task owned by Nicole. Discuss
which one or two of Adaeze's proposed experiments to prioritize; keep each
variant separate from the corrected C1 baseline.
