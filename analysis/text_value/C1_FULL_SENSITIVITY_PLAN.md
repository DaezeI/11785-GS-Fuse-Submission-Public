# Full C1 paired comparison under an explicit timing assumption

**Superseded for execution on November 12.** Article-date conflicts were
found in the original validation and test entries. Training job `23812725`
was put on a user hold; its dependent jobs must not be used as the full C1
comparison. See the [article-date audit and screened manifest](ARTICLE_DATE_AUDIT.md)
and [draft PR #7](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/7).
This page preserves the November 11 submission record.

**Status:** Slurm jobs submitted November 11, 2026. At the November 12 check,
full training and its two dependent paired jobs were still pending; the
256-event diagnostic had completed successfully.
This is a full-split *sensitivity*
experiment using `category4_1400_strict_before`, not a claim that every
article was public before its assigned price window. Publication times for
many categories and the price CSV timezone/bar-label convention remain
unverified. See [the protocol handoff](C1_PROTOCOL.md).

## Question and design

Does the correct article change errors relative to shuffled articles and two
separate article-free controls when the checkpoint and price examples are held
fixed? The run uses the committed C1 manifest: 6,167 train, 2,500 validation,
and 2,921 test events. Training uses the existing three-stage cross-attention
baseline, one A100, batch 8, stride-10 Stage-1 windows, two epochs per stage,
Llama-3.2-1B and MOMENT-1-large, and a fresh output directory.

After the complete model checkpoint is saved, two independent one-A100 Slurm
jobs evaluate **all** C1 test events using the same checkpoint, prices,
training-only scaler, and event IDs. Each records correct text and three
seeded shuffled assignments (23, 24, 25); one uses `zeroed_article_tokens`
and the other `fusion_bypass_ts_inst`. The jobs may run in parallel after
training succeeds. These article-free interventions must remain distinct.

| Job | Purpose | Dependency | State at submission |
|---|---|---|---|
| `23812725` | Full C1 three-stage sensitivity training | none | Pending (priority) |
| `23812735` | All-test paired control, zeroed article tokens | afterok:`23812725` | Pending (dependency) |
| `23812734` | All-test paired control, fusion bypass | afterok:`23812725` | Pending (dependency) |
| `23812736` | 256-test-event diagnostic using the older 16-training-event smoke checkpoint | none | Completed November 12, exit 0; 8m15s |

The 256-event diagnostic completed all 256 predictions and an analysis with
three shuffles, the `zeroed_article_tokens` control, and no missing provenance
fields. Its log is `/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/logs/c1-paired-nicole-c1-smoke256-zeroed-23812736.out`;
its output is under `/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/paired_smoke256_v1/`.
It is only an evaluator check. It cannot establish text value because its
checkpoint learned from 16 events. The two full paired jobs are the planned
substantive comparison, conditional on a complete checkpoint.

## Reproduction and data handling

- Training script: [`nicole_c1_full_sensitivity.sbatch`](../../scripts/delta/nicole_c1_full_sensitivity.sbatch).
- Paired script: [`nicole_c1_full_paired.sbatch`](../../scripts/delta/nicole_c1_full_paired.sbatch),
  submitted twice with `MODE` and `afterok` on the training job.
- Shared price and manifest SHA-256 values at the prior audited state:
  `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
  and `7da0d5666ccf54fa41f99f2b947c83344acc1fd4c40c4017127e3a5bb87276fa`.
  The run will record actual emitted counts and checkpoint hash; stop if the
  loaded inputs differ unexpectedly.
- Remote working root: `/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/`.
  The staged source commit is `3b71415`; its identifier is recorded beside
  the remote outputs.
  Full NPZ files contain prices and targets and stay on Delta. Publish only
  derived metrics, per-event losses, and prediction-only arrays after review.
- W&B is configured offline; the actual run folder and any shared link will
  be recorded after execution.

The training job requests one preemptible A100 for up to 16 hours and permits
requeue into the same fresh output directory; the code can restore completed
Stage-1/2 checkpoints from that run. Each paired mode has an eight-hour limit.
Actual GPU usage and job states will be recorded from Slurm. No claim
about article value should be made until the checkpoint is complete and both
paired jobs have passed alignment/provenance checks. Even then, the timing
assumption limits the conclusion.
