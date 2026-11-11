# Full C1 paired comparison under an explicit timing assumption

**Status:** prepared November 11, 2026. This is a full-split *sensitivity*
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
  Full NPZ files contain prices and targets and stay on Delta. Publish only
  derived metrics, per-event losses, and prediction-only arrays after review.
- W&B is configured offline; the actual run folder and any shared link will
  be recorded after execution.

The training job has a 24-hour Slurm limit. Each paired mode has an eight-hour
limit. Actual GPU usage and job states will be recorded from Slurm. No claim
about article value should be made until the checkpoint is complete and both
paired jobs have passed alignment/provenance checks. Even then, the timing
assumption limits the conclusion.
