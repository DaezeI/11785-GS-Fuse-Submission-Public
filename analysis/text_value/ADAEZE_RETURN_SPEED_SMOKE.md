# Adaeze return-task speed smoke

## Question

Do two runtime-only changes let the screened-C1 return-task smoke run complete
without changing its mathematical or evaluation contract?

## Comparison and intended change

- Reference: PR #8 return-task code before this focused patch.
- Package under test: anomaly detection is opt-in in `train.py`, and Stage 3 no
  longer forces Python garbage collection plus an empty CUDA cache every step.
- Checkpoint format: frozen text-backbone weights are reloaded from the fixed
  pretrained path instead of copied into every checkpoint. Fine-tuned text
  weights are saved once, and the legacy `llm` key remains loadable.

This is a multi-change operational package. It cannot attribute timing gains to
one component and is not a model-quality comparison.

## Fixed settings

- Manifest: `splits/manifests/purged_split_c1_date_screen_v1.json`
- Timing policy: `category4_1400_strict_before`
- Task: `return_bp`; seed 0; raw-TS residual and learnable last value enabled
- 16 training events, 8 validation/test events; 32/16 Stage-1 windows
- One epoch per stage; batch size 2; cross-attention plus softmax gate
- One A100, TF32, offline W&B, local pretrained MOMENT and Llama-3.2-1B

## Status and provenance

Status: **COMPLETED — operational smoke result only**.

- Owner: Adaeze
- Branch: `adaeze/pr8-speed-smoke`
- Source: PR #8 (`nicole/return-task-integration`)
- Slurm script: `scripts/delta/adaeze_c1_return_speed_smoke.sbatch`
- Delta output: `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/smoke-speed-v1`
- Runtime-package commit: `5d25ace3624c029988bd3a5602f8a90442244031`;
  executed Slurm-script commit: `15566fbd0208e96c642b7105d9e10f5b6966a198`
- Original job ID: `24045381` (canceled while pending; no GPU execution)
- Submission command: `sbatch --parsable scripts/delta/adaeze_c1_return_speed_smoke.sbatch`
- Scheduling update: job `24045381` remained pending for priority. A
  non-submitting `sbatch --test-only` check showed that the same one-GPU smoke
  request was schedulable on `gpuA100x4-interactive` with a 30-minute limit.
  Replacement job ID: `24045929`, submitted from commit
  `15566fbd0208e96c642b7105d9e10f5b6966a198` on
  `gpuA100x4-interactive` with a 30-minute limit. Slurm reported it running
  on `gpua064` with start time `2026-11-16T13:30:15`.
- Completion: job `24045929` finished with exit code `0:0` after `00:08:50`.
  The largest reported step MaxRSS was `12283196K` (about 12 GB of host RAM).
- Preflight checks: 17 focused return/checkpoint tests passed; 25 focused
  text-value tests passed with `--require-torch`; Slurm `--test-only` accepted
  the request.
- Result: the three-stage 16-event smoke completed successfully. This verifies
  that the runtime package executes end to end; it is not a full-C1 quality
  result and does not isolate either speed change's effect.

## Limitations

This 16-event smoke test is not a full C1 benchmark. The screened manifest
removes known date conflicts but does not verify publication times for retained
news, so it cannot support publication-time-certified claims.

## Structured-metrics replay

Status: **COMPLETED — eight-event inference replay only**.

The completed selected checkpoint was replayed without training on the same
eight evenly spaced validation and eight test events. The replay records task,
magnitude, raw and weighted direction losses, return MAE/MSE, and close-direction
hit rate plus eligible counts at each configured horizon. It wrote to the new
directory
`/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/metrics-replay-v1` and does
not modify the source run or checkpoint. The reusable batch entrypoint is
`scripts/delta/adaeze_c1_return_metrics_replay.sbatch`.

Initial replay job `24046517` failed before model initialization after 11 seconds
with `ModuleNotFoundError: No module named 'data'`; its batch step used about
551 MB of host RAM and did not create the replay output directory. The runner
now adds the repository root to Python's import path. This failed launch produced
no metrics and did not read or modify checkpoint tensors.

Replacement job `24046557` completed on `gpua001` with exit code `0:0` in
`00:08:04`; its batch step MaxRSS was `5731336K`. It used replay commit
`fb954c3e74ae212f2ffc1d5c8384d2fe94b25854` and checkpoint SHA-256
`e6c9a90dac8bceaad9339841e5f5ce4ea4b7c3551ec471ba58fff8cc316b27b5`.
The source checkpoint was loaded strictly and was not overwritten.

| Split | Events | Task loss | Magnitude loss | Direction loss | MAE (bp) | MSE (bp2) | Close DHR h35 / h70 / h140 |
|---|---:|---:|---:|---:|---:|---:|---|
| Validation | 8 | 1.686638 | 0.998040 | 0.688598 | 67.8874 | 7676.2405 | 1.000 / 0.625 / 0.750 |
| Test | 8 | 1.978154 | 1.288921 | 0.689233 | 82.3969 | 11294.3127 | 0.625 / 0.750 / 0.625 |

Every DHR cell had eight eligible events. Direction weight was 1.0, so raw and
weighted direction loss are identical. These values confirm structured metric
capture for the selected smoke checkpoint. With only eight evenly spaced events
per split, they do not estimate full-C1 performance, text value, or uncertainty.

Delta artifacts are group-readable under
`/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/metrics-replay-v1`:

- `metrics_replay_summary.json` SHA-256:
  `707605419dcac58e148886fb404f843b29cff9d3c44b39a119da81ac35957de7`
- `validation_return_metrics.json` SHA-256:
  `36f1aac99822f1a6533a7d74ab835f4219ffd822b403db8c00afebe489803cf2`
- `test_return_metrics.json` SHA-256:
  `be4106c6db69e5f6d414714dd63d3dc0afd6c5c347eed01e7ba049912fc810a0`

## Full screened-C1 return baseline

Status: **USER-HELD — no result**.

- Owner: Adaeze
- Job ID: `24050724`; a November 16 terminal check showed `JobHeldUser`, zero
  elapsed time, and unknown start/end times after Adaeze paused the job.
- Executed code commit: `8fef0648afb1f2583e01f56cdb307e3a4085d10f`
- Batch script: `scripts/delta/adaeze_c1_return_full.sbatch`
- Submission command:
  `sbatch --parsable scripts/delta/adaeze_c1_return_full.sbatch`
- Manifest: `splits/manifests/purged_split_c1_date_screen_v1.json`, SHA-256
  `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
- Timing policy: `category4_1400_strict_before`
- Price file SHA-256:
  `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
- Output path:
  `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/full-v1`
- Configuration: return basis-point target, raw-TS residual and learnable
  last-value shortcut enabled, cross-attention plus softmax gate, seed 0,
  batch size 8, Stage-1/2/3 budgets of two epochs each, one A100, TF32,
  offline W&B.

This job is independent of Nicole's price-task jobs `23840368`–`23840370` and
does not supersede or alter them. It is held queued work, not a result.
If it completes, record the scheduler state, runtime, selected checkpoint hash,
resolved configuration, W&B run, metrics, and limitations before interpreting
model quality. Retained article publication times remain unverified, so the
screened run cannot support publication-time-certified text claims.

## Four-GPU return campaign

Status: **USER-HELD — no result**.

- Owner: Adaeze
- Job ID: `24051590`; a November 16 terminal check showed `JobHeldUser`, zero
  elapsed time, and unknown start/end times after Adaeze paused the job.
- Executed code commit: `74e5cf8cca08fce55b478c5919e53e504cf88908`
- Batch script: `scripts/delta/adaeze_c1_return_campaign.sbatch`
- Submission command:
  `sbatch --parsable scripts/delta/adaeze_c1_return_campaign.sbatch`
- Run root: `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-campaign-v1`
- Resources: one four-A100 node, 32 CPU cores, 160 GB host memory, 12-hour
  limit, preempt queue with explicit requeue and provenance-checked resume.
- Parallel variants, all with seed 0 and identical screened-C1 data/budget:
  baseline direction weight 1; direction weights 0.25 and 4; and direction
  weight 1 with the learnable last-value shortcut disabled.

The direction-weight and last-value runs are separate one-factor comparisons
against the campaign baseline, but they predate the later frozen-input R0–R3
protocol on Nicole's branch. Both this campaign and redundant single-GPU job
`24050724` are held; neither should be released as the final protocol without
an explicit reconciliation decision. Nicole's jobs remain separate and
unchanged. A test-only estimate predicted a November 17 slot, but it was not a
reservation and is not recorded as a scheduled start.
