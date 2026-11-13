# Return-task integration handoff — November 13, 2026

This task branch starts from public `main` at `1db75e7` (including Nicole's
screened C1 audit/PR #7) and cherry-picks Adaeze's return design and
implementation commits `cb56231` and `f3ff7c2`. It adds the remaining paired
evaluation, checkpoint/provenance, and launch paths. The earlier queued C1
price job `23840368` is a separate experiment; its results must not be reported
as return-task results.

## What is ready in code

- C1 inputs and all three training stages use cumulative log returns in basis
  points around the last input close. The loss scale comes from training events
  only. The model uses Huber magnitude plus close-direction loss for the
  forecast and for strict pre-cross-attention TS-versus-full gate supervision.
- Raw-TS residual and learnable last-value shortcuts are explicit, separately
  recorded flags. Both are on for the frozen baseline. Return checkpoints save
  the task, flags, manifest/price/scale-population hashes, code commit, and seed; incompatible
  return loads fail before weight loading.
- Stage-3 return checkpoint selection uses validation task loss, return MAE/MSE,
  and eligible-count close DHR. It leaves test events for the final paired run.
- The paired runner saves aligned real, shuffled, no-article, fixed 50/50 gate,
  and strict TS-only predictions; targets, anchors, event IDs/types/dates,
  pre-event volatility, branch task losses, and learned gate weights. The CPU
  analyzer reports all-channel bp errors, endpoint close DHR with counts,
  training-majority and zero-return references, date-block paired intervals,
  and gate-utility diagnostics. Fixed 50/50 **at inference** measures dependence
  on the learned mixture; it is not a separately trained fixed-gate model.
- `scripts/delta/nicole_c1_return_{smoke,full,paired}.sbatch` are distinct from
  existing price scripts. They pin the screened manifest and price-file hashes,
  use fresh group-readable output paths, and explicitly request `return_bp`.

## Checks completed locally

`python -m compileall -q data/forecast_task.py data/c1_protocol.py model/gs_fuse.py train.py smoke_train.py test_pretrained_model.py scripts/run_text_controls.py scripts/analyze_return_predictions.py`

`python -m pytest -q tests/test_forecast_task.py tests/test_c1_protocol.py tests/test_variants.py tests/test_text_mask.py tests/test_text_control_runner.py tests/test_return_analysis.py tests/test_return_evaluator.py`

Result: 22 passed. These are CPU checks. The real three-stage GPU smoke,
checkpoint reload, and paired run remain pending. No new job was submitted.
The repository's `python scripts/check_text_value.py --require-torch` check
also passed (25 tests). All three new batch scripts passed `bash -n`.

## Delta execution order once training resumes

1. Put this branch at `/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/repo` and
   create `/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/logs` **before** `sbatch`.
   Keep the checkout fixed while jobs use it. The batch scripts print its commit.
2. Submit `scripts/delta/nicole_c1_return_smoke.sbatch`; inspect all three
   stages, the resolved config/audit, gradient and gate-scale logs, checkpoint,
   runtime, memory, and environment. It uses a 16-event training subset for
   execution checks, not a performance estimate.
3. Submit `RUN_KIND=smoke-v1 sbatch scripts/delta/nicole_c1_return_paired.sbatch`.
   Require a successful checkpoint reload and eight-event paired output. A
   second `MODE=fusion_bypass_ts_inst` run is optional if the group wants both
   no-article definitions.
4. Only after the smoke succeeds, submit
   `scripts/delta/nicole_c1_return_full.sbatch` into its fresh output directory.
   Then use `RUN_KIND=full-v1 sbatch scripts/delta/nicole_c1_return_paired.sbatch`
   for all test events. Report the selected task settings and seed before
   interpreting test comparisons. Record job IDs, code commit, W&B run, output
   path, checks, and limitations in a GitHub result note.

All new runs use `purged_split_c1_date_screen_v1.json` and the
`category4_1400_strict_before` timing assumption. The prepared-input audit
passed with zero shared price bars across splits. Publication times for retained
articles remain unverified; the screen is a sensitivity subset, not a
publication-time-certified dataset.

Q-Former/MoE changes are separate follow-up experiments. Keep the return task,
screened manifest, evaluation population, and metrics identical before comparing
them with the cross-attention/softmax baseline.
