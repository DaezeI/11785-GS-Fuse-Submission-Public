# Project instructions for coding agents

Read [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) first. It records the November 4 team
decision, ownership, implementation gaps, and experiment handoff requirements.
Follow current user instructions when they update this plan.

Then read [PROJECT_STATUS.md](PROJECT_STATUS.md) and its latest dated update.
They track progress after the original workflow, including reported Delta account
access, feature-branch setup, and the merged padding correction. Preserve the
distinction between verified code/results, teammate reports, and proposed work.
Develop on personal/task branches and open focused PRs for ready changes, following
the November 5 meeting guidance recorded in the update.

## Ownership and agreed protocol

- **C1 is the main split:** `splits/manifests/purged_split_c1_macro_eras.json`.
  C2 is an optional additional historical stress test.
- **Nicole:** window/publication-time checks, shared C1 loading for training and
  evaluation, then correct/shuffled/no-text controls.
- **Adaeze:** original gate, supervision, gradients, constant-gate controls.
  She already supplied split lists and aggregate text-control scores; reuse them.
- **Joye:** MoE routing/expert diagnostics; common Delta setup and baseline handoff.
- **Jessica:** Q-Former padding, attention/query redundancy, and ablations.

## Verify actual implementation and provenance

- At the workflow document's baseline, `train.py` and `scripts/run_text_controls.py`
  still use original loaders. The documented `build_purged_dataloaders` factory
  is absent. C1 integration is not complete merely because a manifest exists.
- All training stages, preprocessing fits, and training negative-report pools
  must respect C1. Check Stage-1 sliding-window boundaries explicitly.
- Old output directories can trigger automatic Stage-1/2 checkpoint resume.
  Verify checkpoint history/compatibility and use distinct directories for new runs.
- Price-window separation passed under two timing assumptions. Actual report
  release times, CSV timezone, and bar-label meaning remain unverified.
- The built-in TS-only comparison uses text-conditioned price features.
  `model/no_text_control.py` supplies inference controls; it does not itself fix
  gate training supervision.
- Nicole's CPU analysis and 25 focused tests ran. Full-model evaluation of the
  new helper has not; see `analysis/text_value/VALIDATION.md`.
- Adaeze's text-control scores and Joye's baseline are separate experiments.
  Do not combine scores across different checkpoints, samples, scalers, or metrics.

## Shared workflow

- GitHub: code/lists/instructions/small results. Drive: data/docs/output backups.
  Delta: GPU execution and working files. W&B: experiment records. ACCESS:
  allocation/membership. None automatically synchronizes the others.
- Locate existing files before asking teammates to recreate/upload them. Ask for
  unresolved access, exact run identity, or missing inputs only.
- Delta paths, working environment, and launch example remain pending until
  Joye verifies them. Do not guess local allocation codes, paths, or readiness.
- Use separate checkouts/output directories with common versioned inputs.
  Coordinate shared-function edits; preserve other contributors' work.
- Follow repository rules and user authorization for publication. Do not invent
  extra approval requirements, impersonate reviewers, or bypass required checks.
- Never commit credentials, tokens, private account records, raw data, or large
  model files. Record non-secret versions/hashes and shared locations when known.

## Verification and reporting

- Run checks appropriate to actual changes. The existing focused suite is
  `python scripts/check_text_value.py`; `--require-torch` fails on skipped tensor
  checks. This is not a full-backbone or GPU-runtime test.
- Use the handoff checklist in `TEAM_WORKFLOW.md`: owner, commit, command,
  environment, data/split/checkpoint versions, W&B link, outputs, controls, checks,
  findings, limits, next action. Mark missing facts PENDING.
- Explain results plainly. Separate executed analysis, unexecuted code,
  hypotheses, and measured effects. A constant gate or small shuffle gap alone
  does not prove a cause or that text is useless.
- Update workflow status and link shareable results when completing a handoff.
  Historical README commands and model filenames are not proof of C1 provenance.
