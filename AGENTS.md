# Project instructions for coding agents

Read [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) first. It records the November 4 team
decision, ownership, implementation gaps, and experiment handoff requirements.
Follow current user instructions when they update this plan.

Then read [PROJECT_STATUS.md](PROJECT_STATUS.md) for current progress and any
relevant dated update for historical context. Preserve the distinction between
verified code/results, teammate reports, queued jobs, and proposed work.
Before any full C1 run, read the [November 12 article-date audit](analysis/text_value/ARTICLE_DATE_AUDIT.md).
The original full-run jobs were canceled after widespread validation/test date
conflicts were found. Use the versioned date-screened manifest only after its
real-input window audit passes; do not release the old jobs or describe the
screened subset as publication-time certified.
The screened real-input audit passed on November 13 with all 10,705 events
emitted and zero cross-split shared price bars. Its [summary](analysis/text_value/c1_date_screen_v1_audit_summary.json)
has exact counts. A brief Delta administrator hold cleared. Training is pending
GPU scheduling, and the paired jobs wait for training; check their live states
before reporting them.
**November 13 team decision:** For new course-project C1 training and matched
real/shuffled/no-text comparisons, use
`splits/manifests/purged_split_c1_date_screen_v1.json` (6,167 train / 2,302
validation / 2,236 test). The original C1 manifest is historical; do not use it
for new full runs. The screened manifest excludes 883 suspect records but does
not repair or verify publication times for retained news. State this as a
limitation of model claims. Do not launch the separate eight-announcement
timing comparison; its queued job was canceled at the user's request.
Develop on personal/task branches and open focused PRs for ready changes, following
the November 5 meeting guidance recorded in the update.

For every proposed or executed experiment, follow [EXPERIMENTS.md](EXPERIMENTS.md).
Record its question, comparison, intended change, fixed settings, status,
provenance, result, and limitations. Prefer one changed factor when attributing
an effect. Label multi-change comparisons as packages, and never infer a
component's effect from them. Update the register and link a result note even
when a run fails or finds no improvement; do not turn queued work into a result.

## Ownership and agreed protocol

- **C1 is the main split:** use the date-screened C1 manifest named above for
  new full runs. The original `purged_split_c1_macro_eras.json` defines the
  historical split; C2 is an optional additional historical stress test.
- **Nicole:** window/publication-time checks, shared C1 loading for training and
  evaluation, then correct/shuffled/no-text controls.
- **Adaeze:** original gate, supervision, gradients, constant-gate controls.
  She already supplied split lists and aggregate text-control scores; reuse them.
- **Joye:** MoE routing/expert diagnostics; common Delta setup and baseline handoff.
- **Jessica:** Q-Former padding, attention/query redundancy, and ablations.

## Verify actual implementation and provenance

- At the workflow document's November 4 baseline, `train.py` and
  `scripts/run_text_controls.py` still used original loaders. Merged PR #4 adds
  opt-in C1 integration through the normal entrypoints; review the actual
  loader and emitted-window audit rather than inferring provenance from a name.
- All training stages, preprocessing fits, and training negative-report pools
  must respect C1. Check Stage-1 sliding-window boundaries explicitly.
- Old output directories can trigger automatic Stage-1/2 checkpoint resume.
  Verify checkpoint history/compatibility and use distinct directories for new runs.
- Price-window separation passed under two timing assumptions. Actual report
  release times, CSV timezone, and bar-label meaning remain unverified.
- The built-in TS-only comparison uses text-conditioned price features.
  `model/no_text_control.py` supplies inference controls; it does not itself fix
  gate training supervision.
- Nicole's CPU analysis and 25 focused tests ran. Merged PR #4 adds five C1 tests,
  a completed three-stage 16-event GPU smoke checkpoint, and paired controls
  on eight test events. See `analysis/text_value/C1_SMOKE_RESULTS.md`; the
  smoke result is not a full C1 benchmark.
- Adaeze's text-control scores and Joye's baseline are separate experiments.
  Do not combine scores across different checkpoints, samples, scalers, or metrics.

## Shared workflow

- GitHub: code/lists/instructions/small results. Drive: data/docs/output backups.
  Delta: GPU execution and working files. W&B: experiment records. ACCESS:
  allocation/membership. None automatically synchronizes the others.
- Locate existing files before asking teammates to recreate/upload them. Ask for
  unresolved access, exact run identity, or missing inputs only.
- Nicole verified her Delta login, shared inputs/environment, and two completed
  one-A100 Slurm jobs; exact paths and jobs are in the PR #4 handoff. Joye's
  common setup and full-baseline handoff remain separately owned. Do not infer
  another teammate's login or checkpoint readiness from Nicole's jobs.
- Use separate checkouts/output directories with common versioned inputs.
  Coordinate shared-function edits; preserve other contributors' work.
- For every Delta run, follow the permanent team-readable handoff rule in
  `TEAM_WORKFLOW.md`: commit the reusable Slurm script/configuration to the task
  branch; use a named owner/run directory under `/work/hdd/bjjg/gs-fuse/runs/`
  or verify group access to a personal project directory; set `umask 007` in
  the batch script; and check that another teammate can read the run record.
  Record the code commit, job ID, split, checkpoint, command, and output path
  in a GitHub result note. Keep large outputs on Delta. Do not move a checkout
  while its submitted jobs depend on its paths.
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
