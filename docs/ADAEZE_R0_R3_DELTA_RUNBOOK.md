# Adaeze frozen-input R0–R3 Delta runbook

**Handoff status:** this file, the one-command launcher, and the three
`adaeze_c1_return_*.sbatch` wrappers form one execution package. Use this
runbook only after that package
is merged to `main`. Do not run from the handoff branch or from
`adaeze/pr8-speed-smoke`.

This is designed to be followed without another coding agent. Run each section
in order. A command that fails is a stop condition: preserve its output and do
not skip the guard, change a hash, reuse an output directory, or release an old
job to work around it.

## One-command launch

After this package is merged to `main`, update any clean Delta clone of the
repository and run:

```bash
git fetch origin main
git checkout main
git pull --ff-only origin main

CANCEL_SUPERSEDED=1 bash scripts/delta/adaeze_c1_return_launch.sh
```

That command safely checks and cancels only Adaeze's superseded jobs
`24050724` and `24051590`, creates a permanent checkout at the current merged
`main` commit, validates all four shell scripts, submits the smoke and the full
R0–R3 dependency graph, and writes every job ID to:

```text
/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/job_ids.env
```

If Slurm has already purged either superseded job from the active queue, the
launcher records it as inactive and continues; it does not treat a missing old
job ID as a launch failure.

If any guard fails, the launcher stops. Do not delete an existing campaign
root or weaken a guard to make it continue. The remaining sections explain
the same process step by step and provide monitoring, recovery, analysis, and
handoff commands; they are not additional submissions to run after the
one-command launcher.

## 1. Source of truth and non-negotiable inputs

- Repository: `DaezeI/11785-GS-Fuse-Submission-Public`
- Start from the current remote `main`; never start these runs from
  `adaeze/pr8-speed-smoke` or another pre-merge branch.
- Frozen snapshot:
  `/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1`
- Required snapshot content SHA-256:
  `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`
- Required manifest SHA-256:
  `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
- Required price SHA-256:
  `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
- Timing policy: `category4_1400_strict_before`
- Retained counts: 6,167 train / 2,302 validation / 2,236 test.
- Retained article publication times remain unverified. Do not call the data
  publication-time certified.

The frozen snapshot is a read-only, content-fingerprinted copy of the screened
manifest, price series, and retained article inputs. It excludes the 883 IDs
removed by the date-screen policy. Do not edit or replace it in place.

## 2. What the four variants mean

All four variants must keep the same snapshot, code commit, seed, backbones,
precision, optimizer, training budget, fusion, gate architecture, and paired
evaluation protocol.

| Variant | Direction weight | Gate target | Learnable last value | One change from previous |
|---|---:|---|---|---|
| R0 | 0 | magnitude only | off | Reference return model |
| R1 | 1 | magnitude only | off | Add direction loss |
| R2 | 1 | complete task | off | Use complete task for gate target |
| R3 | 1 | complete task | on | Add learnable last-value shortcut |

Do not substitute the older direction-weight `0.25/1/4` campaign. It is not
the merged R0–R3 protocol.

## 3. Cancel only Adaeze's superseded jobs

The following jobs were submitted from pre-merge commits and were last verified
as user-held with zero elapsed time. First reconfirm their ownership and state:

```bash
sacct -j 24050724,24051590 -X \
  --format=JobID,JobName%30,User%16,State,ExitCode,Elapsed,Start,End
```

If either job has started or belongs to someone else, stop. Otherwise cancel
both, then verify the final state:

```bash
scancel 24050724 24051590
sleep 2
sacct -j 24050724,24051590 -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End
```

Expected final state: `CANCELLED` with `Elapsed=00:00:00`.

In particular, Nicole's screened price jobs
`23840368`–`23840370` are a separate price-task comparison, not old R0–R3 jobs.

## 4. Confirm the execution package is merged

This runbook is shipped with these reusable files:

```text
scripts/delta/adaeze_c1_return_smoke.sbatch
scripts/delta/adaeze_c1_return_full.sbatch
scripts/delta/adaeze_c1_return_paired.sbatch
scripts/delta/adaeze_c1_return_launch.sh
```

They use the following Adaeze-owned paths:

| Field | Required Adaeze value |
|---|---|
| Job-name prefix | `adaeze-c1-return-...` |
| `RUN_ROOT` | `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1` |
| Slurm logs | `$RUN_ROOT/logs/...-%j.out` written as the absolute path in `#SBATCH --output` |
| `HF_HOME` | `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/hf-cache` |
| W&B name prefix | `adaeze-c1-return-...` |

They preserve:

- snapshot verification and all three required hashes;
- R0/R1/R2/R3 argument definitions;
- seed, batch size, epochs, model paths, cache, TF32, and no-4bit settings;
- `EXPECTED_CODE_COMMIT` and `C1_SNAPSHOT_SHA256` guards;
- fresh-output guards;
- `umask 007`.

Before merging the execution-package PR, run:

```bash
bash -n scripts/delta/adaeze_c1_return_smoke.sbatch
bash -n scripts/delta/adaeze_c1_return_full.sbatch
bash -n scripts/delta/adaeze_c1_return_paired.sbatch
bash -n scripts/delta/adaeze_c1_return_launch.sh
git diff --check
```

The wrapper differences are operational: Adaeze run paths and names, a one-hour
interactive A100 smoke, and non-preempt production A100 queues for full
training and paired full evaluation. Experiment definitions remain identical
to the merged R0–R3 protocol. After the PR merges, the resulting `main` commit
is the only commit used by all jobs below.

## 5. Create a permanent Delta checkout at the exact merged commit

On Delta:

```bash
umask 007

RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1
REPO_ROOT="$RUN_ROOT/repo"
LOG_ROOT="$RUN_ROOT/logs"
SNAPSHOT=/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1
SNAPSHOT_SHA=049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf

test ! -e "$RUN_ROOT"
mkdir -p "$LOG_ROOT"

git clone https://github.com/DaezeI/11785-GS-Fuse-Submission-Public.git "$REPO_ROOT"
cd "$REPO_ROOT"
git fetch origin main
RUN_COMMIT=$(git rev-parse origin/main)
git show "$RUN_COMMIT":docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_smoke.sbatch >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_full.sbatch >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_paired.sbatch >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_launch.sh >/dev/null
git checkout --detach "$RUN_COMMIT"

echo "RUN_COMMIT=$RUN_COMMIT"
test -z "$(git status --porcelain)"
printf '%s\n' "$RUN_COMMIT" > "$RUN_ROOT/source_commit.txt"
```

Do not move, pull, switch, or edit this checkout while submitted jobs refer to
it. Documentation changes after submission do not justify changing a live run
checkout.

Verify the snapshot and script syntax from this exact checkout:

```bash
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate

python scripts/freeze_c1_inputs.py \
  --verify "$SNAPSHOT" \
  --expect-content-sha256 "$SNAPSHOT_SHA"

bash -n scripts/delta/adaeze_c1_return_smoke.sbatch
bash -n scripts/delta/adaeze_c1_return_full.sbatch
bash -n scripts/delta/adaeze_c1_return_paired.sbatch
```

## 6. Submit the frozen-input R3 smoke

The smoke is an execution/reload check, not a performance result.

```bash
cd "$REPO_ROOT"

export EXPECTED_CODE_COMMIT="$RUN_COMMIT"
export C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA"

smoke_job=$(sbatch --parsable \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA" \
  scripts/delta/adaeze_c1_return_smoke.sbatch)

echo "smoke_job=$smoke_job"
squeue -j "$smoke_job" -o "%.18i %.28j %.12P %.10T %.10M %.10l %.25R"
```

Record the job ID immediately in the experiment handoff. Do not call it
completed until `sacct` reports `COMPLETED` with exit code `0:0`.

Monitor without repeatedly reading the whole log:

```bash
squeue -j "$smoke_job" -o "%.18i %.28j %.12P %.10T %.10M %.10l %.25R"
sacct -j "$smoke_job" -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End

tail -n 80 "$LOG_ROOT/return-smoke-$smoke_job.out"
```

Smoke acceptance checklist:

- job state `COMPLETED`, exit `0:0`;
- exact `RUN_COMMIT` printed in the log;
- frozen snapshot verification passed;
- all three training stages ran;
- `smoke-v1/best_model.pth` exists;
- `smoke-v1/resolved_config.json` exists;
- no fallback to price task, live article tree, 4-bit loading, or another split.

## 7. Do not repeat the paired smoke controls

The completed PR #8 smoke and metrics replay already established the small-run
training and structured-evaluation paths. This campaign performs one new smoke
only to verify the exact merged commit, frozen snapshot, and Adaeze-owned paths.
It does not repeat the two eight-event paired controls. The full paired
evaluations in Section 9 remain required for every completed full checkpoint.

If the single training smoke fails, stop. The full jobs use an `afterok`
dependency and will not start.

## 8. Submit full R0, R1, R2, and R3 training

Use the same exact checkout and commit for every variant. The simplest auditable
plan is four one-GPU jobs. They may run in parallel when Delta schedules them.

The launcher submits these jobs immediately with a dependency on successful
completion of the single training smoke:

```bash
gate_dependency="afterok:${smoke_job}"

r0_job=$(sbatch --parsable \
  --dependency="$gate_dependency" \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R0 \
  scripts/delta/adaeze_c1_return_full.sbatch)

r1_job=$(sbatch --parsable \
  --dependency="$gate_dependency" \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R1 \
  scripts/delta/adaeze_c1_return_full.sbatch)

r2_job=$(sbatch --parsable \
  --dependency="$gate_dependency" \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R2 \
  scripts/delta/adaeze_c1_return_full.sbatch)

r3_job=$(sbatch --parsable \
  --dependency="$gate_dependency" \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R3 \
  scripts/delta/adaeze_c1_return_full.sbatch)

printf 'R0=%s\nR1=%s\nR2=%s\nR3=%s\n' \
  "$r0_job" "$r1_job" "$r2_job" "$r3_job" | tee "$RUN_ROOT/training_job_ids.txt"
```

Expected training directories:

```text
ablation-r0-v1
ablation-r1-v1
ablation-r2-v1
full-v1
```

Do not interpret validation losses across variants as directly comparable when
their configured training objectives differ. Compare common return MAE/MSE and
directional hit-rate metrics on aligned events.

## 9. Submit paired full evaluations

For each training job, submit both no-text definitions. The zero-token mode
waits for training; the bypass mode waits for the zero-token mode. This keeps
the two modes for one checkpoint from racing on their shared text cache while
still allowing different variants to run in parallel.

```bash
submit_pair () {
  local dependency_job=$1
  local run_kind=$2
  local mode=$3
  sbatch --parsable \
    --dependency=afterok:"$dependency_job" \
    --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",RUN_KIND="$run_kind",MODE="$mode" \
    scripts/delta/adaeze_c1_return_paired.sbatch
}

r0_zero=$(submit_pair "$r0_job" ablation-r0-v1 zeroed_article_tokens)
r0_bypass=$(submit_pair "$r0_zero" ablation-r0-v1 fusion_bypass_ts_inst)
r1_zero=$(submit_pair "$r1_job" ablation-r1-v1 zeroed_article_tokens)
r1_bypass=$(submit_pair "$r1_zero" ablation-r1-v1 fusion_bypass_ts_inst)
r2_zero=$(submit_pair "$r2_job" ablation-r2-v1 zeroed_article_tokens)
r2_bypass=$(submit_pair "$r2_zero" ablation-r2-v1 fusion_bypass_ts_inst)
r3_zero=$(submit_pair "$r3_job" full-v1 zeroed_article_tokens)
r3_bypass=$(submit_pair "$r3_zero" full-v1 fusion_bypass_ts_inst)

printf '%s\n' \
  "R0_ZERO=$r0_zero" "R0_BYPASS=$r0_bypass" \
  "R1_ZERO=$r1_zero" "R1_BYPASS=$r1_bypass" \
  "R2_ZERO=$r2_zero" "R2_BYPASS=$r2_bypass" \
  "R3_ZERO=$r3_zero" "R3_BYPASS=$r3_bypass" \
  | tee "$RUN_ROOT/paired_job_ids.txt"
```

If a training job fails, its dependent paired jobs should become
`DependencyNeverSatisfied` and must not be described as attempted evaluations.

After all eight paired jobs complete with exit code `0:0`, compare adjacent
variants using the same zero-token control definition:

```bash
cd "$RUN_ROOT"

python repo/scripts/analyze_return_predictions.py \
  --predictions paired-ablation-r1-v1/zeroed_article_tokens.npz \
  --reference paired-ablation-r0-v1/zeroed_article_tokens.npz \
  --out r1-vs-r0-v1.json

python repo/scripts/analyze_return_predictions.py \
  --predictions paired-ablation-r2-v1/zeroed_article_tokens.npz \
  --reference paired-ablation-r1-v1/zeroed_article_tokens.npz \
  --out r2-vs-r1-v1.json

python repo/scripts/analyze_return_predictions.py \
  --predictions paired-full-v1/zeroed_article_tokens.npz \
  --reference paired-ablation-r2-v1/zeroed_article_tokens.npz \
  --out r3-vs-r2-v1.json
```

Each analyzer must confirm aligned events and provenance. A mismatch or an
incomplete input is a stop condition, not a result to summarize manually.

## 10. Queue and accounting commands

Current Adaeze jobs:

```bash
squeue -u "$USER" \
  -o "%.18i %.28j %.12P %.10T %.10M %.10l %.25R"
```

All current team-allocation jobs:

```bash
squeue -A bjjg-delta-gpu \
  -o "%.18i %.15u %.28j %.12P %.10T %.20R"
```

Detailed final accounting for a comma-separated job list:

```bash
sacct -j JOB1,JOB2,JOB3 -X \
  --format=JobID,JobName%32,User%16,Partition,State,ExitCode,Elapsed,Start,End,NodeList
```

Why a pending job is waiting:

```bash
scontrol show job JOB_ID | \
  grep -oE 'JobState=[^ ]+|Reason=[^ ]+|Partition=[^ ]+|TimeLimit=[^ ]+|StartTime=[^ ]+|Priority=[^ ]+'
sprio -j JOB_ID
squeue --start -j JOB_ID -o "%.18i %.10T %.19S %.20R"
```

`sbatch --test-only` prints a simulated job ID and estimated start. It does not
submit a real job and does not reserve the predicted slot.

## 11. Troubleshooting

### `JobHeldUser`

The job was manually held:

```bash
scontrol release JOB_ID
```

Release only jobs that belong to Adaeze and still match the current protocol.
Do not release the superseded `24050724` or `24051590`.

### `Priority` with no start estimate

The submission is valid but Slurm cannot predict a start. Do not repeatedly
cancel and resubmit; a new job loses accumulated age. Use `sprio`, `squeue
--start`, and `sbatch --test-only` only for planning.

### `Dependency` or `DependencyNeverSatisfied`

Inspect the parent:

```bash
scontrol show job CHILD_JOB | grep -oE 'Dependency=[^ ]+|Reason=[^ ]+'
sacct -j PARENT_JOB -X --format=JobID,State,ExitCode,Elapsed,Start,End
```

If the parent failed, keep the child as evidence until the failure is recorded,
then cancel the unusable child. Do not remove the dependency to force evaluation
against a missing or partial checkpoint.

### `EXPECTED_CODE_COMMIT` mismatch

Stop. Do not change the expected value just to make the job run. Confirm:

```bash
cd "$REPO_ROOT"
git rev-parse HEAD
cat "$RUN_ROOT/source_commit.txt"
git status --porcelain
```

The checkout must be clean and both commit values must match the commit recorded
before submission.

### Snapshot hash or manifest/price hash mismatch

Stop. Do not edit the read-only snapshot. Confirm the requested path and SHA.
If the snapshot is genuinely wrong, create and audit a new version; never
silently replace `c1-screened-return-v1`.

### `test ! -e "$OUTPUT"` failure

The output path already exists. Determine why:

```bash
find "$OUTPUT" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

Do not delete or overwrite it. Either document and resume a provenance-matched
interrupted run through a separately reviewed resume procedure, or choose a new
versioned output path.

### Interruption or unexpected job termination

The supplied full-training and full-evaluation wrappers use the non-preempt
`gpuA100x4` partition. Do not override them to a `*-preempt` partition: these
scripts require a fresh output directory and do not automatically resume.
If a job is nevertheless interrupted or terminated unexpectedly, check:

```bash
sacct -j JOB_ID -X \
  --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList
```

Do not blindly resubmit into the same directory. Verify the exact code commit,
resolved configuration, and available Stage-1/2 checkpoints first. The normal
full script's fresh-output guard intentionally prevents accidental resume.
Create a reviewed resume script using `--c1-allow-resume` only after checkpoint
provenance and compatibility are confirmed. Record the interruption as a
failed or interrupted attempt even if the later resume succeeds.

### Out of memory

Capture the end of the log and accounting before changing settings:

```bash
tail -n 150 LOG_FILE
sacct -j JOB_ID \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
```

Do not silently change batch size, precision, quantization, model size, or
hardware inside an R0–R3 comparison. A necessary change creates a new campaign
version and must be applied identically to all four variants.

### Import or module error

Confirm the job started in the permanent checkout and activated the tested
environment:

```bash
pwd
git rev-parse HEAD
module list
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Fix the reusable batch script in Git, test it, merge it, and create a new run
commit. Do not patch the live Delta checkout after submission.

### W&B is offline

Offline logging is expected. Keep the local W&B directory with the run output.
Record `PENDING` for the shared W&B URL until an authorized sync succeeds. A
missing online link does not turn a queued or failed run into a result.

## 12. Result and handoff checklist

For every smoke, training, or paired job record:

```text
Owner and research question:
Status: submitted | pending | running | completed | failed | canceled
Purpose: smoke | R0 | R1 | R2 | R3 | paired evaluation
Git commit and branch/main identity:
Exact sbatch command and job ID:
Environment/module versions:
Frozen snapshot path and content SHA-256:
Manifest and price SHA-256:
Timing policy and publication-time limitation:
Seed, batch size, epochs, precision, cache, backbones:
Output directory and log path:
Checkpoint path and SHA-256:
Resolved configuration path:
W&B run/link or PENDING:
Metrics and eligible counts:
Checks passed and checks still pending:
Finding, limitation, and next action:
```

For completed checkpoints:

```bash
sha256sum RUN_DIRECTORY/best_model.pth
cat RUN_DIRECTORY/resolved_config.json
```

Verify group readability:

```bash
find "$RUN_ROOT" -type d -exec chmod g+rx {} +
find "$RUN_ROOT" -type f -exec chmod g+r {} +
namei -l "$RUN_ROOT"
```

Ask another `delta_bjjg` member to verify read access before calling the handoff
complete.

## 13. Final decision rules

- A queued job has no result.
- A smoke result is not full-C1 performance or evidence of text value.
- Do not compare checkpoints trained from different commits, snapshots, seeds,
  budgets, or hardware as if they were paired.
- R0→R1, R1→R2, and R2→R3 are the supported one-factor comparisons.
- R3 versus the separate screened price baseline is a package comparison; do
  not attribute its difference to a single component.
- Use aligned test event IDs and common return metrics for variant comparisons.
- Record negative, failed, canceled, and inconclusive runs as carefully as
  successful ones.
