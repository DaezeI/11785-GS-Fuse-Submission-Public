#!/bin/bash

# One-command launcher for Adaeze's frozen-input R0-R3 Delta campaign.
# Run this from any clone only after this file has merged to origin/main.

set -euo pipefail
umask 007

RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1
REPO_ROOT="$RUN_ROOT/repo"
LOG_ROOT="$RUN_ROOT/logs"
REPO_URL=https://github.com/DaezeI/11785-GS-Fuse-Submission-Public.git
SNAPSHOT=/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1
SNAPSHOT_SHA=049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf
SUPERSEDED_JOBS=(24050724 24051590)

cancel_superseded_jobs() {
  local job record job_id owner state elapsed

  for job in "${SUPERSEDED_JOBS[@]}"; do
    record=$(squeue -h -j "$job" -o '%i|%u|%T|%M')
    if [[ -z "$record" ]]; then
      echo "superseded_job=$job is not active; nothing to cancel"
      continue
    fi

    IFS='|' read -r job_id owner state elapsed <<<"$record"
    if [[ "$job_id" != "$job" || "$owner" != "$USER" || "$state" != PENDING || "$elapsed" != 0:00 ]]; then
      echo "STOP: refusing to cancel unexpected job record: $record" >&2
      exit 1
    fi

    echo "canceling superseded pending job: $record"
    scancel "$job"
  done

  sleep 2
  for job in "${SUPERSEDED_JOBS[@]}"; do
    if squeue -h -j "$job" | grep -q .; then
      echo "STOP: superseded job $job is still active" >&2
      exit 1
    fi
  done
}

if [[ "${CANCEL_SUPERSEDED:-0}" != 1 ]]; then
  echo "STOP: rerun with CANCEL_SUPERSEDED=1 to authorize cancellation of only jobs 24050724 and 24051590" >&2
  exit 2
fi

if [[ -e "$RUN_ROOT" ]]; then
  echo "STOP: campaign root already exists: $RUN_ROOT" >&2
  echo "Do not delete or reuse it. Inspect it and choose a reviewed new version if needed." >&2
  exit 1
fi

if [[ ! -d "$SNAPSHOT" ]]; then
  echo "STOP: frozen snapshot is missing: $SNAPSHOT" >&2
  exit 1
fi

# Confirm that the complete launcher package is present on remote main before
# canceling an old job or creating the new campaign root.
git rev-parse --is-inside-work-tree >/dev/null
git fetch "$REPO_URL" main
PREFLIGHT_COMMIT=$(git rev-parse FETCH_HEAD)
git cat-file -e "$PREFLIGHT_COMMIT":docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md
git cat-file -e "$PREFLIGHT_COMMIT":scripts/delta/adaeze_c1_return_launch.sh
git cat-file -e "$PREFLIGHT_COMMIT":scripts/delta/adaeze_c1_return_smoke.sbatch
git cat-file -e "$PREFLIGHT_COMMIT":scripts/delta/adaeze_c1_return_full.sbatch
git cat-file -e "$PREFLIGHT_COMMIT":scripts/delta/adaeze_c1_return_paired.sbatch

cancel_superseded_jobs

mkdir -p "$LOG_ROOT"
git clone --no-checkout --branch main --single-branch "$REPO_URL" "$REPO_ROOT"
cd "$REPO_ROOT"
git checkout --detach "$PREFLIGHT_COMMIT"

RUN_COMMIT=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
git show "$RUN_COMMIT":docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_launch.sh >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_smoke.sbatch >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_full.sbatch >/dev/null
git show "$RUN_COMMIT":scripts/delta/adaeze_c1_return_paired.sbatch >/dev/null
test "$RUN_COMMIT" = "$PREFLIGHT_COMMIT"

bash -n scripts/delta/adaeze_c1_return_launch.sh
bash -n scripts/delta/adaeze_c1_return_smoke.sbatch
bash -n scripts/delta/adaeze_c1_return_full.sbatch
bash -n scripts/delta/adaeze_c1_return_paired.sbatch

printf '%s\n' "$RUN_COMMIT" > "$RUN_ROOT/source_commit.txt"
printf '%s\n' "$SNAPSHOT" > "$RUN_ROOT/snapshot_path.txt"
printf '%s\n' "$SNAPSHOT_SHA" > "$RUN_ROOT/snapshot_sha256.txt"
JOB_FILE="$RUN_ROOT/job_ids.env"
printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT" > "$JOB_FILE"

report_partial_launch() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    echo "STOP: campaign submission ended early with status $status" >&2
    echo "Jobs submitted before the failure are recorded in $JOB_FILE" >&2
    cat "$JOB_FILE" >&2
  fi
}
trap report_partial_launch EXIT

smoke_job=$(sbatch --parsable \
  --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA" \
  scripts/delta/adaeze_c1_return_smoke.sbatch)
printf 'SMOKE=%s\n' "$smoke_job" >> "$JOB_FILE"

gate_dependency="afterok:${smoke_job}"
r0_job=$(sbatch --parsable --dependency="$gate_dependency" --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R0 scripts/delta/adaeze_c1_return_full.sbatch)
printf 'R0=%s\n' "$r0_job" >> "$JOB_FILE"
r1_job=$(sbatch --parsable --dependency="$gate_dependency" --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R1 scripts/delta/adaeze_c1_return_full.sbatch)
printf 'R1=%s\n' "$r1_job" >> "$JOB_FILE"
r2_job=$(sbatch --parsable --dependency="$gate_dependency" --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R2 scripts/delta/adaeze_c1_return_full.sbatch)
printf 'R2=%s\n' "$r2_job" >> "$JOB_FILE"
r3_job=$(sbatch --parsable --dependency="$gate_dependency" --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",VARIANT=R3 scripts/delta/adaeze_c1_return_full.sbatch)
printf 'R3=%s\n' "$r3_job" >> "$JOB_FILE"

submit_pair() {
  local dependency_job=$1
  local run_kind=$2
  local mode=$3
  sbatch --parsable \
    --dependency=afterok:"$dependency_job" \
    --export=ALL,EXPECTED_CODE_COMMIT="$RUN_COMMIT",C1_SNAPSHOT_SHA256="$SNAPSHOT_SHA",RUN_KIND="$run_kind",MODE="$mode" \
    scripts/delta/adaeze_c1_return_paired.sbatch
}

r0_zero=$(submit_pair "$r0_job" ablation-r0-v1 zeroed_article_tokens)
printf 'R0_ZERO=%s\n' "$r0_zero" >> "$JOB_FILE"
r0_bypass=$(submit_pair "$r0_zero" ablation-r0-v1 fusion_bypass_ts_inst)
printf 'R0_BYPASS=%s\n' "$r0_bypass" >> "$JOB_FILE"
r1_zero=$(submit_pair "$r1_job" ablation-r1-v1 zeroed_article_tokens)
printf 'R1_ZERO=%s\n' "$r1_zero" >> "$JOB_FILE"
r1_bypass=$(submit_pair "$r1_zero" ablation-r1-v1 fusion_bypass_ts_inst)
printf 'R1_BYPASS=%s\n' "$r1_bypass" >> "$JOB_FILE"
r2_zero=$(submit_pair "$r2_job" ablation-r2-v1 zeroed_article_tokens)
printf 'R2_ZERO=%s\n' "$r2_zero" >> "$JOB_FILE"
r2_bypass=$(submit_pair "$r2_zero" ablation-r2-v1 fusion_bypass_ts_inst)
printf 'R2_BYPASS=%s\n' "$r2_bypass" >> "$JOB_FILE"
r3_zero=$(submit_pair "$r3_job" full-v1 zeroed_article_tokens)
printf 'R3_ZERO=%s\n' "$r3_zero" >> "$JOB_FILE"
r3_bypass=$(submit_pair "$r3_zero" full-v1 fusion_bypass_ts_inst)
printf 'R3_BYPASS=%s\n' "$r3_bypass" >> "$JOB_FILE"

chmod g+r "$JOB_FILE" "$RUN_ROOT/source_commit.txt" "$RUN_ROOT/snapshot_path.txt" "$RUN_ROOT/snapshot_sha256.txt"
trap - EXIT

echo "Submitted the complete guarded R0-R3 campaign."
cat "$JOB_FILE"
echo
echo "Monitor with:"
echo "squeue -u \"$USER\" -o '%.18i %.28j %.12P %.10T %.10M %.10l %.25R'"
echo "The full jobs cannot run unless the single training smoke completes successfully."
