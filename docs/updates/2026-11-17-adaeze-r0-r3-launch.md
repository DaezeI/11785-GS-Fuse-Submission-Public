# Adaeze frozen-input R0–R3 Delta campaign launch — November 17, 2026

Adaeze's guarded one-command launcher submitted the frozen-input R0–R3 Delta
campaign from `main`. All operations ran from the clean checkout at
`/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/repo`.

## 1. Provenance and fixed invariants

- **Repository and commit:** `DaezeI/11785-GS-Fuse-Submission-Public`, submitted from merged `main`.
- **Run root:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1`
- **Source commit record:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/source_commit.txt`
- **Job ID record:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/job_ids.env`
- **Frozen snapshot:** `/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1`
- **Required content SHA-256:** `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`
- **Required manifest SHA-256:** `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
- **Required price SHA-256:** `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
- **Retained counts:** 6,167 train / 2,302 validation / 2,236 test.
- **Timing policy:** `category4_1400_strict_before`
- **Data limitation:** Retained article publication times remain unverified. Do not describe the data as publication-time certified.

## 2. Superseded job resolution

Superseded pre-merge jobs `24050724` and `24051590` were confirmed cancelled and
purged from Slurm's active queue (`Elapsed=00:00:00`). The launcher's cancellation
helper verified their inactive status cleanly.

Nicole's screened price jobs `23840368`–`23840370` remain separate and unaltered.

## 3. Verified smoke execution check

Smoke job `24055019` (`adaeze-c1-return-smoke`) ran on `gpuA100x4-interactive`
as the gating requirement before full training.

- **Slurm accounting:** `JobID=24055019`, State `COMPLETED`, ExitCode `0:0`, Elapsed `00:45:23` (Start: 2026-11-16T21:24:37, End: 2026-11-16T22:10:00).
- **Log output:** `logs/return-smoke-24055019.out` confirms snapshot SHA-256 verification passed, all three training stages completed, and model weights and tokenizer reloaded for test evaluation.
- **Artifacts:**
  - `smoke-v1/best_model.pth` (1.8 GB, group-readable `delta_bjjg`)
  - `smoke-v1/resolved_config.json` (2.5 KB)
- **Offline W&B:** `smoke-v1/wandb/offline-run-20261116_220926-zbfncq9l`
- **Acceptance:** All smoke criteria from `ADAEZE_R0_R3_DELTA_RUNBOOK.md` passed. The dependency gate `afterok:24055019` was satisfied.
- **Epistemic boundary:** Smoke metrics on eight test events are execution and reload checks only. They are not full-C1 benchmark performance or evidence of text value.

## 4. Live submitted campaign roster

All downstream jobs are dependency-gated and submitted to partition `gpuA100x4`:

| Job ID | Job Name | Variant / Role | Gated on | Slurm State | Notes |
|---|---|---|---|---|---|
| `24055019` | `adaeze-c1-return-smoke` | R3 smoke check | None | `COMPLETED` | Exit `0:0`, elapsed `00:45:23` |
| `24055020` | `adaeze-c1-return-full` | R0 full training | `24055019` | `PENDING` | Dependency cleared; Priority 1545; estimated start `2026-11-18 16:36:05` |
| `24055021` | `adaeze-c1-return-full` | R1 full training | `24055019` | `PENDING` | Dependency cleared; Priority 1545; estimated start `2026-11-18 16:36:05` |
| `24055022` | `adaeze-c1-return-full` | R2 full training | `24055019` | `PENDING` | Dependency cleared; Priority 1545; estimated start `2026-11-18 16:36:05` |
| `24055023` | `adaeze-c1-return-full` | R3 full training | `24055019` | `PENDING` | Dependency cleared; Priority 1545; estimated start `2026-11-18 16:36:05` |
| `24055024` | `adaeze-c1-return-paired` | R0 zeroed tokens | `24055020` | `PENDING` | Awaiting R0 training completion |
| `24055025` | `adaeze-c1-return-paired` | R0 fusion bypass | `24055020` | `PENDING` | Awaiting R0 training completion |
| `24055026` | `adaeze-c1-return-paired` | R1 zeroed tokens | `24055021` | `PENDING` | Awaiting R1 training completion |
| `24055027` | `adaeze-c1-return-paired` | R1 fusion bypass | `24055021` | `PENDING` | Awaiting R1 training completion |
| `24055028` | `adaeze-c1-return-paired` | R2 zeroed tokens | `24055022` | `PENDING` | Awaiting R2 training completion |
| `24055029` | `adaeze-c1-return-paired` | R2 fusion bypass | `24055022` | `PENDING` | Awaiting R2 training completion |
| `24055030` | `adaeze-c1-return-paired` | R3 zeroed tokens | `24055023` | `PENDING` | Awaiting R3 training completion |
| `24055031` | `adaeze-c1-return-paired` | R3 fusion bypass | `24055023` | `PENDING` | Awaiting R3 training completion |

## 5. Queue and operational rules

- **Hands-off directive:** Do not cancel, release, duplicate, modify, or relabel jobs `24055020`–`24055031`. Canceling resets accumulated queue age (currently 333 points).
- **Queued status:** Pending jobs are not results.
- **Reporting verification:** Query live state before making claims:
  ```bash
  cat /work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/job_ids.env
  cat /work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/source_commit.txt
  squeue -u ailodianya -o "%.18i %.28j %.12P %.10T %.10M %.10l %.25R"
  sacct -j 24055019,24055020,24055021,24055022,24055023,24055024,24055025,24055026,24055027,24055028,24055029,24055030,24055031 -X \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End
  ```
