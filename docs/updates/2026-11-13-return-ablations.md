# Screened return experiments and frozen-input handoff — November 13, 2026

Status: **code prepared; input snapshot and full CPU audit pending; no return GPU
smoke or full result exists.** Follow [EXPERIMENTS.md](../../EXPERIMENTS.md)
for result records. The original data, original C1 manifest, queued screened
price run, and historical outputs are preserved. Do not edit them for this work.

## Why these runs exist

The queued screened **price** run versus R3 asks whether Adaeze's *whole return
redesign* helps. It changes target units, forecast loss, gate supervision, and
last-value behavior together. It cannot identify which component helped.

The following staged comparison isolates choices **within the return task**.
All four variants use the exact same frozen screened manifest, price and article
snapshot, timing rule, training/validation/test events, seed, code commit,
backbones, architecture, precision, optimizer, training budget, and paired test
protocol. `scripts/delta/nicole_c1_return_full.sbatch` selects the variant.

| Run | Direction weight in forecast task | Gate target loss | Learnable last value | Compare with | Question |
| --- | ---: | --- | --- | --- | --- |
| R0 (`ablation-r0-v1`) | 0 | Huber magnitude | off | screened price (package only) | Reference return model without the three optional choices |
| R1 (`ablation-r1-v1`) | 1 | Huber magnitude | off | R0 | Does adding the directional term to the forecast objective help? |
| R2 (`ablation-r2-v1`) | 1 | complete Huber plus direction | off | R1 | Does supervising the gate with the complete task help? |
| R3 (`full-v1`) | 1 | complete Huber plus direction | on | R2 | Does the learnable last-value shortcut help? |

The direction term changes the forecast objective wherever that objective is
used in the three training stages. R1 keeps the gate target magnitude-only, so
R0→R1 does not simultaneously change the gate target. R2 changes only the
gate's supervision target; forecast loss and diagnostic task losses remain
complete. R3 is the frozen return design from Adaeze's handoff. The raw
time-series residual stays on for all four. The shared screened subset is a
data-validity decision; do not use suspect-date articles as a supposed clean
baseline. Retained publication times remain unverified.

Interpret a comparison only after full training and paired evaluation on the
same test event IDs. Report task loss, return MAE/MSE, close directional hit
rates with eligible counts, real versus shuffled/no-article controls, and
paired date-block uncertainty. A smoke run only checks execution and reload.
If the four-run budget is unavailable, prioritize R3 versus screened price as
a labeled package comparison and R2 versus R3 as a single-factor check. Do not
claim effects for unrun comparisons.

## One-time CPU input freeze and audit

Run these commands on a CPU worker with access to the full article corpus and
the Delta paths below. The copy operation **fails if the versioned destination
already exists**. It only reads the source directories. The snapshot contains
copies of the screened manifest, SP500.csv, and every article folder referenced
by that manifest, plus `snapshot.json` with SHA-256 fingerprints for all copied
files. Keep the old data paths intact. The existing local backup of the price
file and manifest is documented separately in the return integration handoff.

```bash
cd /work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/repo
umask 007
python scripts/freeze_c1_inputs.py \
  --manifest splits/manifests/purged_split_c1_date_screen_v1.json \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series /work/hdd/bjjg/gs-fuse/data/series/SP500.csv \
  --out /work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1
python scripts/audit_c1_protocol.py \
  --snapshot-root /work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1 \
  --forecast-task return_bp \
  --timing-policy category4_1400_strict_before \
  --out /work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/return-input-audit-v1.json \
  --windows-csv /work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/return-input-windows-v1.csv
```

The audit verifies all event return transforms against the last observed close,
price reconstruction, Stage-1 return windows, split coverage, event exclusions,
training-only loss scale, and direction-eligible counts. It fails on incomplete
inputs by default. Save its small JSON/CSV and the snapshot's `snapshot.json`
as the version record. Send the snapshot `content_sha256`, audit paths and
summary, and any failure to the group. Do not silently replace a failed
snapshot; investigate and create `v2` if a correction is justified.

## Later GPU order, after access returns

The code checkout must be clean and fixed at one commit. Export that commit as
`EXPECTED_CODE_COMMIT` and the snapshot's `content_sha256` as
`C1_SNAPSHOT_SHA256` for **every** training and paired job. The scripts verify
them before loading the model. Create the run `logs` directory first.

1. Submit `scripts/delta/nicole_c1_return_smoke.sbatch`; inspect all stages,
   checkpoint reload, resolved config, and input audit. Then submit
   `RUN_KIND=smoke-v1 sbatch scripts/delta/nicole_c1_return_paired.sbatch`.
2. Once smoke passes, run `VARIANT=R0`, `R1`, `R2`, and `R3` via
   `scripts/delta/nicole_c1_return_full.sbatch`. Each has a fresh output path;
   R3 writes to `full-v1` and is the package-comparison return run.
3. For each run, set `RUN_KIND` to its output directory name and run
   `scripts/delta/nicole_c1_return_paired.sbatch`. Save job IDs, selected
   checkpoint, config, hashes, paired outputs, metrics, findings, and
   limitations in a linked result note. Do not interpret before these exist.

The scripts use the same frozen inputs and code, but changing random-number
consumption when the last-value module is enabled can affect stochastic
training. Report that limitation; consider repeated seeds if the result is
small or unstable. Q-Former and MoE are separate later experiments on the same
snapshot, each changing one architecture choice against R3.
