# Screened return experiments and frozen-input handoff — November 13, 2026

Status: **controlled runs prepared.** The [Delta smoke and metrics replay](../../analysis/text_value/ADAEZE_RETURN_SPEED_SMOKE.md)
passed on Adaeze’s predecessor branch. The combined code and frozen-input
return experiments have no full training results. Follow [EXPERIMENTS.md](../../EXPERIMENTS.md)
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
same test event IDs. Report return MAE/MSE and close directional hit
rates with eligible counts, real versus shuffled/no-article controls, and
paired date-block uncertainty. A smoke run only checks execution and reload.
If the four-run budget is unavailable, prioritize R3 versus screened price as
a labeled package comparison and R2 versus R3 as a single-factor check. Do not
claim effects for unrun comparisons. Report each run’s objective values for
debugging. For R0 versus R1, the task losses use different direction weights.
Use common bp errors and hit rates for performance comparisons. Validation
selects each checkpoint using its configured training objective. Before a
price-versus-return comparison, transform each price prediction to a bp return
using its own last observed close, and verify the event IDs, windows, and
article content match.

## Reproducing the input freeze and audit

The Delta batch entrypoint is `scripts/delta/nicole_c1_return_input_audit.sbatch`.
It uses the CPU on a scheduled compute node. The commands below show the same
steps for a CPU worker that can read the corpus. The copy operation **fails if the versioned destination
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

The audit reruns the Reuters dateline screen on retained summaries and verifies
all event transforms against the last observed close,
price reconstruction, Stage-1 return windows, split coverage, event exclusions,
training-only loss scale, and direction-eligible counts. It fails on incomplete
inputs by default. Save its small JSON/CSV and the snapshot's `snapshot.json`
as the version record. Send the snapshot `content_sha256`, audit paths and
summary, and any failure to the group. Do not silently replace a failed
snapshot; investigate and create `v2` if a correction is justified.

## Later GPU order, after access returns

Use one clean checkout and export its commit and the audited snapshot fingerprint
for training and paired jobs. The scripts check both before loading a model.

```bash
cd /work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/repo
mkdir -p /work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/logs
export EXPECTED_CODE_COMMIT=$(git rev-parse HEAD)
export C1_SNAPSHOT_SHA256=$(python -c 'import json; from pathlib import Path; p=Path("/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1/snapshot.json"); print(json.loads(p.read_text())["content_sha256"])')
```

1. Submit `scripts/delta/nicole_c1_return_smoke.sbatch`; inspect all stages,
   checkpoint reload, resolved config, and input audit. Then submit
   `RUN_KIND=smoke-v1 sbatch scripts/delta/nicole_c1_return_paired.sbatch`.
2. Once smoke passes, submit each variant separately as allocation allows, for
   example `VARIANT=R0 sbatch scripts/delta/nicole_c1_return_full.sbatch`. Repeat
   for R1, R2 and R3. Each has a fresh output path.
   R3 writes to `full-v1` and is the package-comparison return run.
3. For each run, set `RUN_KIND` to its output directory name and run
   `scripts/delta/nicole_c1_return_paired.sbatch`. Save job IDs, selected
   checkpoint, config, hashes, paired outputs, metrics, findings, and
   limitations in a linked result note. For each adjacent pair, run the CPU
   analyzer with the new variant as `--predictions` and the preceding variant
   as `--reference`. It verifies aligned events, targets, donor IDs, article
   hashes, and the dataset fingerprint before computing error differences and
   paired date-block intervals.

```bash
python scripts/analyze_return_predictions.py \
  --predictions ../paired-ablation-r1-v1/zeroed_article_tokens.npz \
  --reference ../paired-ablation-r0-v1/zeroed_article_tokens.npz \
  --out ../r1-vs-r0-v1.json
```

The optional last-value module preserves the random state when initialized,
so shared model weights and starting random draws agree for R2 and R3.
Consider repeated seeds if a result is small or unstable. Q-Former and MoE
are follow-up experiments on the same snapshot, each changing one architecture
choice against R3.
