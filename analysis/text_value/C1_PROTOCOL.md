# C1 training and text-control handoff

**Owner:** Nicole Lyu. **Status:** C1 loader and CPU checks complete; one-GPU
three-stage smoke job `23701798` submitted on November 10, 2026. Full C1
benchmark and paired prediction results remain pending until a usable full-model
checkpoint is verified.

## What the code now does

`data/c1_protocol.py` reads the committed C1 manifest, translates its Windows
event paths to the shared article folders, and emits stable IDs such as
`1/20080110.txt`. The seven returned values preserve the model's batch contract.
Every stage uses one sorted, exact-deduplicated price timeline and one scaler
fitted only to bars covered by emitted **training** events. Stage-1 sliding
windows must fit entirely within the corresponding train or validation event
coverage. Training negatives are drawn only from C1 train articles; validation
and test negatives remain within their own partitions. The loader checks exact
input-plus-target bar coverage for train/validation/test overlap and fails if
it finds any. Missing summaries, sentence reports, windows, or negative reports
fail by default; opt-in exclusions are counted and listed in the audit.

The timing assumption is `category4_1400_strict_before`: category-4 events
are placed at 14:00 and the input ends strictly before the assumed release;
other categories are placed at 08:30. **Actual release times, CSV timezone,
and whether bar timestamps label starts or ends are still unverified.** This
is a sensitivity protocol, not verified historical publication timing.

## Real-input audit

The [audit summary](c1_audit_summary.json) and [emitted-window table](c1_emitted_windows.csv)
were generated on Delta from the shared inputs. The table has individual IDs
and bar indices, but no article content or raw price values.

| Check | Result |
|---|---:|
| C1 train / validation / test events emitted | 6,167 / 2,500 / 2,921 |
| Missing or excluded events | 0 |
| Cross-partition shared input-plus-target bars | 0 (loader check passed) |
| Price rows / exact duplicates / clean rows | 331,256 / 162 / 331,094 |
| Unique train bars fitted by scaler | 180,151 |
| Stage-1 train / validation windows, stride 10 | 15,945 / 6,523 |

Manifest SHA-256: `7da0d5666ccf54fa41f99f2b947c83344acc1fd4c40c4017127e3a5bb87276fa`.
Price CSV SHA-256: `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`.
Both hashes match [Nicole's earlier audit](data_audit_summary.json). The
shared prices remain unchanged; the loader removes exact duplicates in memory.

## Reproduce

Use the team's PyTorch environment and access to the shared articles and
price file. The exact remote paths and job request are in
[`scripts/delta/nicole_c1_smoke.sbatch`](../../scripts/delta/nicole_c1_smoke.sbatch).
Run the CPU-only emitted-window audit from this repository root:

```sh
python scripts/audit_c1_protocol.py \
  --event-dir /work/hdd/bjjg/gs-fuse/data/event \
  --series /work/hdd/bjjg/gs-fuse/data/series/SP500.csv \
  --out outputs/c1_audit_summary.json \
  --windows-csv outputs/c1_emitted_windows.csv
```

Train from a **fresh** output directory with `train.py --c1-manifest
splits/manifests/purged_split_c1_macro_eras.json` and the same article/series
paths. `smoke_train.py` accepts the same C1 option for a bounded end-to-end
validation. C1 training refuses to silently resume old checkpoints; explicit
resume requires independently checking their training history.

After a compatible C1 checkpoint exists, run paired controls with
`scripts/run_text_controls.py --c1-manifest ...`. That runner uses the same
event IDs, price windows, and training-only scaler, and saves correct,
seeded-shuffled, and article-free predictions for each selected event.
`--checkpoint-training-split` must describe the checkpoint's actual history.
Keep the two available article-free interventions separate. This inference
helper does not alter gate supervision during training.

DLinear still uses its separate continuous-window baseline protocol. Its old
scores should not be ranked against this event-based C1 evaluation without
matching samples and scaling.

## Checks and limits

- Four new synthetic C1 regression tests passed on Delta's shared PyTorch 2.8
  environment. They cover paths, deduplication, exclusions, the batch contract,
  train-only scaling, role-specific negatives, and Stage-1 windows.
- The existing 25 focused text-value tests also passed with `--require-torch`.
- The real-input CPU audit completed with no exclusions or cross-split bars.
- These checks are not a full-model accuracy result. The prior shared
  `runs/smoke/baseline` log stopped during Stage 2 and saved only a Stage-1
  checkpoint. No completed full-model checkpoint was found in the inspected
  shared Delta run tree or the provided Drive folder.

## Running job

Slurm job `23701798`: one A100 GPU, eight CPUs, 80 GB RAM, one-hour limit,
funded project account `bjjg-delta-gpu`, batch submission to
`gpuA100x4-interactive`. Output directory:
`/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/smoke/run1`.
This is a 16-train-event three-stage **smoke test**, so any resulting scores
will validate execution only and must not be presented as a trained C1
benchmark. Budget reserved: at most one GPU-hour so far; actual usage is
recorded after the job finishes.
