# C1 training and text-control handoff

**Owner:** Nicole Lyu. **Status:** C1 loader, CPU checks, three-stage smoke
training, and paired text-control smoke inference complete on November 10,
2026. The [smoke results](C1_SMOKE_RESULTS.md) are execution checks, not a
full C1 benchmark.

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

The audit and smoke job selected `category4_1400_strict_before`: category-4
events are placed at 14:00 and the input ends strictly before that assumed
time; other categories are placed at 08:30. C1 commands now require a timing
policy explicitly. **Actual release times, CSV timezone, and whether bar
timestamps label starts or ends are still unverified.** This is a sensitivity
protocol, not verified historical publication timing.

The [upstream CAMEF README's category list](https://github.com/lakebodhi/CAMEF#-dataset-access)
describes six categories, while this C1 manifest contains **12**. Its first
six category labels also do not match the team's actual Delta article tree.
Keyword counts on one private summary from each of categories 1–6 (without
publishing article text) indicate: 1 claims, 2 employment, 3 GDP, 4 FOMC
minutes, 5 CPI, 6 PPI.
For example, the category-4 summary dated 2008-02-20 mentions FOMC five
times and GDP zero times. The upstream
[loader's 14:00 branch](https://github.com/lakebodhi/CAMEF/blob/publish-camef-code/CAMEF/data/dataloader.py#L95-L103)
tests the `event_id` setting, not the parsed category. An official
[2024 Federal Reserve minutes release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20240821a.htm)
states 14:00, and a
[2024 BEA schedule](https://www.bea.gov/system/files/2024-06/2024-News-Release-Schedule.pdf)
lists GDP advance estimates at 08:30. This supports the *direction* of the
selected sensitivity policy for those sampled event types. One sample per
category and 2024 examples do not verify all historical release times. A
per-event publication-time table remains a separate task before scientific
C1 claims.

Categories 7–12 account for 5,240/6,167 training, 2,255/2,500 validation,
and 2,578/2,921 test entries. The manifest provides their dates and category
IDs but no article publication time. Assigning all these articles 08:30 can
make text available to the model before it was actually public. The zero
cross-partition bar overlap check does not address that availability risk.

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
  --timing-policy category4_1400_strict_before \
  --out outputs/c1_audit_summary.json \
  --windows-csv outputs/c1_emitted_windows.csv
```

Train from a **fresh** output directory with `train.py --c1-manifest
splits/manifests/purged_split_c1_macro_eras.json --c1-timing-policy
<explicit-policy>` and the same article/series paths. `smoke_train.py`
accepts the same C1 options for a bounded end-to-end
validation. C1 training refuses to silently resume old checkpoints; explicit
resume requires independently checking their training history.

The paired smoke controls ran from the new C1 checkpoint with
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

- Five new synthetic C1 regression tests passed on Delta's shared PyTorch 2.8
  environment. They cover paths, deduplication, exclusions, the batch contract,
  train-only scaling, role-specific negatives, and Stage-1 windows.
- The existing 25 focused text-value tests also passed with `--require-torch`.
- The real-input CPU audit completed with no exclusions or cross-split bars.
- These checks and the 16-example smoke checkpoint are not a full C1
  benchmark. The prior shared `runs/smoke/baseline` log stopped during Stage 2
  and saved only a Stage-1 checkpoint; the new smoke job saved a complete
  three-stage checkpoint in Nicole's separate Delta workspace.

## Completed batch jobs

Slurm jobs `23701798` (three-stage training, 12m38s) and `23701969`
(paired inference, 7m58s) both completed with exit code zero. Each used one
A100 GPU, eight CPUs, and 80 GB RAM under `bjjg-delta-gpu`; the jobs ran
sequentially. Combined use was 20m36s, or 0.343 GPU-hours, within the
eight-GPU-hour cap. The checkpoint and full outputs stay at
`/work/hdd/bjjg/gs-fuse/nicole/c1-nov10/`. Only derived public artifacts are
published in [the smoke results](C1_SMOKE_RESULTS.md).
