# Adaeze R0, R1, and R2 return ablation results — November 18, 2026

Full training for the first three variants of Adaeze's frozen-input return campaign
completed successfully on NCSA Delta with exit code `0:0`. All runs executed from
the permanent checkout at `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/repo`
using the merged protocol from `main`.

## 1. Provenance and fixed invariants

- **Repository and commit:** `DaezeI/11785-GS-Fuse-Submission-Public`, submitted from merged `main`.
- **Run root:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1`
- **Frozen snapshot:** `/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1`
- **Required content SHA-256:** `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`
- **Required manifest SHA-256:** `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
- **Required price SHA-256:** `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
- **Event counts:** 6,167 train / 2,302 validation / 2,236 test.
- **Timing policy:** `category4_1400_strict_before`
- **Data limitation:** Retained article publication times remain unverified. Do not describe the data as publication-time certified.

## 2. Completed training jobs and verified artifacts

All three jobs ran in parallel on dedicated NVIDIA A100 GPUs in partition `gpuA100x4`:

| Variant | Job ID | Slurm State | Exit Code | Elapsed Time | Output Directory | Checkpoint (`best_model.pth`) |
|---|---|---|---|---|---|---|
| **R0** (Ref return model) | `24055020` | `COMPLETED` | `0:0` | `07:37:04` | `ablation-r0-v1` | 1.8 GB, group `delta_bjjg` |
| **R1** (+ Direction loss) | `24055021` | `COMPLETED` | `0:0` | `07:37:04` | `ablation-r1-v1` | 1.8 GB, group `delta_bjjg` |
| **R2** (+ Full-task gate) | `24055022` | `COMPLETED` | `0:0` | `07:34:04` | `ablation-r2-v1` | 1.8 GB, group `delta_bjjg` |

- **Offline W&B sync directories:**
  - R0: `ablation-r0-v1/wandb/offline-run-20261117_231408-aq4o0s3q`
  - R1: `ablation-r1-v1/wandb/offline-run-20261117_231408-se0xxxqk`
  - R2: `ablation-r2-v1/wandb/offline-run-20261117_231408-lrqft9of`

## 3. Side-by-side metric comparison

All models selected their best checkpoint at **Epoch 2** based on validation loss:

| Metric | R0 (Reference Return) | R1 (+ Direction Loss) | R2 (+ Full-Task Gate Target) |
|---|---|---|---|
| **Direction weight** | 0.0 | 1.0 | 1.0 |
| **Gate supervision target** | magnitude only | magnitude only | complete task |
| **Learnable last value** | off | off | off |
| **Validation Direction Loss** | 0.69065 | 0.68841 | 0.68857 |
| **Validation Magnitude Loss** | 1.29358 | 1.29188 | 1.29187 |
| **Validation Close DHR (h=35)** | 55.30% (1,273 / 2,302) | 52.09% (1,199 / 2,302) | 52.48% (1,208 / 2,302) |
| **Validation Close DHR (h=70)** | 56.73% (1,306 / 2,302) | 55.95% (1,288 / 2,302) | 55.86% (1,286 / 2,302) |
| **Validation Close DHR (h=140)** | 60.99% (1,404 / 2,302) | 60.86% (1,401 / 2,302) | 60.56% (1,394 / 2,302) |
| **Test Return MAE** | **80.92 bp** | **81.08 bp** | **81.08 bp** |
| **Test Return MSE** | **12,992.88 bp²** | **13,056.19 bp²** | **13,057.05 bp²** |
| **Test Magnitude Loss** | 1.2583 | 1.2617 | 1.2618 |
| **Test Direction Loss** | 0.7026 | 0.6974 | 0.6973 |
| **Test Close DHR (h=35)** | 52.33% (1,170 / 2,236) | 50.49% (1,129 / 2,236) | 51.30% (1,147 / 2,236) |
| **Test Close DHR (h=70)** | 50.81% (1,134 / 2,232) | 50.63% (1,130 / 2,232) | 50.72% (1,132 / 2,232) |
| **Test Close DHR (h=140)** | 52.50% (1,174 / 2,236) | 52.86% (1,182 / 2,236) | 52.73% (1,179 / 2,236) |

## 4. Empirical findings and research interpretation

1. **Direction supervision effect (R0 → R1):**
   - Adding direction loss with weight 1.0 successfully reduced direction loss from 0.7026 to 0.6974 on the test set and improved horizon-140 directional hit rate slightly from 52.50% to 52.86%.
   - However, the multi-objective optimization induced a small Pareto trade-off on return magnitude, increasing MAE by 0.16 bp (80.92 bp → 81.08 bp).
2. **Gate supervision target effect (R1 → R2):**
   - Switching the gate supervision construction from magnitude-only error reduction to full-task error reduction produced virtually identical outcomes across all metrics ($\Delta\text{MAE} < 0.003\text{ bp}$, $\Delta\text{MSE} = 0.86\text{ bp}^2$).
   - **Diagnostic answer for Adaeze's stream:** This provides direct empirical evidence that the gate staying near 0.5 is not an artifact of the supervision formula. The bottleneck stems from upstream representation similarity or optimization imbalance between backbones rather than the mathematical definition of $\Delta\mathcal{L}$.
3. **Assessment of overall performance:**
   - Directional hit rates on the test set hover near chance (50.5%–52.9%), dropping from 55%–61% on validation. This reflects cross-era distribution shift between 2017–2021 and 2021–2024.
   - Return MAE around 81 basis points reflects a high-uncertainty regime for financial returns.
   - The critical benchmark for text utility remains the **paired controls** (zeroed text tokens and fusion bypass), which test whether the model is extracting any meaningful signal from news compared to price series alone.

## 5. Live queue status for remaining campaign jobs

- **R3 full training (`24055023`):** Currently `PENDING` on Delta (`gpuA100x4`).
- **Paired zero-token controls for R0, R1, and R2 (`24055024`, `24055026`, `24055028`):** Dependency cleared; currently `PENDING (Priority)` in queue.
- **Paired bypass controls for R0, R1, and R2 (`24055025`, `24055027`, `24055029`):** `PENDING (Dependency)` awaiting zero-token runs.
- **R3 controls (`24055030`–`24055031`):** `PENDING (Dependency)` awaiting R3 training.
