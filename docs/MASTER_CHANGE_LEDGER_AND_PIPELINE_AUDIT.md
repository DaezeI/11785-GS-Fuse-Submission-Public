# GS-Fuse Master Change Ledger & Pipeline Audit

**Last Updated:** November 18, 2026  
**Repository:** `DaezeI/11785-GS-Fuse-Submission-Public`  
**Purpose:** Single unified register documenting every architectural, data, and protocol change made across all team streams, why each was made, code locations, experimental outcomes, and remaining pipeline vulnerabilities.

---

**Evidence correction:** see [audit-first revision](experiments/AUDIT_FIRST_AND_COMPACT_V2.md). Calendar-date screening does not certify intraday availability; historical scheduler notes below are dated observations.

## 1. Master Change Matrix Across All Team Streams

| Pipeline Stage | Original Paper Flaw / Gap | Change Made by Team | Code Location | Owner & PR | Impact & Verified Outcome |
|---|---|---|---|---|---|
| **1. Split & Temporal Leakage** | >880 articles in Val/Test occurred *after* target price windows (future-lookahead leakage). | Constructed date-screened temporal split (**C1**), strictly purging 883 contaminated events (6,167 train / 2,302 val / 2,236 test). | `splits/manifests/purged_split_c1_date_screen_v1.json`, `data/purged_dataloader.py` | Nicole (PR #4, PR #7) | **Verified zero cross-split shared price bars under the stated timing assumption.** Publication times remain unverified; this does not establish why the paper obtained its scores. |
| **2. Clean Text Controls** | Time-series comparison was contaminated with text via `s_vec_residual` post-attention. | Created pure price controls: `zeroed_article_tokens` and `fusion_bypass_ts_inst`. | `model/no_text_control.py`, `scripts/run_text_controls.py` | Nicole (PR #1, PR #4) | **Same-checkpoint inference result:** Bypassing text improves price MSE by 10%–20% (PR #21). Shuffled text gap is +0.0085% (statistically zero). |
| **3. Q-Former Attention** | Tokenizer padding mask was hardcoded to all `False`, forcing cross-attention to attend to `[PAD]` tokens. | Propagated true tokenizer `key_padding_mask` through `_QFormerLayer`. | `model/variants.py`, `model/qformer.py` | Jessica (PR #2, commit `96cc6d7`) | Stopped cross-attention from calculating softmax over padding noise. Verified with unit tests. |
| **4. Modality Gating** | Single softmax gate collapsed to a static 0.5 across all events. | Implemented Mixture-of-Experts (`MoEModalityGate`) with top-$k$ routing and Switch-Transformer load balancing. | `model/variants.py`, `feat/moe-qformer-wandb-cache` | Joye | Architectural alternative to test if expert routing prevents gate collapse. |
| **5. Forecast Objective** | Forecasted raw price levels across macro regimes without stationarity. | Transformed target to basis-point returns (`return_bp`), added Huber loss + directional penalty. | `data/forecast_task.py`, `smoke_train.py`, `train.py` | Adaeze (PR #8) | Standardized task across time, eliminated macro price drift bias. |
| **6. Gate Supervision** | Gate was trained only on magnitude error reduction ($\Delta \mathcal{L}_{\text{mag}}$), ignoring direction. | Supervised gate on full-task error ($\Delta \mathcal{L}_{\text{total}}$), created R0–R3 ablation campaign. | `data/forecast_task.py`, `scripts/delta/adaeze_c1_return_launch.sh` | Adaeze (PR #11, PR #12, PR #13) | **Ablation completed (R0–R2):** Found similar scores for the tested R1/R2 runs; this does not identify a universal cause of the performance ceiling ($\Delta < 0.003 \text{ bp}$). |
| **7. Learnable Last Value** | Non-stationary returns suffer from high-dimensional latent drift. | Created `LearnableLastValueResidual` bypass wire anchoring 90% of prediction directly to the last known point. | `model/gs_fuse.py` (line 255) | Adaeze (R3 variant) | Queued as Variant R3 (`24055023`) on Delta `gpuA100x4`. |

---

## 2. Mechanical Breakdown of Critical Architectural Lessons

### A. Distinguish inference controls from gate training
Historical post-attention price features contain text information. The
current gate-training comparison uses pre-attention TS features; the
inference-only no-text helper is a separate intervention and does not itself
repair training supervision. Verify the code/checkpoint of each result.

### B. What the text controls establish
Matched and shuffled article errors were nearly equal for the cited screened
price checkpoints. Fusion bypass improved error in several of those
same-checkpoint comparisons. This supports a bounded observation about
article-match sensitivity; it does not prove that all text is useless or
identify generic embedding centroids as the unique mechanism. Independent
price-only training and persistence are separate controls; the existing
result packs report their measured outcomes and intervals.

### C. The R0 $\to$ R1 $\to$ R2 Return Ablation Finding
On the clean date-screened return task (6,167 train, 2,302 val, 2,236 test):
* **R0 (Base return):** MAE 80.92 bp, DHR 50.8%–52.5%
* **R1 (+ Direction loss):** MAE 81.08 bp, DHR 50.6%–52.9%
* **R2 (+ Full-task gate):** MAE 81.08 bp, DHR 50.7%–52.7%
* **Conclusion:** The difference between R1 and R2 is $< 0.003 \text{ bp}$. Rewriting gate supervision does not move the needle. The experiment does not isolate a unique bottleneck.

---

## 3. Systematic Pipeline Audit: Vulnerabilities & Next Improvement Points

| Vulnerability / Limitation | Current State | Why It Limits Performance | Recommended Future Improvement |
|---|---|---|---|
| **Intraday Publication Times** | Articles have calendar dates, but lack minute/second timestamps. | Relies on `category4_1400_strict_before` heuristic. Some afternoon news could technically leak into same-day morning bars. | State clearly in final report as an unverified publication-time limitation. |
| **Frozen Llama Embeddings** | Historical reference keeps Llama frozen; the campaign branch adds opt-in LoRA with cache/gradient/replay checks. | Llama was trained on general web text, not price alpha. Gradients cannot tune text tokens to emphasize market-moving words. | Apply LoRA or financial headline classification/sentiment pooling rather than raw sequence cross-attention. |
| **Short Time-Series Window** | Only 35 historical 5-minute bars (< 3 hours of trading). | Modern financial transformers use 200–500 bars to capture multi-day volatility and support/resistance levels. | Expand historical input context length to 140–280 bars. |
| **Lack of Pure Statistical Baseline** | Focus was on multimodal ablation. | Need an external anchor to compare against standard time-series models. | Benchmark against pure DLinear, PatchTST, or ARIMA on the identical C1 return split. |

---

## 4. Delta Submission and Provenance Register

* **Campaign Root:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1`
* **Frozen Snapshot SHA-256:** `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`
* **Active Jobs:**
  * Smoke: `24055019` (`COMPLETED`, `0:0`, 45m23s)
  * R0 Training: `24055020` (`COMPLETED`, `0:0`, 7h37m, MAE 80.92 bp)
  * R1 Training: `24055021` (`COMPLETED`, `0:0`, 7h37m, MAE 81.08 bp)
  * R2 Training: `24055022` (`COMPLETED`, `0:0`, 7h34m, MAE 81.08 bp)
  * R3 Training: `24055023` (`PENDING` in `gpuA100x4`)
  * Paired Controls: `24055024`–`24055031` (`PENDING`)
