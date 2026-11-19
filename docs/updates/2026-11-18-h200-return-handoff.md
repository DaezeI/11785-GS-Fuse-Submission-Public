# H200 Return Campaign Handoff & Paired Evaluation Guide

**Date:** November 18, 2026  
**Target Resource:** Standalone H200 GPU / ABCI PBS Node  
**Reference Commit:** `e051195` (Delta R0–R3 campaign source) / `b7452fc` (Main)  
**Authors:** Adaeze Ilodianya (Return Lead), Nicole Lyu (Experiment Coordination)

---

## 1. Overview & Objectives

Full three-stage training for return variants **R0** (base return), **R1** (+ directional loss), and **R2** (+ full-task gate target) completed successfully on Delta with exit code `0:0`, generating verified 1.8 GB checkpoints on 2,302 validation and 2,236 test events.

To eliminate queue latency and finish the evaluation matrix rapidly, this handoff enables executing all paired text controls (`zeroed_article_tokens` and `fusion_bypass_ts_inst`) on Nicole's H200 GPU resource alongside the active **R3** (`full-v1`) run (`2988131.pbs1`).

---

## 2. Checkpoint Provenance and Paths

### A. Source Locations on Delta
* **Campaign Root:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1`
* **R0 Checkpoint:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r0-v1/best_model.pth`
* **R1 Checkpoint:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r1-v1/best_model.pth`
* **R2 Checkpoint:** `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r2-v1/best_model.pth`

Each directory contains its matching `resolved_config.json`, `selected_test_return_metrics.json`, and `full_training_log.txt`.

### B. SHA-256 Verification Command (Run on Delta / Staging Host)
```bash
sha256sum \
  /work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r0-v1/best_model.pth \
  /work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r1-v1/best_model.pth \
  /work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1/ablation-r2-v1/best_model.pth
```

### C. Dataset & Text Cache Hashes
* **Manifest (`c1-screened-return-v1`):** `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
* **Price Series (`SP500.csv`):** `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`
* **Snapshot SHA-256:** `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`
* **Text Cache (`text_inst_llama1b_tf32.pt`):** Verified full-precision TF32 cache (no 4-bit quantization).

---

## 3. Turnkey Execution on H200

On the H200 host, run the turnkey evaluation script from the repository root:

```bash
# Optional: Set paths if different from defaults
export SNAPSHOT=data/snapshots/c1-screened-return-v1
export CHECKPOINT_ROOT=runs/adaeze/return-r0-r3-v1
export OUTPUT_ROOT=runs/h200/paired-return-evaluations

# Run both controls across R0, R1, and R2 (takes ~15 minutes per variant on H200)
bash scripts/research_cycle/run_r0_r2_h200_paired.sh
```

### What the Script Produces for Each Variant
1. `<OUTPUT_ROOT>/<VARIANT>/zeroed_article_tokens.npz` (Inference with zeroed text embeddings)
2. `<OUTPUT_ROOT>/<VARIANT>/zeroed_article_tokens.analysis.json` (1,000-replicate date-block bootstrap CI)
3. `<OUTPUT_ROOT>/<VARIANT>/fusion_bypass_ts_inst.npz` (Inference bypassing cross-attention and gate)
4. `<OUTPUT_ROOT>/<VARIANT>/fusion_bypass_ts_inst.analysis.json` (1,000-replicate date-block bootstrap CI)

---

## 4. Comparing the Complete R0–R3 Matrix

Once R3 finishes training via `2988131.pbs1` and its paired controls run:
1. Compare matched real-text test MSE/MAE against shuffled text across all 4 variants ($H \in \{35, 70, 140\}$).
2. Measure whether the `LearnableLastValueResidual` in R3 broke the ~80.9 bp MAE persistence floor.
3. Check whether the fusion bypass gain persists on returns (i.e. whether text fusion acted as additive noise on returns as it did on price levels).
4. Run the macro announcement subgroup evaluation:
   ```bash
   python scripts/evaluate_macro_subgroup.py --npz-path <OUTPUT_ROOT>/full-v1/zeroed_article_tokens.npz
   ```
