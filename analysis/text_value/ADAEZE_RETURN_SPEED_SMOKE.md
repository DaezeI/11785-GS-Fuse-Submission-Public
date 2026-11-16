# Adaeze return-task speed smoke

## Question

Do two runtime-only changes let the screened-C1 return-task smoke run complete
without changing its mathematical or evaluation contract?

## Comparison and intended change

- Reference: PR #8 return-task code before this focused patch.
- Package under test: anomaly detection is opt-in in `train.py`, and Stage 3 no
  longer forces Python garbage collection plus an empty CUDA cache every step.
- Checkpoint format: new checkpoints avoid a second copy of the text backbone;
  the legacy `llm` key remains accepted when loading old checkpoints.

This is a multi-change operational package. It cannot attribute timing gains to
one component and is not a model-quality comparison.

## Fixed settings

- Manifest: `splits/manifests/purged_split_c1_date_screen_v1.json`
- Timing policy: `category4_1400_strict_before`
- Task: `return_bp`; seed 0; raw-TS residual and learnable last value enabled
- 16 training events, 8 validation/test events; 32/16 Stage-1 windows
- One epoch per stage; batch size 2; cross-attention plus softmax gate
- One A100, TF32, offline W&B, local pretrained MOMENT and Llama-3.2-1B

## Status and provenance

Status: **PLANNED — not yet a result**.

- Owner: Adaeze
- Branch: `adaeze/pr8-speed-smoke`
- Source: PR #8 (`nicole/return-task-integration`)
- Slurm script: `scripts/delta/adaeze_c1_return_speed_smoke.sbatch`
- Delta output: `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-speed-pr8/smoke-speed-v1`
- Commit, job ID, command, checks, and result: **PENDING**

## Limitations

This 16-event smoke test is not a full C1 benchmark. The screened manifest
removes known date conflicts but does not verify publication times for retained
news, so it cannot support publication-time-certified claims.
