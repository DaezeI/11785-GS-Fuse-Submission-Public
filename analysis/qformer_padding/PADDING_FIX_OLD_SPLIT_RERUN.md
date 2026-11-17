# Q-Former padding-mask fix: before/after rerun (original split, not C1)

**Owner:** Jessica
**Status:** Q-Former and matched baseline both completed
**Purpose:** historical diagnosis (original positional split). **Not a C1 result.**

## Question

The text padding mask passed to the text attention was hard-coded to all False
(also in the authors' released code), so PAD tokens were treated as real text.
After fixing it (PR #2, PR #5), does the Q-Former forecast better or use the
text gate differently?

## Setup

| Item | Value |
|---|---|
| Code | `main` at `10edc1e` (merge of PR #5; includes PR #2 and PR #5) |
| Where | Delta, `/projects/bjjg/zchen58/gsfuse`, account `bjjg-delta-gpu`, 1x A100 (`gpuA100x4`) |
| Slurm jobs | Smoke `23739724` (completed); Q-Former `23739926` (completed); baseline `23739927` (completed) |
| Scripts | [`scripts/delta/jessica_fixedmask_full.sbatch`](../../scripts/delta/jessica_fixedmask_full.sbatch), [`scripts/delta/jessica_fixedmask_smoke.sbatch`](../../scripts/delta/jessica_fixedmask_smoke.sbatch). The submitted copies had no `umask 007` line; it was added afterwards. |
| Outputs on Delta | Baseline checkpoint `/projects/bjjg/zchen58/gsfuse/outputs/fixedmask/baseline/best_model.pth`; log `/projects/bjjg/zchen58/gsfuse/logs/full-jessica-baseline-23739927.out`; W&B offline run `outputs/fixedmask/baseline/wandb/offline-run-20261112_204900-1wdpdgh2`. Q-Former checkpoint `/projects/bjjg/zchen58/gsfuse/outputs/fixedmask/qformer/best_model.pth`; log `/projects/bjjg/zchen58/gsfuse/logs/full-jessica-qformer-23739926.out`; W&B offline run `outputs/fixedmask/qformer/wandb/offline-run-20261111_052313-ujs10xwq` |
| Checkpoint history | Trained from scratch in a fresh output directory (no Stage-1/2 resume) |
| Split | Original positional 60/20/20 split (TRAIN 6,988 / VALI 2,330 / TEST 2,329) |
| Data version | Original GS-Fuse release via `/projects/bjjg/gs-fuse/repo/GS-FUSE` symlinks; manifest/price/article hashes PENDING |
| Backbones | `pretrained/Llama-3.2-1B` (fp32, `--no-4bit`), `pretrained/MOMENT-1-large`, TF32 on |
| Text cache | `cache/text_inst_llama1b_tf32.pt` (26,917 embeddings) and `data/cache_llama_emb/` |
| Training | batch 8; 2 epochs per stage; stage-1 sliding-window stride 1 (198,579 train windows) |
| Model | `--fusion-type qformer --gate-type softmax`, 16 queries, 2 layers |
| Seeds | Not fixed (single run) |
| Before (reference) | Joye's midterm Q-Former run, same settings (W&B run `qformer`, project `11785-GS-Fuse`) |
| Only difference | The padding-mask fix |

Command (inside `full.sbatch`, `VARIANT=qformer`):

```sh
python -u train.py --moment-path pretrained/MOMENT-1-large \
  --text-model-path pretrained/Llama-3.2-1B --text-cache cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 8 --fusion-type qformer --gate-type softmax \
  --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 --sw-stride 1 --interpret-samples 200 \
  --output-path outputs/fixedmask/qformer
```

## Results (H = 140)

| Metric | Before (Joye) | After (fixed mask) |
|---|---|---|
| Stage-1 best vali MSE (x1e-3) | 5.11 | 5.14 |
| Stage-2 best vali MSE (x1e-3) | 0.895 | 1.007 |
| **Test MSE (x1e-4)** | **6.33** | **6.32** |
| Vali MSE (x1e-4) | 5.83 | 5.83 |
| Test MAE (x1e-2) | 1.530 | 1.531 |
| Mean text gate by event type | ~0.524-0.528 | ~0.520-0.523 |
| Q-Former attention on padding | 34% | 0% |

Stage 1 uses prices only and is unaffected by the mask, so its match confirms the
setup reproduced Joye's run.

### Baseline (cross-attention fusion, softmax gate), H = 140

| Metric | Before (Joye) | After (fixed mask) |
|---|---|---|
| Test MSE (x1e-4) | 6.33 | 6.32 |
| Vali MSE (x1e-4) | 5.80 | 5.81 |
| Test MAE (x1e-2) | PENDING | 1.523 |
| Mean text gate by event type | 0.524 +/- 0.003 | ~0.521-0.525 |

## Interpretation

The padding bug was real and the fix works: Q-Former attention on padding drops
from 34% to 0%. But forecasts and the gate are essentially unchanged for both
the Q-Former and the baseline (test MSE 6.32 vs 6.33 in each; gate still ~0.52
across event types). So the padding bug is **not** why the gate does not use the
text. The models now attend to real tokens yet forecast the same, consistent
with the shuffled-text control (most of the gain survives shuffled articles) and
with the models relying little on article content. This alone does not prove
that the text is uninformative.

Suggested next focus: on the screened C1 return snapshot, run the Q-Former as a single
architecture change against R3, and measure query-attention and query-output
similarity.


## Limitations

- Single unseeded run; a 0.1% MSE difference is within expected seed variance.
- Original split, which has known window overlap between partitions.
- The before numbers come from a different machine (Joye's run), same settings.


## Pending

- W&B links after `wandb sync` of `outputs/fixedmask/qformer/wandb/offline-run-20261111_052313-ujs10xwq` and `outputs/fixedmask/baseline/wandb/offline-run-20261112_204900-1wdpdgh2`.
- corr(gate, text benefit) and text gain from the full log.

Checkpoints and full logs stay on Delta under `/projects/bjjg/zchen58/gsfuse/` (group `delta_bjjg`; team read access: PENDING verification by a teammate).
