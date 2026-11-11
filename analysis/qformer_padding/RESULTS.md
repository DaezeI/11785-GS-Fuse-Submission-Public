# Q-Former padding-mask fix: before/after rerun

**Owner:** Jessica
**Status:** Q-Former completed; matched baseline queued
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
| Slurm jobs | Q-Former `23739926` (completed); baseline `23739927` (queued) |
| Split | Original positional 60/20/20 split (TRAIN 6,988 / VALI 2,330 / TEST 2,329) |
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

## Interpretation

The padding bug was real and the fix works: Q-Former attention on padding drops
from 34% to 0%. But forecasts and the gate are essentially unchanged (test MSE
6.32 vs 6.33; gate still ~0.52 across event types). So the padding bug is **not**
why the gate does not use the text. The model now attends to real tokens yet
forecasts the same, consistent with the shuffled-text control (most of the gain
survives shuffled articles): the model barely depends on article content.

Suggested next focus: event timing alignment (all events stamped 08:30) and the
gate target (sigma(0) = 0.5 when text does not help), evaluated on C1.

## Limitations

- Single unseeded run; a 0.1% MSE difference is within expected seed variance.
- Original split, which has known window overlap between partitions.
- The before numbers come from a different machine (Joye's run), same settings.
- Matched baseline rerun with the fix is still queued (job `23739927`).

## Pending

- Baseline before/after once job `23739927` finishes.
- W&B link after `wandb sync` of `outputs/fixedmask/qformer/wandb/offline-run-20261111_052313-ujs10xwq`.
- corr(gate, text benefit) and text gain from the full log.

Checkpoints and full logs stay on Delta under `/projects/bjjg/zchen58/gsfuse/`.
