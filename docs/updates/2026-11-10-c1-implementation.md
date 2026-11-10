# C1 implementation and Delta handoff — November 10, 2026

Nicole's [C1 implementation PR](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/4)
connects the committed C1 manifest to GS-Fuse training, Stage-1 sliding
windows, and paired text controls. It preserves the old entrypoint behavior
unless the C1 option is selected. The implementation, tests, exact emitted
windows, and reproduction instructions are in the
[technical handoff](../../analysis/text_value/C1_PROTOCOL.md).

## Verified today

- Noninteractive Delta SSH reached `dt-login04`. The funded project allocation
  had 987 hours remaining when checked. No jobs were queued under Nicole's
  username before this task submitted its own job.
- Shared Delta input inspection found the original SP500 price CSV, CAMEF
  article tree, local Llama-3.2-1B and MOMENT-1-large backbones, and a
  precomputed text cache. The existing smoke run saved only a Stage-1
  checkpoint; its log stopped during Stage 2. A completed full GS-Fuse
  checkpoint was not found in the inspected shared run tree.
- A CPU audit using the **actual new loader** emitted all C1 events:
  6,167 train, 2,500 validation, 2,921 test, with zero exclusions and zero
  cross-partition full-window bar overlap. The price and manifest hashes
  match Nicole's earlier published audit. The scaler fitted 180,151 unique
  train-covered bars. Stage-1 emitted 15,945 training and 6,523 validation
  windows at stride 10.
- Four new C1 regression tests and the existing 25 focused text-control tests
  passed in the Delta PyTorch 2.8 environment. This verifies the loader and
  CPU inference helper contracts; it is not full-model performance evidence.
- The provided team Drive folder is accessible and contains documents. A
  search there did not find a trained GS-Fuse checkpoint. GitHub remains the
  distribution point for code and small derived records.

## Running work

Slurm job `23701798` is a one-GPU, one-hour, 16-training-event, three-stage
smoke test using a separate personal checkout and output directory. It was
submitted to `gpuA100x4-interactive` with `sbatch` under
`bjjg-delta-gpu`. The reserved amount is one GPU-hour, within Nicole's
eight-hour task cap. Its purpose is to establish an end-to-end checkpoint
and surface runtime problems; any score from 16 training examples is not a
C1 benchmark. A second one-GPU batch script is prepared for eight fixed
test events, three shuffle seeds, and both article-free inference modes if
the smoke checkpoint completes.

## Next handoffs

1. **Nicole:** inspect job `23701798` and account for actual GPU time. If it
   saves a full-model checkpoint, run the paired smoke evaluation and publish
   individual derived predictions and analysis. Fix any runtime issue on
   this same PR with a regression test when appropriate.
2. **Joye:** a common full C1 baseline still needs a planned training budget
   and a separate, provenance-checked checkpoint. The 16-example smoke
   checkpoint must not serve as that baseline.
3. **Team/TA:** confirm actual article release times and the price CSV's
   timezone and bar-label convention. The current C1 timing is an explicit
   `category4_1400_strict_before` assumption. It is not verified release
   metadata.
4. **Adaeze/Joye/Jessica:** continue the separately owned gate, MoE, and
   Q-Former diagnostics; PR #4 does not change those model functions.
