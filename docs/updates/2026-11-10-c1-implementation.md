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
- Five new C1 regression tests and the existing 25 focused text-control tests
  passed in the Delta PyTorch 2.8 environment. This verifies the loader and
  CPU inference helper contracts; it is not full-model performance evidence.
- The provided team Drive folder is accessible and contains documents. A
  search there did not find a trained GS-Fuse checkpoint. GitHub remains the
  distribution point for code and small derived records.

## Completed GPU smoke work

Slurm job `23701798` completed all three training stages and saved a
full-model smoke checkpoint. Job `23701969` then evaluated eight fixed C1 test
events with correct articles, three shuffled-article seeds, and both distinct
article-free modes. Both jobs completed successfully on one A100 at a time;
total use was 20m36s, or 0.343 GPU-hours, within Nicole's eight-hour cap.
The [derived paired results](../../analysis/text_value/C1_SMOKE_RESULTS.md)
and per-event records are on GitHub. Full NPZ files with price arrays and the
checkpoint remain on Delta. This 16-training-event run establishes execution
only; it is not a C1 benchmark.

## Next handoffs

For a teammate or agent starting fresh, use the concrete
[owner-by-owner action plan](../../TEAM_WORKFLOW.md#9-immediate-next-actions-agent-handoff-for-wednesday-november-11).
It names starting artifacts, dependencies, Wednesday deliverables, and what
counts as evidence for each diagnostic stream.

1. **Nicole:** review the paired smoke artifacts in PR #4 and preserve the
   two article-free modes as separate controls. No larger GPU run is started
   until the timing assumption is resolved and a full baseline is planned.
2. **Joye:** a common full C1 baseline still needs a planned training budget
   and a separate, provenance-checked checkpoint. The 16-example smoke
   checkpoint must not serve as that baseline.
3. **Team/TA:** confirm actual article release times and the price CSV's
   timezone and bar-label convention. The current C1 timing is an explicit
   `category4_1400_strict_before` sensitivity assumption, selected for the
   smoke run. The upstream CAMEF README category numbers differ from the
   actual Delta article tree: sampled category 4 is FOMC minutes and category
   3 is GDP. The manifest has 12 categories; categories 7–12 make up most
   C1 entries and have no publication-time field. Official 2024 release
   examples make 14:00 for FOMC and 08:30 for GDP plausible, but this is not
   verified per-event release metadata. See source links and the bounded
   evidence in the technical handoff. C1 commands require the timing policy
   explicitly.
4. **Adaeze/Joye:** continue the separately owned gate and MoE diagnostics.
   **Jessica:** her padding-mask fix and CPU tests are already merged in
   `main`. Ask for any post-fix Q-Former results she has already produced
   before deciding whether she should extend that diagnosis or take another
   team task. PR #4 does not change those model functions.
