# Research decision brief

**Decision:** use the existing evidence first; the ranked batch below is proposed,
not launched. Do not select a new volatility cutoff or model from test results.

## Evidence and valid comparisons

- Nicole's [screened-price pack](../results/screened-price-v1/summary.md) is the
  only verified full screened result pack in the current result hub: seed 0,
  2,236 test events, 845 dates. Correct and shuffled text have nearly identical
  price MSE at horizons 35, 70, 140; the paired 20-date block intervals include
  zero. This is one checkpoint, not evidence that text is generally useless.
- [Pre-event volatility analysis](../results/price-volatility-v1/) uses tercile
  boundaries fitted on training raw close log-return standard deviation, not
  test-tuned thresholds (0.0004110705 and 0.0006358354). For high volatility,
  H140 real-versus-shuffled gain is -0.0009298% (95% 20-date-block interval
  -0.0296067% to 0.0242171%; 1,018 events, 382 dates); all three horizons
  include zero. The 5/10/20/40 global-date circular block sensitivity also
  includes zero for each horizon and volatility group. These exploratory
  intervals are conditional on one checkpoint and three shuffle draws, with
  no multiple-testing correction; they do not establish differences between
  groups. High volatility means pre-event variability, not a verified news shock.
- Adaeze's R0–R3 campaign was submitted; its current queue state and full
  findings are not established here. Collect owner-verified records rather than
  duplicate jobs. R0→R1, R1→R2, and R2→R3 address the specified individual
  return choices when inputs and evaluation match. Price versus R3 changes a
  package and the units; it cannot isolate one component.
- Git also contains [Jessica's Q-Former padding result](../analysis/qformer_padding/PADDING_FIX_OLD_SPLIT_RERUN.md)
  on an older split. It is historical, not a screened-C1 comparison. Joye owns
  MoE. New architecture comparisons require their matching R3 reference and
  compatible hardware/environment, data, seed and budget.
- Zeroed tokens and fusion bypass are interventions on the trained multimodal
  checkpoint, not independently trained price-only models. Shuffle donors may
  come from future test dates, so shuffling is a diagnostic placebo. Retained
  publication times remain unverified. A second team-accessible copy of the
  full checkpoint and predictions, with verified hashes/access, is still due.

## Ranked next batch

1. **Analyze existing owner R0–R3 outputs** — no new training. Once Adaeze's
   verified records and predictions are available, check hashes, resolved
   settings, population and seed, then compute the protocol's paired controls,
   per-horizon return metrics and date-block intervals on aligned examples.
   Reuse available paired outputs; run a missing evaluation only if the owner
   confirms it is absent and the checkpoint/config are verified. This depends on
   owner outputs, not a presumed queue state; effort and evaluation runtime are
   unknown until inspected. Compare the three adjacent pairs, and label any
   price-versus-return comparison as a package in common units.
2. **Train a price-only reference** — seed 0, using the same frozen screened
   manifest/snapshot, price task, split, backbones where applicable, 2/2/2
   epochs, batch 8, TF32 and evaluation population as the completed price run.
   Define and implement a training-time text-free path and validation selection
   rule before running; verify with a three-stage smoke, checkpoint reload and
   matched evaluation. Compare paired per-event price MSE and date-block
   intervals against the seed-0 multimodal model. This is a model-package
   comparison unless training differences are isolated and documented. The
   existing bypass is not this baseline. One H200 price training took 28:55;
   the two paired evaluations took 7:07 and 6:13. These are reference timings,
   not a runtime promise for the new implementation; allow smoke/development and
   evaluation overhead. No ready-to-launch price-only config has been verified.
3. **Replicate the screened multimodal price run** — seeds 1 and 2 with the
   seed-0 frozen input, code, model, budget, batch, precision and same three
   shuffle draws and both article-free controls. Select checkpoints by the
   existing validation rule, evaluate all 2,236 test events at 35/70/140, and
   report each seed plus between-seed variation and paired date-block intervals.
   Two training/evaluation chains at the measured price timings total about
   84:30 GPU wall time before setup, smoke and scheduling; actual time can vary.
   These runs quantify training-seed sensitivity and do not independently
   establish the value of a separately trained price-only model.

Before any GPU launch, freeze the exact commit, commands, output paths and
resolved settings, verify inputs and access, check the owner campaign's current
records where relevant, and run the appropriate smoke. Use the existing result
pack template and preserve negative or inconclusive outcomes.
