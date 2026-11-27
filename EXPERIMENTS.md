## November 27 shared-query Transformer — completed

Nicole completed `none+Transformer128x4~none~none` on `SPX-PX6-GRID5-v1`: 204,694 training / 40,548 validation windows, six supervised positions through 30 minutes, batch 204,694, 1,000 optimizer updates, one H200, exit 0 in 31m32s including calibration. Selected update 50 mean MAE is 11.689790 bp versus Persistence 11.706526 bp; training continued fitting while late validation deteriorated. This is a small exploratory validation gain. [Result and provenance](results/shared-token-transformer-v1/summary.md).

# Experiment register and reporting rule

This is the protocol and citation index for team experiments. Use the
[research registry](research_home/records.json) for current statuses,
findings and evidence links. The registry feeds the research home. Keep
planned runs, queued runs, completed runs, and measured findings distinct.
Large data, predictions, and checkpoints stay on Delta; link their shared paths
and commit only small run records or summaries here.

## Before a run

Record the question the run answers, the comparison run, the one intended
change, and what stays fixed: manifest and timing policy, event population,
price and article snapshot hashes, code version, model/backbones, training
budget, seed, metrics, and evaluation controls. Publish changed data as a new
version, preserving the prior copy and its audit. Save the exact launch command and a fresh output path.
Run the same evaluation population and metric definitions for both sides.

One-change comparisons support claims about that change. A bundled redesign is
also useful, but label it as a package comparison and do not credit its gain to
one component. A data screen is a validity decision; do not present an
unscreened run with suspect article dates as a clean benchmark merely to make
an ablation. If the population changes, state that the scores are not paired.

## After a run

Add a result note or update the linked handoff with: status, owner, code commit,
manifest, price, and article snapshot hashes, training seed and resolved settings, job ID, exact
command, checkpoint, output path, paired evaluation IDs, metrics with counts and
uncertainty, observed finding, interpretation, limitations, and next action.
Use `PENDING` for unknown fields. A queued job has no result. A smoke run checks
execution, not model quality. Do not copy a result from one checkpoint or data
population into another run's row.

## November 26 training diagnostics and rolling-price comparison

Nicole completed a [CPU training diagnostic](results/fed-training-diagnostics-v1/summary.md): the text-head training loss fell another 10.33% over ten extra updates while validation error rose. This is not an original W&B history or a new selected checkpoint. The loader audit explains 978 as event-window subsampling despite 225,561 available training-period price bars.

At the user's request, a separate [rolling-price training comparison](experiments/dense-price-grid-v1/config.json) replaces event-only training with 204,694 five-minute-grid windows or 68,219 fifteen-minute-grid windows. Both use the same `none+CNN~none~none`, common rolling validation, original target clock and detailed training-loss logging. Real-input separation audit and four focused boundary tests passed. Both fits completed on one H200 in 6m43s allocated time. Selected forecasts have small ordinary rolling-validation gains, but no improvement on the 149 matched Fed validation releases. Continued fitting worsened late validation. [Completed result](results/dense-price-grid-v1/summary.md). No test scoring was performed.

The requested [shared nonlinear decoder](docs/updates/2026-11-26-joint-decoder.md) is implemented and functionally checked, with one or two hidden layers over concatenated price/text representations. This is engineering work, not a completed new text experiment or an early-fusion result. A separately trained price-only readout must be included before assigning any future gain to text.

## Nicole's Fed text pilot (November 24)

[Exact 8B pilot](experiments/fed-spx-text-llama31-8b-v1/README.md), owner Nicole, #55: **training and validation completed November 26**. The 171/149 text arm essentially ties CNN at 20 minutes and is slightly worse at 30; all ALL-cohort matching comparisons are inconclusive. Frozen-price checks passed; one H200, exit 0, 7m46s. [Result pack](results/fed-spx-text-llama31-8b-v1/summary.md). Test remains reserved pending the team freeze.

## Current comparison index

**Fed/SPX price-only checks, completed on validation:** [result pack](results/fed-spx-price-baselines-v1/summary.md).
Last-close persistence beat ridge at 35, 70 and 140 bars. Flagged validation
events are reported separately. Matched compact models and DLinear scores on
this split are pending.

**New planned campaign:** [Controlled improvement v1](docs/experiments/IMPROVEMENT_CAMPAIGN_V1.md)
provides 17 existing-pipeline configurations, a five-arm default smoke screen,
and 11 evidence/implementation-gated designs. Adaeze owns the experiment
design; Nicole is the intended executor. All records remain planned; this
handoff submits no GPU jobs. [Operator prompt](experiments/improvement-v1/NICOLE_PROMPT.md).

The [research decision brief](docs/RESEARCH_DECISION_BRIEF.md) inventories the
current evidence and ranks the next proposed analyses and experiments.

| Run or comparison | Purpose | Intended change | Status and evidence |
|---|---|---|---|
| Screened C1 price baseline | Establish the price-task reference and matched text controls | Uses the November 13 screened C1 manifest | Completed on H200 with full paired controls on 2,236 test events. See [the verified result pack](results/screened-price-v1/summary.md); real and shuffled article MSE are nearly identical for this checkpoint. |
| Screened C1 price replication | Check whether the matched-versus-shuffled result repeats | Training seed 1 under the same screened price protocol | Completed. [Seed-1 pack](results/screened-price-seed1-v1/summary.md). All matched-versus-shuffled intervals include zero. |
| Last-price persistence vs. seed-0 news model | Benchmark the news model against a flat price forecast | Repeat each price channel’s last observed value on the identical 2,236 test events | Completed CPU reanalysis. [Persistence result](results/screened-price-v1/persistence.md) shows 20.81%, 15.85%, and 8.21% lower MSE at 35, 70, and 140 bars. |
| Price controls by pre-event volatility | Test whether the near-zero text gap hides a volatile-regime benefit | CPU reanalysis of the same saved checkpoint and predictions; no model/input changes | Completed: [result pack](results/price-volatility-v1/summary.md). All real-versus-shuffled intervals include zero across three groups, three horizons and four bootstrap block lengths. |
| Screened C1 return redesign | Test Adaeze's return design as a package on the same screened population | Return target, Huber/direction task loss, gate target, and last-value shortcut change together | Merged to `main`. Smoke job `24055019` completed `0:0` (16 events, 3 stages, verified checkpoint/config). Full training job `24055023` (R3) submitted and queued on Delta (`gpuA100x4`, pending with estimated start `2026-11-18 16:36:05`). Paired control jobs `24055030`–`24055031` pending on dependency. No full result yet; see [launch update](docs/updates/2026-11-17-adaeze-r0-r3-launch.md). |
| Return component ablations R0→R3 | Identify which return-design choices matter | Add direction penalty, then full-task gate target, then last-value shortcut, one at a time | Merged to `main`. Full training for R0 (`24055020`), R1 (`24055021`), and R2 (`24055022`) completed `0:0` with verified 1.8 GB checkpoints on 2,302 validation and 2,236 test events. Test return MAEs are 80.92 bp (R0), 81.08 bp (R1), and 81.08 bp (R2); test DHRs are 50.5%–52.9%. Differences between R0, R1, and R2 are minimal ($\Delta\text{MAE} < 0.16\text{ bp}$, R1 vs R2 difference $< 0.003\text{ bp}$). R3 full training (`24055023`) and paired control jobs (`24055024`–`24055031`) are queued on Delta (`gpuA100x4`). See [R0–R2 results handoff](docs/updates/2026-11-18-adaeze-r0-r2-results.md). |
| Independently trained price-only models | Test incremental performance of the news-training package | Start each price-only model from its corresponding price pretraining and compare with the original news checkpoint and persistence | Completed. The news-trained package beat these two references. The last-price rule beat both trained approaches. [Two-seed result](results/screened-price-only-v1/summary.md). |
| Zero return vs. Adaeze R0–R2 | Benchmark return error and direction on the existing test population | Score zero return and always-up forecasts against the same frozen population and eligible directions | Completed descriptive check. Zero-return MAE was 80.49 bp versus 80.92–81.08 bp for R0–R2. [Aligned return summary](results/screened-return-r0-r2-v1/summary.md). |
| Adaeze R3 replay on one H200 | Measure article-match sensitivity of R3 and establish the zero-return reference | Replay Adaeze’s exact code and frozen R3 protocol on Nicole’s personal GPU resource | Completed. Matched and shuffled MAE were almost identical. Zero return had slightly lower error. The H200/A100 comparison is descriptive. [Verified R3 result](results/adaeze-screened-r3-h200-v2/summary.md). |
| Jessica’s Q-Former Stage 3 smoke | Verify that Q-Former weights update after PR #28 | Use Q-Former fusion with the R3 return flags on a small frozen C1 subset | Passed. The projection and LayerNorm weights changed during Stage 3. [Verified smoke evidence](results/jessica-r3-qformer-unfreeze-smoke-v1/summary.md). |
| Simple article readout | Test whether a simple model can use the article content | Add matched or reassigned frozen article features to a price and metadata regression | Completed on 2,191 held-out events. Matched articles did not lower MAE over price and metadata. [Result pack](results/c1-article-readout-v1/summary.md). [Comparison details](docs/RESEARCH_DECISION_BRIEF.md#2-give-the-article-a-simpler-way-to-help). |
| Verified article availability | Test for incremental gains around the first available forecast origin | Audit availability and compare matched text, older text and prices at the verified clock | Proposed. [Comparison details](docs/RESEARCH_DECISION_BRIEF.md#3-check-fresh-news-at-the-moment-it-can-actually-be-read). |

The original unscreened C1 manifest and historical outputs remain preserved for
audit context. The screened subset excludes 883 suspect articles; publication
times for retained articles are still unverified.

## Short-horizon Fed/SPX baseline

**Design proposal, Adaeze Ilodianya:** [short-horizon protocol](docs/experiments/FED_SPX_SHORT_HORIZON_PROPOSAL.md).
Compare the compact price-only architecture and a matched text extension on
20- and 30-minute Close returns from a shared forecast origin. The horizons
are chosen. The [compact price-only validation run](results/fed-spx-endpoints-price-v1/summary.md)
completed with three RTX 4060 seeds. The paired intervals cover zero at both
horizons. The exact text-version audit and its coverage limits are linked in
the result pack. The [Board and testimony audit](results/fed-spx-board-testimony-evidence-v1/README.md)
reviewed 603 events and verified eight testimony PDF bodies by their frozen
cutoffs. The [independent minutes and speech audit](results/fed-spx-minutes-speeches-codex-v1/README.md)
reviewed all 472 assigned events and found no exact-version proof in the
publisher records and archive pilot it checked. The
[combined Fed/SPX news cohort](results/fed-spx-news-cohort-v1/README.md)
reconciles all 1,185 train and validation event IDs and records the sensitivity
to the minutes and speech evidence rules. A [Board release re-check](results/fed-spx-board-recheck-v1/README.md)
adds 10 train and 33 validation rows under the document rule; the cohort file must be rebuilt.
The [macro coverage count](results/macro-coverage-v1/README.md) finds 2,158 candidate training
events before text checks. Decisions and limits are in the
[November 23 update](docs/updates/2026-11-23-fed-spx-text-evidence.md). The matched experiment and held-out
scoring remain pending.

## Entry template for future work

Copy this block into a focused result note linked from the index:

```text
Question / hypothesis:
Comparison run and exact one change (or explicitly named package):
Held fixed (manifest/hash, population, code, model, budget, seed, metrics):
Owner and status (planned / queued / running / completed / failed):
Command, code commit, environment, job ID, checkpoint, output path:
Validation selection rule and test evaluation population:
Metrics, eligible counts, paired uncertainty, controls:
Observed finding (include negative or inconclusive results):
Interpretation, limitations, and next action:
```

## Final-delivery assets

Use [results/README.md](results/README.md) for linked metric sources, result-pack
templates and figures; use [report/final](report/final/README.md) for versioned
final writing. The [delivery workflow](docs/FINAL_DELIVERY_WORKFLOW.md) explains
owner handoffs and how these feed the paper and presentation.

## Compact direction-loss ablation

**Completed, validation only:** [Compact direction-loss ablation](results/compact-direction-v1/summary.md). The penalty did not improve MAE or H140 direction accuracy; Smooth L1 remains the default.

## Separate Fed/Treasury cross-attention and gate pilot

**Completed, validation only:** [Fed/Treasury cross-attention × gate pilot](results/compact-treasury-crossattention-v1/summary.md). No change had the lowest MAE; attention worsened the matched pooling comparison with and without the gate. Not an intraday SPX or screened C1 result.
