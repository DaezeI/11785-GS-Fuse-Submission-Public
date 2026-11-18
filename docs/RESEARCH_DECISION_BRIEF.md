# Research decision brief

## November 18 group meeting

**Both price-only comparisons are complete.** At the main 140-bar forecast
length, the news-trained models cut error by **12.5%** and **26.2%** versus
the separately trained price-only references. The simple rule of carrying the
last prices forward then beat those news-trained models by **8.2%** and
**18.2%**. It had the lowest error at every forecast length for both seeds.
Correct and shuffled articles still gave nearly identical results. The
evidence supports a difference between the training packages. It leaves the
incremental value of the particular article unestablished.
[Side-by-side figure and result](../results/screened-price-only-v1/summary.md)

**Adaeze’s first three return runs are complete.** Changing direction loss
and the gate target hardly moved overall error, which stayed near 81 basis
points. A zero-return forecast scored 80.49 basis points on the same examples.
R3 and Adaeze’s paired article checks were still queued at the final Delta
check. Keep the result descriptive until those checks are available.
[Aligned return evidence](../results/screened-return-r0-r2-v1/summary.md)

**Recommended team decision.** Put the simple baselines and the shuffled
control in the main paper table. After Adaeze’s queued controls come back,
use a small article model to see whether the text carries information beyond
prices and event type. This will help decide whether to revisit fusion or
focus on how the stories were timed and selected.

The sections below record the comparison design and the evidence that
motivated it.

## Established findings

| Evidence | What it establishes | Scope |
| --- | --- | --- |
| Nicole's [seed 0](../results/screened-price-v1/summary.md) and [seed 1](../results/screened-price-seed1-v1/summary.md) | Both full screened price runs found almost the same error with matched and shuffled articles at 35, 70 and 140 bars. At 140 bars the relative gains were −0.0006% (95% interval −0.0277% to +0.0234%) and +0.0085% (−0.0061% to +0.0244%). | Same 2,236 S&P 500 test events and 845 dates, three shuffles per checkpoint. The intervals describe each trained model and do not measure variability over training seeds. |
| Same price checkpoints, [seed 0 controls](../results/screened-price-v1/metrics.csv) and [seed 1 controls](../results/screened-price-seed1-v1/metrics.csv) | Zeroing article tokens lowered measured error at all three horizons for both seeds. Bypassing fusion lowered error further, with intervals favoring bypass except at 140 bars for seed 0. | Both interventions change what an already trained model receives. Bypass also changes its computation. Neither measures the performance of independent text-free training. |
| [Volatility subgroup analysis](../results/price-volatility-v1/summary.md) | At 140 bars, matched versus shuffled gain was about −0.0009% in the high prior-volatility group (1,018 reports, 382 dates). Intervals cross zero at every horizon and subgroup under 5-, 10-, 20- and 40-date blocks. | Exploratory, seed 0 only. Price movement before the forecast does not identify a fresh news surprise. |
| Adaeze's [earlier text controls](../analysis/text_value/MEETING_BRIEF.md) and [R0–R3 launch record](updates/2026-11-17-adaeze-r0-r3-launch.md) | Historically, 98.1% of the observed overall real-vs-no-text price-error improvement survived shuffling. On frozen C1, the return campaign's 16-training-event smoke and reload finished, and full training and paired evaluations were submitted. Adaeze’s R0–R2 full runs later completed with nearly identical test return error, around 81 basis points. [Owner result handoff](updates/2026-11-18-adaeze-r0-r2-results.md) | The historical control's article-free implementation is unverified. The eight-event return evaluation checked execution. The November 18 scheduler check confirmed R0–R2 completion. R3 and the article controls were still pending at that check. |
| Jessica's [fixed-mask rerun](../analysis/qformer_padding/PADDING_FIX_OLD_SPLIT_RERUN.md) and Joye's [MoE diagnostics](../figures/interpretability/interpretability_moe.json) | The Q-Former mask fix took attention on padding from 34% to 0% while test MSE moved from 6.33 to 6.32 × 10⁻⁴. Joye's 200-example diagnostic showed category-dependent routing, with mean expert text weights about 0.497–0.536. | Both are historical. Jessica's before/after used an older split, unfixed seeds and different machines. The routing plot does not show whether article meaning improved accuracy. No screened MoE result was located. |

The frozen C1 audit found zero shared price bars between splits and removed 883
articles with suspect dates. Publication times for retained articles, the price
clock convention, and whether each story was available at the forecast start
remain unresolved. [Article-date audit](../analysis/text_value/ARTICLE_DATE_AUDIT.md)

### Last observed prices beat the first news model

A CPU check reused the hash-verified seed-0 predictions and repeated each
channel's last observed value (open, high, low and close) across the forecast.
Both sides used the same 2,236 examples, 845 dates, training-standardized
units and price targets. Lower mean squared error is better.

| Forecast length | News model MSE | Last-price MSE | Lower error with last price (95% interval) |
| --- | ---: | ---: | ---: |
| 35 bars | 0.006834 | 0.005412 | 20.81% (14.83% to 27.84%) |
| 70 bars | 0.007559 | 0.006362 | 15.85% (10.41% to 21.97%) |
| 140 bars | 0.011376 | 0.010442 | 8.21% (4.51% to 12.33%) |

The paired intervals use 1,000 resamples of 20 consecutive event dates. The
conclusion also holds at block lengths of 5, 10 and 40 dates. In the same
archive, swapping the article did change forecasts, but the root mean squared
change was only 0.37–0.47% of the model's forecast RMSE across horizons. The
three shuffles used later-date donors for about 49% of events each. The
baseline result and prediction-change check are conditional on seed 0.
[Reproduction command and provenance](../results/screened-price-v1/persistence.md)

## What the current experiments answer

- **Did the trained price model benefit from the particular article it
  received?** The same-checkpoint shuffle is a direct diagnostic. The answer
  on average was no detectable gain in price-path MSE under this shuffle
  policy. Random swaps can retain broad topic, style or macro regime and can
  draw from the future. The small forecast changes show how little the output
  moved under those swaps.
- **Would training with articles beat training on prices alone?** The
  last-price forecast is a useful operational reference. A matched,
  independently trained price-only model is still missing. The early
  implementation's so-called TS-only features could contain text. Both
  screened price runs used commit
  [bb09396](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/bb09396881cc92c5e654c4210944ce5c6da17ced/model/gs_fuse.py#L1064),
  which already computes the gate target from price features before text
  attention. Its internal strict branch still shares a decoder trained with
  articles. The historical implementation issue cannot explain these newer
  results by itself. The original GS-Fuse paper's Table 4 likewise compares
  the full system with its internal time-series branch. [GS-Fuse, §6.5](https://arxiv.org/html/2605.28520#S6.SS5)
- **Do the return changes make text useful?** R0→R1 changes direction
  supervision, R1→R2 changes gate supervision, and R2→R3 adds the learned
  last-value shortcut. Adjacent scores tell us which changes help the return
  model overall. The matched-versus-shuffled gap within each version, and
  the paired change in that gap between adjacent versions, can test whether a
  change improves use of the particular article. Once artifacts arrive, use
  the same event IDs and endpoint close-return error and direction, with a
  zero-return and always-up reference as appropriate. Price versus R3 changes
  the target and training together. Convert both to endpoint close returns
  with the same last input close for any package comparison.

## Plausible explanations

- **The story may be late, repetitive or about a different move.** Date
  conflicts are documented. A recap may describe a move already present in the
  35 input bars, and a dated story may still have arrived after the supposed
  forecast start. Research on high-frequency macro releases finds responses to
  surprises and differences across market conditions. Financial text pairing
  work also suggests that relevance and curation can matter. These results
  motivate verifying the clock and the text, with no inference about how often
  either issue occurs in our sample. [Andersen et al.](https://www.federalreserve.gov/pubs/ifdp/2006/871/default.htm)
  [Boudoukh et al.](https://www.nber.org/papers/w18725)
  [FinTexTS preprint](https://arxiv.org/abs/2603.02702)
- **The model may be better at using recent prices than article details.** A
  raw price residual gives it a direct price path. Its gate is supervised
  using comparison-branch losses, so the gate can learn which branch generally
  helps without learning which article matters. The small swap effect, bypass
  results and historical routing/mask diagnostics are consistent with this
  explanation. Replacing one architecture could change general accuracy even
  when content use stays flat. [Wang, Tran and Feiszli](https://arxiv.org/abs/1905.12681)
  [Sridhar et al., Time-MMD audit preprint](https://arxiv.org/html/2608.22321)
- **A price-path score can miss a brief or direction-specific effect.** The
  current score averages squared error over four channels and all steps up to
  each horizon. A short move or a rare release type may contribute little to
  that average. The volatility analysis lowers confidence in a simple
  high-volatility explanation. Verified timing, predeclared event groups and
  return endpoints address a different question. A result near zero with tight
  bounds would still be specific to the data, target and tested models.

## Three proposed experiments

### 1. Train the same price forecaster without articles

**What it teaches.** Whether the news-trained system improves over a capable
model trained entirely on prices, and whether either beats the last-price rule.

**What changes.** Add an independent price-only training path from the price
pretraining used for the corresponding seed, if that stage's checkpoint can
be verified. Use the same price backbone, decoder and raw residual. Train the
price decoder on the same event windows with the price regression objective.
The article warm-up and text/fusion objectives are inapplicable to this path.
Match eligible event-forecast updates and document the stage and parameter
differences. This compares two training packages.

**Fair comparison.** Pair with existing matched-news checkpoints in seeds 0
and 1. Hold the frozen snapshot, scaler, price task, windowing, applicable
hyperparameters, batch size and hardware/precision fixed. Use the original
price checkpoint selection rule of lowest validation MSE. Score the original 2,236 test IDs in the same order. Include last-price
persistence and the existing shuffled and bypass diagnostics. Prespecify
140-bar prefix OHLC MSE as primary, 35 and 70 bars and endpoint close-return MAE
and direction as secondary. Show paired date-block intervals for each seed and
the variation between seeds. As a proposed paper threshold, use a 1% relative
MSE gain agreed before scoring. An interval that still includes a worthwhile
gain leaves that question open. If text beats the trained comparator while
real and shuffled stay tied, the performance gain cannot be assigned to
article matching. [Wang, Tran and Feiszli](https://arxiv.org/abs/1905.12681)
[Diebold and Mariano on dependent forecast errors](https://users.ssc.wisc.edu/~behansen/718/DieboldMariano1995.pdf)

### 2. Give the article a simpler way to help

**What it teaches.** Whether useful article information is available to a
simple model even though GS-Fuse barely responds to swaps.

**What changes.** On frozen C1, fit a small regularized regression for the
35-bar endpoint close return. Start with lagged price features and a
metadata-only version that also knows the calendar and recorded event type.
Add frozen, pooled article embeddings to the same feature set. Reuse verified
embeddings if available. Fit a twin with reassigned embeddings as a negative
control using within-split donors. Also rescore the matched-embedding model
with reassigned embeddings at test time. Keep event rows and all price/calendar features fixed.

**Fair comparison.** Choose preprocessing and regularization on training and
validation only. Use seeded different-date donors from the same event type and
period for the content comparison, excluding exact and near duplicates with a
train-defined rule. Use past-only assigned dates when timing allows and label
unknown-time shuffles as diagnostic. Compare every arm on the same eligible
test IDs and report donor coverage and fallback exclusions. Better MAE for
matched text over both metadata and reassigned text would motivate fixing
fusion or training. A tie leaves weak signal, timing and inadequate encoding
plausible. A later matched R3 architecture test should first earn its case
here or in Adaeze's own controls. [Sridhar et al.](https://arxiv.org/html/2608.22321)
[Andrews and Mengaldo, text-information preprint](https://arxiv.org/abs/2609.11282)

### 3. Check fresh news at the moment it can actually be read

**What it teaches.** Whether poor timing and averaging across long price paths
could be hiding a short-lived benefit.

**What changes.** First audit a prespecified source-backed set of screened
events, starting with scheduled releases. Verify the actual article's first
availability and the market bar clock. Official announcement time alone
cannot verify when third-party stories were published. Tag recaps using a
rule written before looking at price outcomes. If enough dates can be
verified, freeze an audited subset and a clock starting at the first
tradable forecast origin after the article is available. Use only price bars
that had finished by that origin. Log exclusions and re-audit window
boundaries. This is a new proposed audit and comparison. The previously
canceled eight-announcement GPU comparison remains canceled.

**Fair comparison.** Use identical post-availability forecast windows for
matched articles, earlier same-type nonduplicate donors, and a price-only
reference. Compare those same arms at the existing assumed origin on the
same subset only as a labeled timing sensitivity. At the verified origin,
predeclare a short close-return endpoint (first five available bars, with
35 and 140 as context), compare paired MAE and direction, and report event
and date counts. Improvement concentrated at the verified short endpoint
would support dilution or misalignment. With too few verified dates, report
coverage and uncertainty as an audit rather than a model conclusion. Use
release differences versus contemporaneous consensus only where that
expectation is independently available. [Andersen et al.](https://www.federalreserve.gov/pubs/ifdp/2006/871/default.htm)

## Recommended overnight batch

1. **Start with the existing evidence.** Collect Adaeze's completed full and
   paired outputs if available. Check split, snapshot, checkpoint and event
   IDs, then extract the adjacent R0–R3 changes and within-run text gains.
   Reuse the existing saved predictions and record any outstanding artifacts.
2. **Use the two GPU slots for experiment 1, one single-H200 chain for each of
   seeds 0 and 1.** Reuse the existing multimodal checkpoints. Before launch, verify an independently text-free config, initial
   checkpoint provenance, and a smoke that covers the relevant stages,
   save/reload and matched evaluation. If both references are ready, run
   both seeds. If only one is ready, run that seed when authorized and use the
   remaining time for the CPU probe or timestamp audit. If neither reference
   passes its checks, keep the window for those CPU tasks and owner-result
   collection. Freeze configs and
   selection rules before test scoring. The previous full price chain took
   about 42 minutes of GPU execution before smoke/setup/queue overhead. For
   planning, allow up to two allocated GPU hours per proposed seed chain within
   the approved overnight window. This is a cap proposal, not a measured
   runtime. Confirm scheduling and the budget before launch.
3. **Morning decision.** Read the paired results against both references. If
   news gains are absent but the simple text readout helps, investigate the
   fusion/training bottleneck. If timing coverage is good, prioritize the
   verified short-horizon test. Leave Q-Former and MoE extensions behind a
   matching screened R3 reference and the respective owners' evidence.

Nicole approved the price-only batch for November 18. Both runs and their evaluation are complete. [Verified results](../results/screened-price-only-v1/summary.md)
