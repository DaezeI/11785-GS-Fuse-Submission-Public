# Short-horizon Fed/SPX baseline proposal

**Baseline design · November 23, 2026.** Adaeze Ilodianya selected 20- and
30-minute horizons and owns the baseline design. Share the target, cutoffs,
eligibility and metrics with the text-model owner so the forecasts can be
compared on identical examples. The question is whether official Fed release
text improves a near-term S&P 500 Close forecast given the same pre-release
price history.

## How this builds on the existing work

- **Official announcements.** Jessica's v2 catalog on `main` records official
  releases, market-time evidence and fingerprints of downloaded material.
  Market time and historical availability of the exact downloaded version are
  separate questions. [Release catalog and evidence rules](../../analysis/macro_releases/README_v2.md)
- **Price alignment.** Joye's alignment v1 at `a4269dc` fixes the input cutoff,
  processing lag and chronological purged split. Its bar-start convention is
  supported by the close and announcement-reaction checks. The source file
  does not document its clock or when a bar reached a user.
  [Alignment and clock audit](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/a4269dce74eb0cc829fa6772b049eed7ed463abd/docs/updates/2026-11-22-fed-price-alignment.md)
- **Existing baseline results.** Repeating the last pre-release Close beat
  ridge on the 140-bar path across 215 main validation rows. This establishes
  the result for those two price forecasts. Keep the overnight-spanning
  full-path task as a supporting experiment with its own target and metrics.
  [Price-only result pack](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/96c4e3caec22edfc12713360186d177aa2e93bb6/results/fed-spx-price-baselines-v1/summary.md)
- **Learned baseline.** Reuse the compact causal CNN from Adaeze's merged
  laptop-GPU work, with a two-output Close-return head and fresh training on the
  approved Fed/SPX rows. The earlier compact run used screened C1 and archived
  summaries. Its architecture transfers to this experiment. Its checkpoints
  and scores belong to the earlier cohort and target.
  [Merged laptop model](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/34)
  · [Architecture](../../experiments/improvement-v1/COMPACT_ARCHITECTURE.md)

## Proposed target and clock

Use **20- and 30-minute Close log returns in basis points**. Report MAE and
direction at each horizon. The 20-minute result is the shorter-window headline
and the 30-minute result is a companion on the same events. Both focus the
forecast on the intraday period remaining after text can be processed.
The immediate release jump may already have happened by then. A standard FOMC
event study uses a window around the release, including some time before it.
The proposed forecast starts at a later decision time.
[San Francisco Fed event-window definitions](https://www.frbsf.org/wp-content/uploads/wp2025-30.pdf)

1. Let `E` be the verified market time in New York time. Set the decision time
   `T0` to `E + 5 minutes` rounded **up** to the five-minute grid, retaining exact
   boundaries. Require the alignment v1 target start to equal `T0`. A missing
   bar at `T0` makes the row ineligible at this horizon.
2. Both arms receive the same 35 OHLC bars that ended by `E`, normalized using
   the last pre-release Close. The text arm also receives the eligible release
   text. The claim is about adding text given this pre-release price history.
3. Let `C(t)` be the Close of the bar **ending** at `t`. Each label is
   `10,000 × ln(C(T0 + horizon) / C(T0))`. Use the Close of the fourth and sixth
   future bars for 20 and 30 minutes, respectively. The bar ending at `T0`
   supplies the starting Close for the labels only. Its OHLC values stay out
   of both input histories. Both labels isolate the return from `T0` onward.
4. Require the anchor and all six target bars on the same date, with exact
   five-minute spacing. The anchor must start at or after 09:30, and the last
   bar must end by 16:00 and within the actual session. The separate 16:00
   closing-print row is outside the proposed window. The source does not
   document dissemination exactly at the assumed bar boundary.

| Example for a 14:00 ET release | Time |
|---|---|
| Last price input | 13:55–14:00 bar |
| Decision time and starting Close for the label | 14:05, using the Close of 14:00–14:05 |
| Six target bars | 14:05–14:10 through 14:30–14:35 |
| 20-minute Close | 14:25, using the Close of 14:20–14:25 |
| 30-minute Close | 14:35, using the Close of 14:30–14:35 |

**Follow-on releases.** For a 14:00 FOMC statement, the 30-minute target can
include the start of a 14:30 press conference. Flag known follow-on Fed
communications from `T0` through the endpoint using timestamp evidence and
report the no-overlap subset as a sensitivity. The 20-minute endpoint falls
before the press conference in this example. Actual follow-on times determine
the overlap flag for each horizon.
[Federal Reserve study of statement and press-conference timing](https://www.federalreserve.gov/econres/feds/files/2026029pap.pdf)

## Eligibility and timing evidence

Start with alignment v1's **main** rows and original split assignments. Apply
the target-clock and text-version rules within those splits. Preserve purged
exclusions and keep flagged sessions and press-conference transcripts separate.
This first comparison inherits the 140-bar screen. Recovering rows that qualify
only at short horizons would use a separately versioned cohort.

Use the same events for both horizons. Require all six future bars and the
same text-version cutoff for both labels. This makes the price-eligible
shared cohort the 30-minute subset in the timestamp check below. Apply any
later text-version removals to both forecasts and both horizons.

**Text.** Record each event ID, source and historical evidence URLs, exact
text or source artifact fingerprint, extraction method, and a defensible
latest time by which those contents were public. Require that upper bound
plus five minutes to be at or before `T0`. A historical capture with matching
contents or a versioned official release can support the bound. A market-time
entry, today's matching hash, or a server modification header alone cannot
establish it. For a PDF, link the historical artifact to the exact extraction
fed to the model.

Check prepared speeches and written testimony separately from live delivery,
later edits and Q&A. Admit the full text only where the exact prepared version
passes the cutoff rule. Keep press-conference transcripts separate. A transcript
dated the meeting day or assigned an assumed midnight cutoff gives insufficient
evidence for use at an intraday forecast origin. Exclude unresolved versions
from both matched arms and report counts and reasons by split and type. A
price-only score on the broader price-eligible cohort is a separate analysis.

**Prices.** Seek source documentation or an independent clock check for bar
labels and boundary availability. If the evidence stays empirical, state that
assumption in the result and qualify the forecast-time claim. Audit emitted
windows, including the new anchor bar, for cross-split overlap before unlocking
held-out scoring. Changing the target supplies no new clock or text evidence.

### Timestamp-only feasibility check

I checked **main train and validation** metadata. I verified the aligned start
equals `T0`, the anchor and required bars exist on the same date and grid, and
the 35 observed price bars end by `E`. The check used the 09:30–16:00 clock
and observed bar presence. Independent verification of early-close sessions
is pending. The check calculated no returns and inspected no reserved test
windows.

| Horizon | Train rows / instants | Validation rows / instants | Shared train/validation bars |
|---|---:|---:|---:|
| 20 minutes | 1,008 / 959 | 213 / 198 | 0 |
| 30 minutes | 978 / 929 | 207 / 192 | 0 |

The starting main cohorts have 1,013 train and 215 validation rows. At 30
minutes, 35 train and eight validation rows failed the session boundary rule.
The check retained 787 train dates and 165 validation dates. The 207 validation
rows include 95 speeches, 21 testimonies and 18 FOMC statements. The number
that passes exact text-version review remains unknown. The alignment CSV hash
was `0ab047bf9f918d798bd1c4f4eac8c6d59e0317c8d36fcb25c81a6801cfee0270`.
The price and catalog hashes matched the
[v1 input audit](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/a4269dce74eb0cc829fa6772b049eed7ed463abd/analysis/macro_releases/price_alignment_v1/price_alignment_audit_v1.json).

## Model, targets and scoring

- **Matched control.** Keep the 35-bar, four-channel input and the compact
  price encoder. Replace the forecast head's last layer with two outputs in
  fixed order `[20 min, 30 min]`. Fit on the shared eligible train rows. Use
  equal loss weight per horizon after scaling each target on eligible train.
  Freeze that price checkpoint and add a matching two-output text residual
  in the text arm. Use identical input price bytes, evaluation rows, cutoffs
  and targets. Fit vocabulary and normalization on eligible train only. If
  the text model uses a different architecture, pair it with its own
  price-only counterpart.
- **Target and score shapes.** The current full-path loader scales `y` from
  the last pre-release Close. The endpoint label builder must read the Close
  at `T0` separately and produce `[event, 2]` targets. Both models emit that
  shape. An endpoint scorer consumes predictions in unscaled bp and reports
  errors independently for each horizon. The historical full-path scorer
  expects `[event, 140, 4]` and applies to the supporting experiment.
- **Independent references.** Report zero return from `C(T0)`. Ridge and
  DLinear can be separately refitted on this target and cohort. A wrong-text
  diagnostic can use same-split donors whose text was public by the recipient's
  cutoff. Freeze donor draws and score the same diagnostic rows in each arm.
- **Unit of analysis.** Keep concurrent releases in the same split. Give every
  release instant unit weight by assigning each retained row weight `1/n`,
  with `n` calculated within the shared cohort. Report counts of rows, distinct
  instants and dates, with exclusions by split and release type.
- **Metrics.** Use weighted terminal-return MAE in bp at each horizon. Report
  `MAE(price-only) − MAE(price+text)`, where positive favors text. Score the
  signs of nonzero realized returns and report the eligible count and frequency
  of flat predictions. A flat forecast counts as a miss on nonzero moves.
  Use paired 95% intervals from 2,000 draws of blocks of ten consecutive
  announcement dates, with five-date blocks as a sensitivity. Use the same
  sampled dates for both arms and horizons. Compute the weighted absolute
  error difference in bp in each draw and keep same-date rows together. The
  existing bootstrap reports percentage improvement. Reuse its block sampling
  and change the statistic for the endpoint comparison.

A synthetic check on the laptop RTX 4060 produced two outputs per event from
the compact model and matched a zero text residual to the frozen price forecast.
Synthetic label and metric calculations matched the expected four- and six-bar
returns, weighted MAE and direction. These checks exercise the mechanics on
synthetic examples.

## Freeze before held-out scoring

Use train and validation to check coverage and select model settings. Before
held-out scoring, freeze the two horizons and their reporting order, common
eligibility rule, price and text inputs, training settings, checkpoint
selection and metrics. Check that the text-model handoff uses the same
definitions. Register the configured run before fitting it and keep all
held-out examples out of training and selection.
