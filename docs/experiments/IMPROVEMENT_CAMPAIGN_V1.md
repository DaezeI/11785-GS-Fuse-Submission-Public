# Controlled improvement campaign v1

Owner: Adaeze Ilodianya. Executor: Nicole Lyu. Prepared November 18, 2026.
State: **planned; no remote jobs submitted and no performance results**.
Config: [matrix.json](../../experiments/improvement-v1/matrix.json).
Launch code: [campaign.py](../../scripts/improvement/campaign.py).
Operator prompt: [NICOLE_PROMPT.md](../../experiments/improvement-v1/NICOLE_PROMPT.md).

## What the repository establishes

This design starts from main `dd03d59`, after PR #32 added native q/v LoRA.
The original statement that Llama must remain frozen is now out of date as an
implementation description. The new path still needed end-to-end repairs:
Stage 3 froze adapters before gathering optimizer parameters; the outer token
helper suppressed projection gradients; adapted evaluation could reuse stale
disk features; instance features could be memoized; and combined checkpoints
omitted adapters. This branch repairs these paths and adds regression tests.
LoRA is deliberately enabled **only in Stage 3**, keeping the first two stages'
learning policy fixed. Native adapters use the base layer's dtype and device.
Adapted features bypass disk and instance caches in training and evaluation.
Adapter state and architecture are saved and validated on reload.

These repairs also change projection training for the frozen reference.
Therefore `ref` must be trained afresh on this code. Old R3 is historical
context, not the matched reference for a new-code effect estimate. All new
arms use the same corrected code; no gain is assigned to one repair.

The screened C1 population has 6,167/2,302/2,236 events. Existing price and
return controls do not establish a useful matched-article advantage; simple
last-price/zero-return references remain essential. See the existing
[decision brief](../RESEARCH_DECISION_BRIEF.md) and
[R3 result](../../results/adaeze-screened-r3-h200-v2/summary.md).

## Locked protocol for launchable arms

- Frozen snapshot fingerprint:
  `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`.
  The manifest fingerprint is
  `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`.
  The operator locates the existing identical snapshot on her resource.
- MOMENT, Llama-3.2-1B, cumulative OHLC log returns in basis points, H=140,
  horizons 35/70/140, R3 task gate target and learned last-value shortcut,
  raw TS residual disabled. Batch size 2, sliding stride 10, 2/2/2 epochs,
  fp32 with TF32, no quantization, no checkpoint reuse between arms.
  Resolve exact backbone paths/revisions and environment in the private profile.
- Timing assumption stays `category4_1400_strict_before`. It remains a
  heuristic. The screen removed 883 suspect dates; it did not certify the
  release times of the 10,705 retained reports. Improvements are conditional
  on this assumption until the separate timestamp design succeeds.
- Preprocessing and return-loss scales use training only. The worker verifies
  snapshot content and runs the real-input audit with the **actual** L/H,
  requiring all counts and zero cross-split shared input-plus-target bars.
  Increasing history must preserve target origins, target arrays, event order,
  and anchors; check those against `ref` before paired scoring. Do not silently
  drop events or shorten a context to pass. If a longer context fails, mark
  it failed and design a versioned common cohort for *every* comparator.
- Same epochs match event-training exposure. Longer histories can increase
  Stage-1 eligible windows, parameters and FLOPs. Report these as part of the
  context package; do not call the total compute identical. A separate
  fixed-update sensitivity is required to isolate history from update count.
- Each task runs its own 16/8/8 three-stage smoke, checks adapted-weight
  changes and a frozen base weight when applicable, and reloads the saved
  checkpoint before full training. A failure stops that scheduler lane.
  No silent batch-size, precision, tokenizer or loss changes after OOM.

## Thirteen executable configurations

Every row has a machine-readable override, reference and intended change.
Rank variants keep alpha/r=2, dropout=.05, q/v targets, and adapter LR=5e-5
unless specified. Frozen backbones and projection/decoder training otherwise
follow `ref`.

| ID | Hypothesis / comparison | Change and diagnostic |
| --- | --- | --- |
| ref | Establish a fresh reference on corrected code | Frozen 1B text, L=35, token budget 1024. Compare to zero return and matched controls. |
| lora8 | Financial forecasting gradients can make text useful | vs ref: rank 8, alpha 16 in Stage 3. Check adapter gradients, weight deltas, loss and swap sensitivity. |
| lora4 | Less adaptation capacity may generalize better | vs ref and lora8: rank 4, alpha 8. Report trainable count and validation overfit gap. |
| lora16 | More adaptation capacity may improve content use | vs lora8: rank 16, alpha 32. A score gain without a matching advantage is not semantic evidence. |
| lora8-lowlr | Lower adapter LR may preserve pretrained features | vs lora8: LR=1e-5 only; record convergence and adapter norms. |
| context70 | Some missed regime information lies just beyond 35 bars | vs ref: L=70 only; same forecast origin and H. |
| context140 | Cross-session price history may add predictive context | vs ref: L=140; audit exact observed-bar spans, targets and Stage-1 coverage. |
| context280 | Several sessions may help volatility/regime estimation | vs ref/context140: L=280; check memory, patch behavior, actual timestamps and update counts. |
| tokens256 | Shorter text may remove irrelevant recap material | vs ref: token-fusion budget 256 only. The pooled instance pathway still sees the chunked article. |
| tokens2048 | Truncation may hide useful details | vs ref: token-fusion budget 2048 only. This is not a clean whole-article information ablation because instance pooling is unchanged. |
| qformer8 | A smaller learned text bottleneck may improve fusion | vs ref: 8-query, 2-layer Q-Former package. Require query/projection weight changes, zero padding attention, query redundancy diagnostics. Coordinate with Jessica's existing work. |
| moe4 | Conditional experts may help heterogeneous releases | vs ref: 4-expert top-2 gate package. Log expert load, entropy, category routing, output diversity and collapse. Coordinate with Joye's work. |
| lora8-context140 | Adaptation and longer history may complement each other | Complete 2x2: ref, lora8, context140, combined. On aligned losses estimate `(combined-context140)-(lora8-ref)`; do not attribute package gains to either component alone. |

35 bars represent 175 minutes of observed five-minute bars. Do not translate
140/280 bars into elapsed days without checking the actual trading sessions,
overnights, holidays and extended-hours coverage. This launcher caps L at
280; it does not assume that MOMENT's architecture supports arbitrary lengths.
GPU smokes must verify its actual patch/position behavior at every length.

## Selection, controls and claims

The default screen submits six seed-0 tasks: ref, lora8, context140,
context280, tokens2048, and lora8-context140. Optional arms are prepared
but are not an instruction to exhaust every combination.

For each validation-selected checkpoint save real text, three seeded shuffled
assignments (23/24/25), zeroed article tokens and fusion bypass, with aligned
targets, event IDs/dates/types, anchors, donor IDs and provenance. Include
strict TS and fixed-gate diagnostics already exported by the evaluator.
Use two no-text definitions as distinct interventions; neither is independently
trained article-free forecasting. Present legacy random shuffles as
diagnostics: some donors can be future dated. They do not establish deployable
availability. Design T4 below supplies a stricter content control.

Primary campaign metric: **140-bar endpoint close-return MAE in bp**.
Secondary: close endpoint MAE at 35/70, DHR on nonflat realized returns with
eligible counts and training-majority baseline, all-channel path MAE/MSE,
and matched-minus-shuffled endpoint error. Zero return is an error baseline;
it abstains from predicting direction, so its directional hit rate is not
an always-up comparator.

For promotion rank validation MAE; advance at most two candidates with at
least 1% relative improvement over the new ref. Treat this threshold as a
prespecified practical target, not a power calculation. If none qualify,
report that and stop broad GPU search; the CPU readout and timing audit are
the next decisions. Do not choose by published old test scores or by new
test metrics. The legacy trainer emits test summaries while training;
quarantine those files during screening. The checkpoint is still selected
only by validation task loss. The campaign's paired screen evaluates `vali`.

Before confirmation save a selection JSON with exact variants (include ref),
validation evidence paths/hashes, rule, code commit and timestamp. Freeze all
hyperparameters. Run selected arms and ref in seeds 1 and 2; evaluate the
locked models on test including the existing seed-0 screen checkpoints.
The generic worker can make a fresh seed-0 replay if requested; prefer a
separate evaluation of its verified screen checkpoint to save training.
Do not tune again after test. Since this test split has already informed the
project, call these prespecified follow-ups on an existing holdout, not a
previously untouched confirmatory test.

Use paired date-block bootstrap intervals (1,000 resamples, 20 event-date
blocks, sensitivity 5/10/40), resampling all reports on a date together.
Report every seed and their mean/range; checkpoint-conditional intervals
do not quantify training-seed uncertainty. For the fixed primary family use
Holm adjustment when making multiple superiority claims. No best-seed
reporting. For semantic claims require useful matched-text gains over
shuffled controls as well as gains over ref/zero return; otherwise describe
an architecture/training-package gain. Record negative and failed arms.

## Eleven full designs that need additional implementation or evidence

These are deliberately **not runnable matrix variants**. Each has a defined
question and acceptance gate. Implement on a focused family branch, test,
commit a resolved config, then register a new execution attempt. They must
never be submitted as if the current loader already supports their flags.

### T1: clock-ledger — establish what was actually public

Question: for which retained events can we verify availability of the exact
article/version used? Owner Nicole; evidence work first, no GPU. Build a
versioned ledger with event ID, article content hash, canonical URL, archived
version or receipt hash, published_at_utc, updated_at_utc, first_seen_at_utc
where available, source timezone/UTC offset, evidence URL, evidence kind,
confidence, raw price timezone, bar-start/end convention, and exclusion reason.
Keep private article content out of Git. Require explicit evidence for the
version of a summary: the source's original publication time cannot date
later edits or an unverifiable derived summary. Unknown fields stay null.
Do not infer article availability from its event's official calendar time.

Verify the price clock and DST/bar semantics independently. Under verified
bar-start labels, a bar's OHLC becomes available at its end, not its start.
The first feasible decision origin is the first executable grid boundary
after exact article availability plus a prespecified 5-minute processing lag.
Inputs use only completed bars at that origin; targets start strictly after
the decision boundary. Gate: evidence-backed clock and version checks,
monotonic windows, no cross-split bars, training-only preprocessing, coverage
and exclusions by category/era. If coverage is small, report feasibility
and descriptive uncertainty; do not manufacture a full-population benchmark.
This is a new ledger, not a relaunch of the canceled eight-announcement job.

### T2: verified-clock-delay — predict after verified arrival

Question: does fresh content help once it could actually be consumed?
Dependency T1 and a timestamp-aware loader/evaluator. On the same verified
cohort train ref and lora8, independently trained return-only, and a small
readout. Compare matched text, last older available same-category article,
metadata only, and no text. Fix origins within each paired comparison;
score origins at availability+5 minutes as primary and +15/+30 as latency
sensitivities. Moving the origin changes targets: compare models *within*
each latency, never subtract different targets as a causal text effect.
Repurge boundaries, create a new snapshot, match seeds/budget and recompute
training-only scales. Metrics/intervals follow the campaign. Gate: no use of
text versions published after origin, adequate declared coverage, full replay
assertions and smoke. Different cohort scores are not paired with full C1.

### T3: date-upper-bound — stress an explicit date-only assumption

Question: do conclusions survive an intentionally late assumed availability?
For reports with trustworthy calendar date but no intraday evidence, assume
availability only after that local date has ended, then process at the next
tradable boundary with completed bars. Add one-session and two-session
delays as prespecified stresses. This is conditional on the calendar date
being correct, not certified availability. For every delayed origin retrain
ref and return-only on identical targets; add matched/shuffled/older text.
Preserve the original snapshot and publish a derived audited version.
Gate: trustworthy date, calendar/timezone implementation, coverage and
repurging; unknown or conflicting dates are excluded explicitly. Report
latency versus predictive value without asserting an intraday leakage cure.

### T4: past-donors — a stronger article-content control

Question: is matching useful beyond category, era and language style?
Implement deterministic earlier-date donors from the same split/category
within the previous 90 calendar days; exclude same-date and exact duplicate
content, and apply a near-duplicate threshold defined on training only.
Use a frozen eligibility intersection across arms/draws with donor coverage
and exclusions. Three seeds; no fallback to future or other-split articles.
For verified-time data additionally require donor version availability before
origin; for date-only data label this chronological sensitivity only.
Keep targets/checkpoints fixed and compare real versus donor losses and
prediction-change norms. Gate: donor assertions and exported IDs, paired
population identical across models. No donor text may enter training from
validation/test. Failed eligibility is reported rather than silently relaxed.

### S1: ridge-readout — test whether the text has a simple usable signal

Question: can a small model exploit article features without fusion? CPU
first, using verified frozen 1B pooled features. Predict close returns at
35/70/140 separately; primary H=140. Arms: zero return; training-median
intercept; lag returns/realized volatility; price plus calendar/category;
same plus matched embeddings; same plus within-training reassigned embeddings.
At inference also swap the matched model's embeddings using T4 donors.
Features: last 1/5/10/35-bar close moves, trailing OHLC ranges and realized
volatility, cyclic month/day-of-week and one-hot category. Standardize numeric
features on training, handle unknown categories explicitly. Optionally fit
PCA=32/128 on training embeddings; choose PCA and ridge alpha from
{.01,.1,1,10,100,1000} by validation MAE with a fixed deterministic tie rule.
Refit on training only after choice; do not fit preprocessing on train+test.
Gate: aligned IDs/targets, training-only transforms, source-cache signature,
date-block intervals and a serialized small config. Gains over metadata
and reassigned features motivate more fusion work; a tie is informative.

### S2: train-shuffled — separate text meaning from training regularization

Question: does forecasting need correct article pairing during training?
Train ref and lora8 with seeded category/era-matched, different-date training
donors; every text-derived sentence report and associated training auxiliary
must follow the donor consistently. No validation/test donor in training.
Use the same fixed training permutation per seed, matched train steps and
validation policy; keep evaluation real-text inputs fixed and also run T4.
This is a retraining control, unlike inference swaps. Gate: donor provenance
for every text input, no hidden original sentence reports, gradients/smoke,
then three-seed matched evaluation. If matched and reassigned training tie,
do not attribute an overall gain to article meaning.

### S3: return-only — a fair retrained article-free comparator

Question: do these multimodal packages beat an independently learned return
forecaster? Implement a return-aware counterpart of the existing price-only
trainer. Use each seed's compatible Stage-1 TS weights, matched event updates,
R3 task objective and last-value design, with no text encoding, article
warmup or text-conditioned price features. Train L=35/140/280 separately.
It changes the applicable stages/parameters and is a declared package
comparison. Gate: a clean text-free computational graph, exact same event
targets/scales, no accidental old price checkpoint reuse, and zero-return
comparison. Report training-time savings and independent validation selection.

### C1: multiscale-780 — longer history without raw long-sequence fusion

Question: do slow regimes help beyond 280 raw bars? Build a causal dual
stream: most recent 35 five-minute bars plus 156 aggregates of five observed
bars, each completed strictly before the same origin (up to 780 observed
bars). Aggregate OHLC correctly (first open, maximum high, minimum low, last
close), encode coarse history with a shared frozen encoder and a separate
projection, then fuse with a small learned gate. Include masks and time gaps;
do not blend across sessions unknowingly. Controls: coarse-only, fine-only,
dual stream, and a parameter-matched fine-only gate; freeze forecast origins.
Audit the full union of bars at partition boundaries and version the common
cohort if needed. Gate: aggregation causality/shape tests and GPU smoke.
This is an architecture package, not a pure L ablation.

### G1: volatility-gate — test a past-only regime signal

Question: can price-derived uncertainty decide when text helps? Add only
trailing 35/140-bar realized volatility and completed-bar range to a gate
MLP; standardize and define quantiles from training only. Compare learned
gate with and without features, validation-chosen constant gate, and
feature-shuffled gate. Keep text/TS experts and objectives fixed. Use S3
for independent price-only context. Gate: no future volatility/realized
target surprise enters inputs; log calibration against realized held-out
utility, gate distributions and sensitivity. Low historical volatility
subgroup gains reduce priority; this experiment does not assume high
volatility implies useful news. Three seeds only after validation promotion.

### A1: dapt-lora — financial domain adaptation before forecasting

Question: does train-only language adaptation help more than direct LoRA?
Using only C1 training-date articles and permitted earlier training corpus,
fit q/v adapters by causal LM loss for one epoch, max 1024 tokens, rank 8,
alpha 16, dropout .05, LR=1e-5. Exclude held-out and near-duplicate copies,
and cap tokens/updates at a committed count before launch. Compare direct
Stage-3 LoRA, domain-adapted then frozen adapters, and domain-adapted then
Stage-3-trained adapters. All forecast arms share the same training budget;
extra language updates are disclosed. The frozen Llama pretraining corpus
itself is not newly certified. Gate: corpus hashes/date filters, LM-loss
adapter objective implementation, no static adapted cache, checkpoint replay.

### E1: calibrated-ensemble — reduce variance after model selection

Question: can validation-weighted shrinkage toward zero outperform a single
trained forecast? Freeze selected ref/LoRA/context outputs from seeds 0/1/2.
Fit nonnegative weights summing to one, including zero return, using only
validation MAE. Compare equal-weight ensemble, validation weights and
zero-shrinkage of the best single model. A single shared weight vector across
horizons is primary; no event-type fitting unless newly prespecified.
Export weights before test. Gate: validation/test separation, aligned IDs,
no test-derived calibration, all seed outputs retained. An ensemble gain
does not prove more semantic text use; repeat text controls on the ensemble.

## Execution, budget and result delivery

Use a separate pinned checkout and fresh outputs outside it. The renderer
uses a private JSON profile containing absolute resource paths, PBS
single-GPU/memory/CPU directives, a walltime directive and environment setup.
Example schema (replace placeholders locally; never commit operational details):

```json
{
  "checkout": "/ABSOLUTE/PINNED/CHECKOUT",
  "python": "/ABSOLUTE/ENV/bin/python",
  "snapshot": "/ABSOLUTE/c1-screened-return-v1",
  "moment": "/ABSOLUTE/MOMENT-1-large",
  "text_model": "/ABSOLUTE/Llama-3.2-1B",
  "run_root": "/ABSOLUTE/PRIVATE/RUNS/improvement-v1",
  "pbs_directives": ["#PBS -l YOUR_SINGLE_GPU_RESOURCE", "#PBS -l walltime=06:00:00"],
  "setup": "source /ABSOLUTE/ENV/ACTIVATE",
  "walltime_hours": 6
}
```

```bash
# In an appropriate CPU allocation, before rendering/submitting GPU work:
for L in 35 140 280; do
  python scripts/audit_c1_protocol.py --snapshot-root "$SNAPSHOT" \
    --forecast-task return_bp --timing-policy category4_1400_strict_before \
    --seq-len "$L" --pred-len 140 \
    --out "$FRESH_PRIVATE_AUDIT_ROOT/L${L}.json" \
    --windows-csv "$FRESH_PRIVATE_AUDIT_ROOT/L${L}-windows.csv" || exit 1
done
# Verify all emitted counts and identical origins/targets before continuing.
python scripts/improvement/campaign.py render --profile "$PRIVATE_PROFILE" \
  --out "$PRIVATE_PACKAGE_ROOT/screen-v1" --slots 2
# Install the rendered directory at the printed resource_dir, inspect scripts,
# check all current owner jobs, then submit from that directory:
python submit.py --available-slots 2
```

Render/submit **once**, not one campaign per variant. PBS dependencies create
two serial lanes; a failed task blocks successors in that lane. If one slot
is already occupied, use one available lane and keep the other job unchanged.
The submitter journals every scheduler response and refuses duplicate starts.
Existing active jobs must fit the global two-job limit; don't release old
Delta jobs or cancel another owner's work. There is no automatic resubmission.
For partial submission or lane failure inspect the queue/journal, collect the
failure, cancel only campaign-owned blocked descendants if appropriate, and
create a new named attempt with remaining work. Never delete the journal.

Default six-task screen at a 6-hour job cap: **at most 36 GPU-hours** including
audit, smoke, training and evaluations. No measured speed claim is made.
If up to two candidates are promoted, ref plus two candidates at seeds 1/2
add at most six jobs/36 GPU-hours. Locked seed-0 test re-evaluation needs an
explicit separate bounded job. Optional arms have no default compute
authorization from this file; choose a named bounded batch after inspecting
screen evidence and the operator's current resource limits. The renderer
caps batches at 18 tasks and jobs at 8 declared hours. Stop on systematic
failure instead of burning the rest of a lane.

Each worker verifies hashes, emits a real-input audit and windows, runs the
smoke, trains to a new output directory, and exports both paired controls and
return analyses. It marks execution pending scientific review. Complete the
provenance with scheduler ID/accounting, elapsed/peak VRAM, dependency list,
exact command, environment and backbone revisions/hashes, adapter/projection
gradient evidence, checkpoint hash and all artifact checksums. For every
completed/failed attempt update its registry ID using the existing reporting
skill. Publish small results/metrics/figures in a focused PR and the Research
Home registry; verify main after merge, then archive the reviewed aggregate
pack in Drive's `Final Deliverables / 01 Results and Evidence` and read back
the receipt. Keep checkpoints, raw predictions and private logs off GitHub.

These designs are hypotheses. No arm is promised to improve performance,
and no calendar-time assumption is presented as verified availability.
