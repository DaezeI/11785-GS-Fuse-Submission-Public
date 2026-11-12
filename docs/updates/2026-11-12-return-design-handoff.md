# C1 return forecasting decisions and implementation handoff

Adaeze's handoff to Nicole, November 12, 2026. Status: design accepted;
return training implementation and GPU validation pending. This change supplies
the implementation contract; it does not make the existing launch scripts run
the new task.

## Submission decision

The team's redesigned experiment predicts cumulative log returns and uses the
complete magnitude-plus-direction forecasting loss to supervise the gate.
The existing date-screened launch script still predicts standardized prices.
Do not submit that script expecting the redesigned return baseline.

Nicole can continue the date-screen audit and compute/environment checks now.
Submit the return baseline after the implementation and smoke checks below pass.
A separately named price baseline may remain a useful reference, but it is a
different experiment and consumes additional training budget.

## Repository update checked

The remote was fetched for this review. Shared `main` at `2bb71c1` already
contains C1 integration, all-stage loader coverage, paired text controls, and
Jessica's padding fixes. The older local checkout was 26 commits behind it.
Do not rebuild C1 loading from that stale checkout.

Nicole's branch `nicole/c1-article-date-audit` at `316455d` adds the
[article-date audit](https://github.com/DaezeI/11785-GS-Fuse/blob/316455d/analysis/text_value/ARTICLE_DATE_AUDIT.md),
screened manifest, audit tests, and replacement launch scripts. These changes
are not in the inspected shared `main`. Her audit records:

- Original full training was held while pending; dependent evaluations remain
  pending. This is a repository record, not a fresh scheduler query.
- The screen removes 198 validation and 684 test Reuters-dateline candidates,
  plus one source-verified additional test article.
- Expected retained counts are 6,167 train, 2,302 validation, and 2,236 test.
- The real-input window audit for this subset was still pending in that record.

Adopt that versioned screen only after its emitted-window audit passes. Preserve
the original dataset and manifest. The screen remains a timing-sensitivity subset:
unrecognized article dates, exact release times, price timezone, and bar labeling
are not certified. A log-return transformation cannot repair incorrect dates.

The [current C1 loader](../../data/c1_protocol.py) fits a training-only price
scaler and emits standardized prices. The [training entry point](../../train.py)
selects it with an explicit manifest/timing policy. The
[model](../../model/gs_fuse.py) still uses price regression losses and an internal
TS reference that may contain article-conditioned features. Return supervision
and the strict training reference require implementation.

## Accepted task definition

| Item | Decision |
|---|---|
| Scope | SP500, original cross-attention fusion and softmax gate; 35 past bars, 140 future bars; OHLC channel order Open, High, Low, Close |
| Historical input | For every OHLC value in the past window, use `10000 * log(past_OHLC / last_observed_Close)`; this preserves relative OHLC structure and removes nominal price level |
| Target anchor | Use that same scalar last observed Close for all four future channels: `R[h,c] = 10000 * log(future_OHLC[h,c] / last_observed_Close)` |
| Reconstruction | `future_price[h,c] = last_observed_Close * exp(predicted_R[h,c] / 10000)` |
| Scaling | No dataset-wide target centering or standardization initially. Basis points are a fixed unit conversion. Any later learned/fitted preprocessing uses training data only |
| Magnitude | Huber on all 140 horizons and four channels, averaged per event |
| Direction | Close endpoint movement at horizons 35, 70, 140; average their smooth directional penalties per event |
| Gate utility | Strict TS-only task loss minus full task loss, using identical magnitude/direction settings, targets, and reduction rules |
| Supervision target | Retain adaptive difference scaling and sigmoid initially; detach the resulting gate target |
| Shortcuts | Both existing explicit price shortcuts enabled for baseline, with separate flags; they now contribute in return units. Ordinary Transformer skip connections remain enabled |
| Volatility | First extension: append one pre-event close-volatility scalar to the gate; baseline omits that extra scalar |
| Event category | Retain metadata for subgroup evaluation. Explicit category embedding and MoE/Q-Former changes are deferred |

The common close anchor above settles the earlier per-channel-anchor alternative.
Every training stage, control, reconstruction helper, and checkpoint must record
this convention. Do not combine predictions from the two anchor conventions.

## Remaining numerical settings and bounded pilot

Freeze the method for selecting settings now; numerical data-dependent values
must come from the actual audited training set. Do not infer them from toy data.

1. Compute `s_bp = max(median(abs(training close returns at horizons 35/70/140)), 1 bp)`.
   Count each emitted training event once, save the population/hash and result,
   and freeze it for all stages and evaluation. This is a loss reference scale,
   not a centered target scaler; zero return still means no movement.
2. To keep direction from disappearing beside squared basis-point errors, define
   `L_mag = mean(Huber_d(predicted_R - R)) / s_bp**2`.
   Define `L_dir = mean(softplus(-sign(R_close) * predicted_R_close / tau_bp))`
   over the selected close horizons outside the actual-return deadband. An
   event with no eligible horizons receives zero direction loss. Exact-zero
   actual returns are always omitted from direction scoring.
3. Initial pilot: `d_bp = s_bp`, `tau_bp = s_bp`, `deadband_bp = 0`,
   `lambda_dir = 1`. Thus `L_task = L_mag + lambda_dir * L_dir`.
   These are explicit starting settings, not empirically optimal constants.
4. If runtime permits, compare only `d_bp / s_bp` in `{0.5, 1, 2}` with the
   other pilot settings fixed. Then compare `lambda_dir` in `{0.25, 1, 4}`
   at the selected threshold. Reuse the middle candidate rather than rerunning it.
   Limit short pilots to five distinct configurations. Do not search temperature,
   deadband, architecture, and the loss family simultaneously this week.
   Time the smoke first and cap pilots so the baseline, its evaluations and a
   final analysis day still fit the available allocation and deadline. Matching
   pilot budgets must be recorded; a short pilot cannot establish final convergence.
5. Select among those candidates by validation close directional hit rate averaged
   at 35/70/140, restricting eligibility to models whose validation all-channel
   return MAE is within 5% of the lowest candidate MAE. Break ties by lower MAE.
   Use the same zero deadband and evaluation population for every candidate.
   This makes the magnitude/direction tradeoff explicit. Do not compare candidates
   by their differently weighted raw training loss values.
6. Record the selected values and seed before final test evaluation. If resources
   permit only the initial pilot, retain its preregistered settings after sanity
   checks and disclose that no search was performed. Do not claim they were tuned.

Stage 1, Stage 2, and Stage 3 use the same forecast task. Existing auxiliary
alignment/gate coefficients need a logged gradient/scale check because changing
the forecast loss changes their relative influence. Record effective values;
do not silently retune them between controls. Keep the current gate sigmoid
gain/clamp for the first pilot and record its adaptive scale distribution.

## Implementation changes required before return training

| File or component | Required change | Acceptance check |
|---|---|---|
| `data/c1_protocol.py` and Stage-1 dataset adapter | Add an explicit return mode while preserving the existing price mode. Transform each window around its own last input close; return raw prices/anchors and stable IDs for reporting | No validation/test observations fit preprocessing; all windows and auxiliary report pools obey audited roles |
| New pure return/loss helpers | Transformation, reconstruction, pre-event volatility, per-event Huber, directional and full task loss | Hand-calculated positive/negative/flat examples; round trip; finite values and gradients |
| `train.py`, `smoke_train.py` | Explicit task/config options; use return mode in every stage; seed Python/NumPy/Torch/loaders; write resolved configuration | A run record identifies units, anchors, split/hash, task settings, stages and seed |
| `model/gs_fuse.py` | Strict TS reference from pre-cross-attention TS features; reuse decoder; full task loss for branch comparisons and forecast training; independent shortcut flags | Changing article text cannot change strict TS output; full-better gives target above 0.5; detached target; finite gate/decoder gradients |
| Stage-1/2/3, validation and test paths | Replace all price-only forecast scoring in return mode, including named TS auxiliary training. Give strict TS reference an explicit differentiable forecast loss when training it | Strict TS decoder path is trained, rather than used only under no-grad; every loss receives the same units and target convention |
| Evaluation and paired controls | Save aligned predictions, targets, anchors, IDs, categories, past volatility, branch losses, gate targets/weights; explicit inverse reconstruction | Correct/shuffled/no-text and learned/fixed gate use matching events/checkpoints/metrics |
| Checkpoint loader/saver | Save task metadata plus resolved loss settings/scale, shortcut flags, code/data versions | Reject price-space or incompatible-anchor stage checkpoint resume in a return run |
| Return launch config | Fresh output; disable unverified resume; use audited screened manifest; explicit task mode | Loading the launch configuration fails if return support is absent, rather than falling back to the price default |

Strict TS training must not reuse the text-conditioned `s_vec_residual`. Reuse
untouched TS encoder outputs and permitted raw historical shortcut inputs. Keep
the gate-target construction detached; use a separate differentiable TS forecast
loss if the strict branch shares trainable components. The existing inference
helper alone does not complete this training change.

For the volatility extension, calculate the standard deviation of the 34
consecutive log-close changes within the 35-bar input window, using population
standard deviation (`ddof=0`). Express it in basis points and divide by the
training median of those volatility values, floored at 1 bp. Append it to the
gate input. The extra feature is derived only from past bars; its scaling is
fitted only on training events. Run the same baseline configuration and training
budget with and without this feature. It does not require MoE.

## Submission conditions and outputs

Before full return training, the screened real-input audit, focused CPU checks,
and an end-to-end real GPU smoke must pass. The smoke must execute all stages,
save/reload the checkpoint, emit return forecasts and metrics, and run a paired
learned/fixed-gate comparison. Record the resource, runtime, memory, environment,
seed, effective training parameters and output access. Expanse access/allocation
readiness must be checked separately; Delta-tested settings do not establish it.

Do not use the historical price-trained Stage-1/2 weights as automatic resume
for the return task. Start fresh trainable projections/decoder stages unless a
separate compatible initialization experiment is explicitly recorded. Frozen
pretrained encoders and a provenance-matched frozen text cache may be reused.

Evaluate return MSE/MAE, endpoint close DHR at 35/70/140 with eligible counts,
reconstructed price errors, zero-return and training-majority-direction
references. Estimate uncertainty by event date because events share windows.
Learned versus fixed 50/50 at inference tests dependence on the learned mixture;
it does not replace a separately trained fixed-gate architecture comparison.

## Prepared repository package

This prepared package changes the design handoff and a link in the team workflow. The explanatory
LaTeX guide and older audit drafts remain local until their recommendations are
reconciled with this frozen contract and Nicole's new date screen. Avoid uploading
the entire dirty workspace, historical generated figures, private chat notes,
raw data, checkpoints, or unrelated Q-Former changes.

Implementation remains a subsequent focused change on the current C1 code.
Nicole owns the screened data audit; Adaeze supplies the gate/task contract;
coordinate the shared training/loader edits before combining implementation.
No scheduler job was submitted, released, or cancelled by this review.
