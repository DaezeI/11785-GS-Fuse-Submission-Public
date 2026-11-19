# Compact GS-Fuse: forward direction

The compact model is the development target, not an approximation that must be
confirmed on R3. Keep the independently trained 35-bar compact price-only model
as the reference, with the current event-summary residual as an experimental arm.
See the [verified baseline results](RAW_PATH_TEST_RESULTS.md),
[architecture changes](COMPACT_ARCHITECTURE.md), and
[portable runbook](COMPACT_PORTABLE_RUNBOOK.md).

The screened test result is final for this configuration. Do not use it to
select seeds, context lengths or text settings. Move follow-on designs to the
compact ablation branch. Evaluate them on validation, record any changes to the
eligible population, and report coverage and paired deltas. Launch status
requires an actual job/log, not just a runnable script.

First comparison: 35 versus 140 observed price bars, same model family, matched
events and forecast origins, with cross-split bar isolation rechecked. Later
candidates: summary length, volatility-conditioned news gating and matched vs.
shuffled-text training. Do not bundle these changes together. News availability
at forecast time remains unverified.
