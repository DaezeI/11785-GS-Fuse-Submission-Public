# Screened C1 price baseline: first full result

**Owner:** Nicole. **Status:** training and both full paired evaluations completed;
execution checks passed. One checkpoint, seed 0, 2,236 test events across 845 dates.

## Question and finding

Does correctly matched article text help this price model beyond shuffled text
or article-free inference? Correct and shuffled text have nearly identical MSE
at all three horizons. Their date-block bootstrap intervals include zero.
This run provides no evidence that correct article matching improves MSE.

At H=140, real-text MSE is 0.01137609 and shuffled-text MSE is 0.01137603
in training-standardized price units. Removing article tokens gives 0.01133556;
fusion bypass gives 0.01115690. These are two different inference interventions.
The bypass gain interval includes zero at H=140. Do not call either control an
independently trained TS-only baseline.

## Paper assets

Use metrics.csv for the numeric source, metrics.tex for the paper table, and
text-control-gains.pdf or .svg for figures. source_metrics.json preserves the
small aggregate inputs; provenance.json identifies the code, data and checkpoint.
Rebuild with `python results/export_price_result.py` from the repository root.

## Interpretation and limits

This supports a narrow negative result for this price checkpoint, not a claim
that financial text is generally useless. Retained article release times are
unknown. Uncertainty is conditional on one training seed and three shuffle
samples with 20-date bootstrap blocks. Check block-length sensitivity and other
training seeds before strengthening claims. Shuffling is a diagnostic placebo
and can use future-date donors. Baseline and return metrics require common
units and aligned examples; current architecture experiments should use R3.

Large model and prediction artifacts are retained by Nicole. A second,
team-accessible archive remains to be completed. Cluster account/access details
are kept out of this result folder.
