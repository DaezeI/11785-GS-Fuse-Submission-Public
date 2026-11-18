# Seed-1 replication

The correct article showed no clear forecasting advantage at any of the three forecast lengths. At 140 bars, the correct article’s estimated improvement in mean squared error was +0.0085% (95% interval -0.0061% to +0.0244%).

Correct-versus-shuffled improvement (positive favors correct text):

| Horizon | Gain | 95% date-block interval |
|---|---:|---:|
| 35 | 0.0084% | [-0.0102%, 0.0293%] |
| 70 | 0.0102% | [-0.0074%, 0.0298%] |
| 140 | 0.0085% | [-0.0061%, 0.0244%] |

The two article-free controls and all metrics are in [metrics.csv](metrics.csv). Intervals are conditional on this checkpoint, not across-seed confidence intervals.

Each interval describes one trained model and three shuffle draws. Training effects can vary across seeds. The release times of retained articles remain unverified. The shuffled article is a diagnostic comparison and can come from later test dates. The article-free controls remove text during prediction. Training a price-only model is a separate comparison.
