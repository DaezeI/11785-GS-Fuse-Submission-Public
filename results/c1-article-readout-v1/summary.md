# Do frozen article embeddings improve a simple return forecast?

**Completed November 18, 2026.** Nicole’s readout found no held-out improvement
from adding pooled article embeddings to prices and metadata. Replacing the
article at test time also made little difference. The run used the frozen
screened C1 split and completed in 3 minutes 3 seconds.

## Held-out results

All models forecast the same 35-bar endpoint close log return on
2,191 eligible events across 834 dates. Lower mean absolute error is better.

| Forecast | Test MAE (basis points) |
| --- | ---: |
| Zero return | 63.795 |
| Price only | 64.320 |
| Price and metadata | 64.462 |
| Price, metadata and matched article | 64.721 |
| Model trained and scored with reassigned articles | 65.293 |
| Matched model with reassigned articles at test time | 64.741 |

Positive relative improvement means the matched-article model had lower error
than the listed comparison. The intervals resample paired 20-date blocks.

| Compared with the matched-article model | Relative MAE improvement | Paired 95% interval |
| --- | ---: | ---: |
| Price and metadata | -0.401% | -1.101% to +0.215% |
| Model trained and scored with reassigned articles | +0.877% | +0.102% to +1.712% |
| Matched model with reassigned articles at test time | +0.032% | -0.523% to +0.570% |

Training with matched articles did better than training and scoring with
reassigned articles. The matched model still had higher error than prices and
metadata alone. Giving the matched model a different article at test time
changed MAE by about 0.02 basis points, with an interval spanning zero.

## Design and scope

The price input consists of the 35 observed OHLC bars in basis points relative
to the last observed close. Metadata adds event type, weekday and month. The
article feature uses the frozen Llama-3.2-1B pooled summary embedding. All
10,694 article rows used across the three splits hit the existing cache.
Training-only PCA reduced the embeddings to 64 dimensions. The training split
also fitted the scalers and category encoding. Validation MAE selected the
ridge penalty separately for each model.

Donors came from the same split, event type and calendar year, with a strictly
earlier recorded date. The rule excluded normalized exact duplicates and
word five-shingle Jaccard overlap of at least 0.80. All arms used the same
6,060 training, 2,250 validation and 2,191 test events. There were 45 excluded
test events without an eligible donor. The reassigned training control uses
seed 23. The matched model’s test-time reassignment averages per-event
absolute error over seeds 23, 24 and 25. Exact report release times remain
unverified. These results apply to this pooled representation, readout and
horizon. Compare Q-Former with cross-attention on the separate matched R3
architecture experiment.

[Metrics and block-length sensitivity](metrics.json) · [Provenance](provenance.json) · [Analysis source](article_readout.py)
