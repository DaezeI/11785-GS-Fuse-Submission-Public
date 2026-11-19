# Adaeze’s R3 return model on one H200

**Completed November 18, 2026.** Adaeze designed the experiment. Nicole ran
it on her personal GPU resource. The job finished at about 7:16 p.m. Eastern
after 45 minutes 35 seconds. All three training stages, checkpoint reload,
both full article checks and checksum verification succeeded.

## What we learned

**The correctly matched article barely changed error.** R3’s average
absolute error was 80.615 basis points with correct articles and 80.618
with shuffled articles. The estimated relative improvement was 0.0037%,
with a paired 95% interval from −0.0026% to +0.0104%. At 140 bars, both
conditions got the direction right on 52.37% of eligible events.

The simple forecast of zero return scored 80.487 basis points. That is
0.128 basis points lower error than R3 with correct articles. This is a
descriptive comparison. Removing article tokens at prediction time also
slightly lowered error. The results point to little measured value from
the particular article for this training seed and test split.

| Forecast condition | Mean absolute error (basis points) |
| --- | ---: |
| R3 with correct articles | 80.615 |
| R3 with shuffled articles | 80.618 |
| R3 with article tokens zeroed | 80.432 |
| R3 with fusion bypassed | 80.680 |
| R3 with a fixed gate | 80.327 |
| Predict zero return | 80.487 |

Errors average the four OHLC channels and all 140 forecast steps.
[Metrics and date-block intervals](metrics.json) also include squared error,
direction accuracy and results by event group. [Figure](article-comparison.png)

## What ran and what the comparisons show

Adaeze’s R3 adds a learned last-value shortcut to R2’s return model. This
replay used the exact executable source revision of her Delta campaign,
[e051195](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/tree/e051195ce50f496c8877f724e8dc16a28c0ff6b8),
and matched the [owner's training protocol](../../docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md).
It trained with batch size 8, seed 0, and two epochs per stage. The first
epoch had the lowest validation task loss, so its checkpoint was used.
Both paired checks used the same checkpoint, 2,236 held-out events across
845 dates, and three shuffled article assignments. The frozen split,
environment, configuration and checksums are in [provenance.json](provenance.json).

- The comparison with shuffled articles changes the story given to the same
  model. Donors can come from later test dates, so this is a diagnostic.
  Retained article release times remain unverified.
- Zeroing tokens, bypassing fusion and fixing the gate change how the
  already-trained model predicts. The intervals resample 20-date blocks
  and describe this checkpoint and these shuffle draws.
- On the same frozen test set, Adaeze’s earlier R2 on A100 scored 81.080
  basis points and R3 on H200 scored 80.615. This comparison changes
  hardware and environment along with the shortcut, so it records the
  observed difference of 0.465 basis points. The Delta campaign remains
  the source for the matched-hardware comparison.

The full phase through the selected-checkpoint evaluation took 23 minutes
40 seconds. Article checks and checksums took a further 15 minutes
41 seconds. The earlier preflight attempt exited on a mistyped price
checksum before training. The retry used the exact checksum from Adaeze’s
script and verified the frozen inputs from scratch.

The aggregate pack is suitable for the team’s Results folder. Nicole retains
the checkpoint and full predictions in private GPU storage. Their hashes
are recorded with the result. The separate Delta jobs were untouched.

## What to carry into the paper

Show zero return as the plain reference beside R3 and keep the correct and
shuffled comparison in the same table. Collect Adaeze’s existing Delta
checks when available. Use the same-hardware R2 and R3 comparison to judge the
shortcut.
