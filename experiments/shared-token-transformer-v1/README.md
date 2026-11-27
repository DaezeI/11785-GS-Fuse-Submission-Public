# Six-position shared-query Transformer

Owner: Nicole. Implementation requested by the researcher reviewing the final-project backbone.
This branch depends on the rolling-price loader in PR #72. Use the immutable runtime
commit recorded with the result, rather than a moving checkout.

`none+Transformer128x4~none~none @ train:SPX-PX6-GRID5-v1 → score:SPX-PX6-GRID5-v1 val`
uses all 204,694 eligible training windows and 40,548 separated validation windows.
The six outputs are cumulative Close log returns in bp at +5/+10/+15/+20/+25/+30
minutes. All six positions are supervised. Membership and the original five-minute
processing gap are inherited from the audited rolling-price protocol; there is no
text/event filter or constituent-stock panel. No test cohort is constructed.

The [configuration](config.json) fixes four pre-LN Transformer layers, width 128,
four heads, FFN width 512, and a shared linear final readout. Exactly one trainable
128-vector initializes all six prediction positions. Past and future positions use
the same sinusoidal temporal encoding. Overnight and DST gaps use elapsed UTC
minutes. Past states cannot attend to future query positions; future states can
attend to each other and all historical states. None receives future prices.

On one H200, calibrate the largest confirmed unique batch with actual forward,
backward, clipping and AdamW updates, including initialized optimizer states.
Search both ordinary and checkpointed training if necessary, reserve at least
5% device memory or 4 GiB, and confirm the selected size for ten updates. Attention
batch-axis tiling controls kernel/workspace size; it does not change the logical
batch or accumulate gradients over separate backwards. If all training windows
fit, the batch is limited by the dataset, not an asserted universal hardware limit.

Calibration states are discarded. Training starts from the saved initial state
with a separate budget of 1,000 optimizer updates, capped at 90 minutes. Record
every optimizer update and full train/validation diagnostics every 25 updates.
Check the resulting loss curve for convergence; the budget alone is not evidence.
A time-limited run must retain its partial status and continuation checkpoint.

```sh
python scripts/train_prediction_token_transformer.py --prices /path/to/SP500.csv \
  --out /fresh/output --code-commit COMMIT --max-updates 1000 \
  --eval-every 25 --max-training-seconds 5400
```

Compare with `Persistence` on identical rows. Comparing the common 20/30-minute
endpoints to the earlier CNN changes architecture, supervision and optimization
together. It cannot attribute a gain to any one component. This reused validation
is exploratory; date-block intervals condition on the selected checkpoint and do
not account for selection. It cannot establish the benefit of adding text.

## Verified batch calibration

The [ten-update confirmation](batch-confirmation.json) passed with 204,694 unique windows, activation checkpointing and BF16 autocast. Peak allocated memory was 24,140,269,056 bytes; peak PyTorch reserved memory was 32,975,618,048 bytes. This is dataset-limited full-batch training. Formal training was running at the last check. Repeated local polling stopped at the user’s request; final accuracy and convergence evidence remain pending collection.
