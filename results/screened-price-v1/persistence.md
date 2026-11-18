# Seed-0 price persistence comparison

Completed November 18, 2026 as a CPU-only reanalysis of the saved screened
price predictions. A flat forecast that repeats the last observed value of
each open, high, low and close channel had 20.81%, 15.85% and 8.21% lower
prefix MSE than the matched-article model at 35, 70 and 140 bars. The pairing
uses the same 2,236 examples, 845 event dates, targets and training-standardized
units. All paired 95% intervals favor persistence for 5-, 10-, 20- and
40-date blocks.

In the same archive, the root mean squared forecast change induced by an
article swap was 0.47%, 0.45% and 0.37% of the model's forecast RMSE at those
horizons. Approximately 49% of each draw's donors were dated after the
recipient. These are diagnostics on seed 0 and the existing three swaps.
The repeat's full prediction archive was not used in this reanalysis.
Article release times remain unverified.

See [aggregate metrics](persistence.json) for the full intervals and
[run provenance](provenance.json) for the checkpoint, snapshot and price
fingerprints. The saved matched-article MSE agrees with the published
[metrics](metrics.csv) up to floating-point rounding. The original archive
stays in the campaign's saved artifacts.

## Reproduce

From the repository root, use the verified seed-0 prediction archive with
the paired zeroed-article control. The script reads its matched and shuffled
predictions and observed input prices. It checks the archive hash and reuses
the existing circular date-block uncertainty calculation with 1,000 draws.

```sh
python3 results/analyze_price_persistence.py /path/to/zeroed_article_tokens.npz \
  --sha256 fcc5f06018ef63c5137be2b74d7b3777c8a7ce3bc0f92775f056c1be5921a84f
```
