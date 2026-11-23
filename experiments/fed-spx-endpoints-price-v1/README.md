# Compact price-only control at 20 and 30 minutes

Owner: Adaeze Ilodianya. Run ID `fed-spx-endpoints-price-v1`.

Does the existing compact CNN improve on zero return when the target is the
20- or 30-minute SPX Close return after the forecast origin? This run uses the
price-eligible Fed/SPX cohort. A matched evaluation of text uses a refit on its
own common, text-eligible train and validation events.

The [endpoint protocol](../../docs/experiments/FED_SPX_SHORT_HORIZON_PROPOSAL.md)
defines the fixed window, label, input timing and weights. The label builder
checks XNYS sessions using `exchange-calendars==4.13.3` and preserves the v1
train/validation split. The old 35 OHLC input bars are normalized to their
last Close. The labels use the Close at the decision time as their denominator.

The [config](config.json) sets seeds 0–2, 30 epochs, batch size 64 and AdamW.
Use one input RMS scale and separate target RMS scales fitted on train. The
loss is Smooth L1 with equal horizon weight after scaling and unit total
weight per release instant. Select each checkpoint by lowest weighted
20-minute validation MAE over epochs 0–30, breaking ties by earliest epoch.
Report both horizons using that checkpoint, including weighted direction on
nonzero targets. Compare the seed-average per-event MAE with the paired zero
forecast by date blocks. The held-out split is reserved.

With a CUDA-enabled PyTorch, NumPy, pandas and `exchange-calendars==4.13.3`
environment, use the frozen `SP500.csv` and a new ignored output directory:

```text
python scripts/run_fed_spx_endpoints_price.py --price-csv downloads/SP500.csv --out-dir downloads/fed-spx-endpoints-price-v1 --device cuda
```

Keep checkpoints and per-event predictions in local or agreed storage. For
the historical text audit, install the Fed catalog tool's HTML/PDF packages
(`beautifulsoup4`, `lxml`, `pypdf`, `requests`), then run:

```text
python scripts/audit_fed_spx_text.py --population downloads/fed-spx-endpoints-price-v1/population.csv --out-dir downloads/wayback-text-v1
```

Only matching archival extracts with a capture timestamp at or before the
cutoff enter the verified-text directory. Record coverage and inaccessible
replays as part of the timing audit. The optional `--index availability`
mode checks the nearest timestamp through the separate Availability API. A
nearest capture after the cutoff leaves earlier captures unresolved if the
as-of index was inaccessible. Each pass stops sending requests on HTTP 429.
