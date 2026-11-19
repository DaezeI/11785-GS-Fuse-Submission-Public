# Replacement-news dataset audit

Owner: Adaeze Ilodianya
Status: Access pilot complete; modeling blocked. No training performed; screened C1 unchanged.

The validation agent reported full publisher text access via CMU TDM Studio. It inspected 10 distinct WSJ records from one date and reported four public article-to-cash-SPX pilot joins across three dates in 2023. The pilot does not establish a training-ready dataset.

Still needed: evidence of the exact article version available before each forecast, continuous intraday cash-SPX bars with documented time zone and bar boundaries, and permission for research use and model exports.

Next: Check CMU Libraries entitlements before collecting more data or training. No publisher text or raw prices are included in this note.

## Dataset acceptance checklist

1. **Relevant full text.** Use original publisher article bodies about U.S. market-wide or macro events, not headlines or hand-written summaries. Keep a source ID, version ID or hash, and deduplicate syndicated copies.
2. **Text available before the forecast.** Verify when that exact body was published, captured, or first seen, with a documented time zone. Use a conservative availability time and account for processing lag. A date, an original publication timestamp on a revised body, or an undated current copy is insufficient.
3. **Matched cash-SPX prices.** Use actual S&P 500 cash-index 5-minute OHLC bars, not SPY, futures, or daily closes. Document the source, time zone, trading sessions, and whether timestamps mark bar starts or ends. Keep only events with complete input and target windows; no price input may use a bar that ends after the cutoff.
4. **Enough independent events.** Count retained events and dates after all checks, show why rows were excluded, and verify enough coverage for chronological train, validation, and test splits with no overlapping price windows across splits. Set the minimum before training. Four pilot joins are not enough.
5. **Permission for the actual workflow.** Verify rights for both sources and whether full text may be processed on the local GPU. If it must stay in a restricted workspace, confirm permitted computation and which outputs may leave it, including metrics, features, and model weights. Do not commit source data.

**Decision:** All five checks must pass before training a replacement benchmark. Keep screened C1 frozen and record the source, evidence, and reason for any failed check.
