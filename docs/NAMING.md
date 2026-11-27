# Naming: cohorts, architectures, comparisons and controls

Adopted November 23, 2026 (Adaeze). **Refer to every model, data set and gate by its ID.**
Do not write "the model", "this run's model", "the data" or "the gate" in code comments, result
notes, PRs, chat handoffs or agent answers. A sentence such as "the model beat persistence" must
name the architecture ID, the training cohort and the scoring cohort.

New IDs are added to the tables here in the same PR that first uses them. Never change what an
existing ID means; create a new version instead.

## 1. Cohort IDs (which releases)

Format: `<SOURCE>-<CONTENT>[-<EVIDENCE>]-<CLOCK>-v<N>`

| Part | Values |
|---|---|
| `SOURCE` | `FED` Federal Reserve releases (catalog v2). `MACRO` U.S. statistical releases (planned). |
| `CONTENT` | `PX` complete price input and 20/30-minute targets. `PXTX` the same plus exact release text proven public by the text cutoff. |
| `EVIDENCE` (only with `PXTX`) | `ALL` every audited eligible class. `STRICT` official record or archive copy before the cutoff only. |
| `CLOCK` | `REL5` decision time = release + 5 min rounded up to the 5-minute grid, text cutoff 5 min before it. `OPEN` pre-market release, decision time 09:35, text cutoff 09:30. `REL5+OPEN` the union. |

| Cohort ID | Train / Val / Test rows | Defined by |
|---|---:|---|
| `FED-PX-REL5-v1` | 978 / 207 / 216 | `data/fed_spx_endpoints.py` (train/val), `scripts/build_fed_spx_test_population.py` (test) |
| `FED-PXTX-ALL-REL5-v1` | 160 / 123 / 121 | `results/fed-spx-news-cohort-v1/eligible_cohort.csv`, `results/fed-spx-test-text-v1/test_text_ledger.csv` |
| `FED-PXTX-STRICT-REL5-v1` | 90 / 27 / 26 | the rows above whose evidence class is official-record or pre-cutoff archive copy |
| `FED-PX-OPEN-v1` | 153 / 37 / 43 | `scripts/build_fed_spx_premarket_population.py` |
| `FED-PXTX-ALL-OPEN-v1` | pending | pre-market text checks (running November 23) |

A **row** is one release with its price window. Releases at the same minute share a **release
instant**. Instants are a weighting unit (each instant has total weight 1), not a cohort.

## 2. Architecture IDs (what the forecaster is)

Format: `<TextEncoder>+<PriceEncoder>~<Reader>~<Fusion>`

| Part | Meaning | Values used so far |
|---|---|---|
| `TextEncoder` | Turns release text into vectors. Add `(frozen)` when not trained. | `none`, `Llama-<size>(frozen)`, `<name>(frozen)` for a small encoder |
| `PriceEncoder` | Turns the 35 OHLC bars into vectors | `CNN` (`model/compact_return.py`), `MOMENT` |
| `Reader` | How text vectors become the few vectors fused with price | `none`, `Pool` (average), `CrossAttn` (price and text attend to each other), `QFormer` (a fixed set of learned queries read the text) |
| `Fusion` | How text changes the price forecast | `none`, `Add` (price + text correction), `Gate-Learned` (price + g × text, g learned from forecast error), `Gate-Granger` (g also supervised by the Granger target), `Gate-Const<g>` (fixed g), `MoEGate` (router over experts) |

Each ID names **one** design. Alternatives are separate IDs, never written with `|`.

| Architecture ID | Where | Notes |
|---|---|---|
| `none+CNN~none~none` | `fed-spx-endpoints-price-v1`, `fed-spx-matched-control-v1` | Price-only baseline. Three seeds, frozen. |
| `none+DLinear~none~none`, `none+Ridge~none~none`, `Persistence` | `fed-spx-matched-control-v1` | Baselines. `Persistence` forecasts a zero return from the decision-time Close. |
| `Llama-<size>(frozen)+MOMENT(frozen)~CrossAttn~Gate-Granger` | `model/gs_fuse.py` default | Original GS-Fuse. Also has `~QFormer~…` and `~…~MoEGate` variants (`model/variants.py`). |
| `<TextEncoder>(frozen)+CNN(frozen)~Pool~Add` | planned Fed/SPX text arm | Proposed by Adaeze for the text-model owner. Not yet built. |
| `<TextEncoder>(frozen)+CNN(frozen)~Pool~Gate-Learned` | planned | Trust-aware arm. |

## 3. Comparison labels

Every reported number states: `<ArchitectureID> @ train:<CohortID> → score:<CohortID>[ <split>]`

Example: `none+CNN~none~none @ train:FED-PX-REL5-v1 → score:FED-PXTX-ALL-REL5-v1 val`.

Run IDs (such as `fed-spx-matched-control-v1`) name a registered execution and its evidence. They
are not model names.

## 4. Baselines versus controls

- **Baseline:** an independent reference forecast with its own architecture ID (`Persistence`,
  `none+Ridge~none~none`, `none+DLinear~none~none`, `none+CNN~none~none`).
- **Control:** the *same trained* text architecture with one input or component neutralized, scored
  on the same rows. Write it as a suffix on the architecture ID:
  - `[text=zero]` text vector replaced by zeros.
  - `[text=shuffled-<seed>]` text from another same-split release that was public by this
    release's cutoff (frozen donor draws).
  - `[gate=0]` or `[gate=<g>]` gate fixed at inference.
- A control isolates what the text or gate contributes. A baseline shows whether the whole design
  beats simpler forecasts. Report both, and never call a baseline a control.

`fed-spx-matched-control-v1` predates this rule. Its CNN is the baseline `none+CNN~none~none`;
the text arm will reuse it as its frozen price encoder.

## November 24 Person 4 exact IDs

`FED-PXTX-ALL-REL5-v2`: 173 / 149 / 172 releases, defined by the merged v2 ledgers and verified-text manifest. The #55 training configuration drops the two documented truncated train texts (171 used); validation is unchanged. Preserve that exclusion in every comparison label/provenance record.

`Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add`: Nicole's exact #55 architecture, implemented in `scripts/run_fed_spx_text.py`, using the frozen `none+CNN~none~none` checkpoints. Training and validation completed November 26; see `results/fed-spx-text-llama31-8b-v1/summary.md`. Test remains reserved. The registered pilot records token budget, preprocessing and evaluation controls.

## November 26 rolling-price cohorts

`SPX` denotes the existing cash-index OHLC file, without an event/text filter.
`GRID5` and `GRID15` select hypothetical release anchors every 5 or 15 minutes,
aligned to 09:30 ET. The decision time is five minutes after that anchor,
preserving the original 35-bar information cutoff and 20/30-minute labels.
No component-stock panel is implied.

- `SPX-PX-GRID5-v1`: 204,694 training windows and 40,548 validation windows,
  constructed by `data/dense_price_endpoints.py`. Full price support stays
  within each temporal partition. No test cohort is constructed.
- `SPX-PX-GRID15-v1`: 68,219 training windows, the 15-minute subset of GRID5
  training anchors. The initial training comparison selects and scores on
  the common `SPX-PX-GRID5-v1 val`, not a separate GRID15 validation score.

The architecture remains `none+CNN~none~none`. Dense-training RMS scaling and
the training population differ from the earlier Fed-event-only experiment;
report this as a training-population package comparison. Validation selection
uses the common rolling-price grid; `FED-PX-REL5-v1 val` and
`FED-PXTX-ALL-REL5-v2 val` are scored after checkpoint selection.

## November 26 shared nonlinear decoder implementation

- `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~MLP1-Residual`: pooled text is
  projected to 32 dimensions, concatenated with the 64-dimensional frozen
  price context, and passed through a joint MLP with one 32-unit GELU hidden
  layer and a two-output correction. The correction is added to the frozen
  price forecast. Implemented, not benchmarked.
- `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~MLP2-Residual`: the same design with
  two 32-unit GELU hidden layers in the joint decoder. Implemented, not benchmarked.

Both are opt-in `CompactReturn(fusion="joint_mlp")` variants. The text head
receives price context before its nonlinear layers, allowing price-dependent
text corrections. This is encoded-feature fusion; no token-level early-fusion
result is claimed. The existing `~Pool~Add` design remains the default and is
the architecture of the completed exact 8B result. See the
[implementation and comparison requirements](updates/2026-11-26-joint-decoder.md).

## November 27 shared-query price Transformer

- `SPX-PX6-GRID5-v1`: the same 204,694 train / 40,548 validation rolling
  windows as `SPX-PX-GRID5-v1`, with six supervised cumulative returns at
  +5/+10/+15/+20/+25/+30 minutes. A new ID distinguishes the changed target
  task. There is no constituent-stock panel and no test cohort is constructed.
- `none+Transformer128x4~none~none`: linear OHLC embedding, four pre-LN
  Transformer layers (width 128, four heads, FFN 512), one shared learnable
  prediction vector expanded over six temporally encoded future positions,
  and one shared linear output projection. The run is
  `shared-token-transformer-v1`; [protocol](../experiments/shared-token-transformer-v1/README.md).

The depth comparison adds `none+Transformer128x8~none~none` and
`none+Transformer128x12~none~none`, with the same width, query/readout, cohort,
six targets, full batch and update budget. `transformer-depth-sweep-v1` reuses
the completed depth-4 fit and runs depths 8/12 sequentially on one H200.
[Fixed comparison](../experiments/transformer-depth-sweep-v1/README.md).

`transformer-depth-sweep-v2` supersedes the canceled sequential attempt: depths8/12 run concurrently as two processes sharing one H200, with all scientific settings retained. [Concurrent protocol](../experiments/transformer-depth-sweep-v2/README.md).
