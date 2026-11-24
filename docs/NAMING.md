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

`Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add`: Nicole's exact #55 architecture, implemented in `scripts/run_fed_spx_text.py`, using the frozen `none+CNN~none~none` checkpoints. Prepared, not trained. See the registered pilot for token budget, preprocessing and evaluation controls.
