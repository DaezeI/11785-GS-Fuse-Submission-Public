# November 23, 2026: Fed/SPX matched cohort, test text check and frozen price control

Owner: Adaeze Ilodianya (Person 3, price baselines). Branch `adaeze/fed-spx-matched-control`,
a draft PR stacked on PR #47. No new model was trained (the control reuses existing
checkpoints; DLinear and ridge are small refits), and **no test price, return or score was read**.

## In one paragraph

The Fed/SPX question is whether official Fed release text improves a 20- or 30-minute S&P 500
forecast beyond what pre-release prices give. To compare fairly, the price-only and text models
must be scored on the same releases, and the text model may only see wording that was public at
the release time. This update fixes that shared set of releases for all three splits, freezes the
price-only model Nicole's text model will be compared against, and scores the baselines. Result:
with prices alone, nothing beats "the price stays put" on validation.

## What changed

1. **Matched release set rebuilt.** Two re-checks were folded in: Board releases (archive copies)
   and the 110 testimony rows that had never been checked. Text-verified releases are now
   **160 train / 123 validation**. [Cohort](../../results/fed-spx-news-cohort-v1/README.md)
2. **Test text checked, without reading test outcomes.** Test releases were listed from bar
   timestamps only, then their documents were version-checked with the same rules as train and
   validation. **121 of 216 usable test releases** have verified text.
   [Test text check](../../results/fed-spx-test-text-v1/README.md)
3. **Price-only control frozen and scored on the 123 validation releases**, with persistence,
   DLinear and ridge. [Result](../../results/fed-spx-matched-control-v1/summary.md)

| Forecast (validation, 123 releases) | MAE 20 min | MAE 30 min |
|---|---:|---:|
| Persistence (zero return) | 17.71 bp | 22.06 bp |
| **Price CNN control** | 17.62 bp | 22.04 bp |
| DLinear | 18.34 bp | 22.74 bp |
| Ridge | 21.03 bp | 26.26 bp |

The CNN's gain over persistence (+0.08 / −0.00 bp) is within noise. DLinear and ridge are worse.

## Sample sizes (one row = one Fed release with its price window)

| | Train | Validation | Test |
|---|---:|---:|---:|
| Usable price windows | 978 | 207 | 216 |
| **Text verified (all evidence classes)** | **160** | **123** | **121** |
| Distinct release times | 156 | 113 | 104 |
| Strictest evidence only | 90 | 27 | 26 |

The sample is small. Validation intervals are about ±0.3 bp, so a text effect needs to be roughly
half a basis point or larger to register.

## Decisions made (Adaeze)

- **The control's price model is trained on all 978 train releases**, not only the 160 with text.
  It uses prices only, so text eligibility does not apply to it. Both arms share this frozen
  model, so the comparison stays matched. This updates the
  [short-horizon proposal](../experiments/FED_SPX_SHORT_HORIZON_PROPOSAL.md), which said to fit
  on the shared rows.
- **Testimony uses the speech document rule** ("For release on delivery" line, unmodified PDF, or
  a pre-cutoff archive copy) on all splits, not only House committee PDFs.
- **Test is scored once, after both arms are frozen**, on the 121 test releases. The evidence level
  (all classes or a stricter one) must be chosen before then and used on all splits.
- **Planned, not built:** if U.S. macro releases are added later, 8:30 a.m. releases would use a
  9:35 decision time (first bar after the open) with targets at 9:55 and 10:05, in a new script
  that leaves the alignment v1 code unchanged.

## What each person needs to do

- **Nicole (Person 4, text model):** build the text arm on the frozen control: keep the price
  CNN frozen, add a text component trained on the 160 text-verified train releases, and score on
  the same 123 validation releases. With text switched off it must reproduce the control exactly.
  Still coming from Adaeze: the verified text files for the eligible releases, and the paired
  forecast file plus scorer packaged for the text arm. Nothing to start on test.
- **Joye (Person 2, alignment):** no action. FYI on the planned pre-market rule above; alignment
  v1 is not modified.
- **Jessica (Person 1, catalog):** unchanged from the text-evidence update: flags and a
  verified-body mode for the releases whose wording changed after release.

## Asking an agent about this

Point it at this file, then the three linked READMEs. Exact numbers are in each result's
`metrics.json` or `counts.json`; per-release decisions are in the ledgers next to them.
Raw downloaded pages, checkpoints and per-release forecasts stay in Adaeze's local
`downloads/` and are not in Git.
