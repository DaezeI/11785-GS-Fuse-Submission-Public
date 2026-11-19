# Compact GS-Fuse: locked whole-path test result

**Keep the compact model as the forward baseline. Use price-only as the reference; keep summary fusion experimental.** On the locked screened C1 test set, adding the frozen-price summary residual did not reproduce its small validation gain. No training or checkpoint reselection occurred during test evaluation.

| Seed | Price only (bp) | Price + summary (bp) | Improvement (bp) |
| --- | ---: | ---: | ---: |
| 0 | 80.109 | 80.309 | −0.201 |
| 1 | 80.669 | 80.669 | 0.000 |
| 2 | 80.107 | 80.046 | +0.062 |
| **Mean** | **80.295** | **80.341** | **−0.046** |

Lower MAE is better. Improvement is price-only minus price + summary, calculated before rounding. The zero-return reference was **80.487 bp**. The mean is across training seeds, not predictions from an ensemble. On [validation](RAW_PATH_VALIDATION.md), the same summary residual reduced mean MAE from 82.626 to 82.403 bp (+0.223 bp). Test effects were mixed: one worse, one tie, one better. No uncertainty interval was computed.

## Scope and provenance

- **Task:** 35 observed bars → 140 five-minute future bars × four OHLC channels. MAE weights all events, bars and channels equally. Units are basis-point log returns relative to the last observed close.
- **Test:** 2,236 screened C1 events on 845 dates; seeds 0, 1 and 2, with the six existing validation-selected checkpoints. CUDA PyTorch 2.5.1+cu124. Test score generation did not train or refit anything.
- **Data:** Canonical snapshot `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`; manifest `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`; test ID digest `0a47dc30790f9271266d54587b7ab0c851dddaa29479f2cda80fb0d6454acb82`; evaluation lock `3ab5d0a7c07268af23803fd129fe5d0520ce51606af2e68188dc65a4269348ce`. Timing policy: `category4_1400_strict_before`. Article publication times remain unverified.
- **Text:** The compact encoder was trainable and used the archive’s `full_summary` files, not original articles. This route did not run Llama or use cached Llama embeddings. The screened archive does not establish event-time availability of the summaries.

[R3’s published matched-text condition scored 80.615 bp](../../results/adaeze-screened-r3-h200-v2/metrics.json) under the matching reported task/data/timing setup, and the zero-return reference agrees to rounding. **This is a contextual reference, not a confirmed improvement over R3.** Its per-event predictions were unavailable to verify exact pairing independently, and this report contains no paired uncertainty estimate. Do not use this test outcome to choose a seed or tune follow-up ablations.

[Exact test summary](metrics/compact-path-test-summary.json) · [Locked evaluation runbook](RAW_PATH_TEST_RUNBOOK.md)
