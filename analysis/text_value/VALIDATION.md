# Execution and validation record

## Executed

**25 focused tests passed, with no skips**, including four real CPU PyTorch helper tests.

- `python scripts/summarize_text_controls.py`: recalculated all reported overall/horizon contrasts, benefit-retention decomposition, CSV/JSON, and chart from Adaeze's committed control metrics. The source hash matches the reference experiment file.
- `python scripts/audit_shared_splits.py`: read the actual internally shared SP500.csv, identified 162 exact duplicate rows, created a separate sorted unique copy, checked every listed 35+140-bar interval under both time assumptions, and fitted C1 CPU price-only references with training-only scaling and validation-only selection.
- `python scripts/check_text_value.py --require-torch`: focused unit checks, including real CPU PyTorch tests of article independence, correct dummy-token mask, price dependence, and the separate fusion-bypass mode. Also checks arithmetic, shuffles, input layout/provenance, overlapping-window detection, conflicting duplicates, and bootstrap edge cases.
- Reproduction of the earlier CPU audit from a fresh extracted package passed. The public aggregate path is also tested from a fresh tree of the files being published.

The aggregate analysis and price references ran in a Python 3.9.6 / NumPy 2.0.2 CPU environment on the local computer. No-text helper tests ran with the repository's pinned PyTorch 2.3.0 installed in a separate temporary Python environment. No Colab GPU time was consumed.

## Not executed

The new `run_text_controls.py` has not run against the full Llama/MOMENT checkpoint. Its shuffle/array contract is tested and the no-text helper is CPU-tested; full-backbone runtime, numerical replication of Adaeze's original zero-text condition, and corrected C1 multimodal retraining are not claimed. The focused suite is not the repository's full backbone-dependent test suite.

## Reproduce the focused checks

With Python, NumPy, and Matplotlib:

```sh
python scripts/check_text_value.py
python scripts/summarize_text_controls.py
```

Without PyTorch, the four no-text helper tests are explicitly skipped. In the team's PyTorch environment:

```sh
python scripts/check_text_value.py --require-torch
```

The private price CSV is needed only for the separate price audit. Full checkpoint inference additionally requires the original report files, matching checkpoint/backbones, the team's model dependencies, and CUDA. These resources are not needed to reproduce the aggregate results already checked in.
