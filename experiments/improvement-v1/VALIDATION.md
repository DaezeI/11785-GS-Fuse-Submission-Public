# Validation and limits

Prepared November 18, 2026 against source base `dd03d59`.

Passed locally on Windows, Python 3.12.14, CPU PyTorch 2.15.0 and
transformers 4.57.1 (these are test-environment versions, not a replacement
for Nicole's verified GPU environment):

```bash
python -m pytest tests/test_improvement_campaign.py tests/test_lora_execution.py \
  tests/test_lora_text_encoder.py tests/test_checkpoint_format.py \
  tests/test_variant_unfreeze.py tests/test_return_analysis.py \
  tests/test_return_evaluator.py tests/test_c1_protocol.py tests/test_text_mask.py \
  tests/test_return_ablation.py tests/test_forecast_task.py \
  tests/test_return_metrics_replay.py tests/test_text_cache.py \
  tests/test_research_home_atomic.py tests/test_audit_first_training.py \
  tests/test_compact_return.py -q
python scripts/check_text_value.py --require-torch
python scripts/research_home.py validate
git diff --check
```

- 60 affected tests pass, including actual tiny random-weight Llama attention
  with projection/adapter gradients and frozen-base isolation; no model download.
- 25 required text-value checks pass without skipped tensor checks.
- The registry validates 40 records, preserving 11 published records and 29 **planned** designs.
- Changed Python modules compile. Render/submit tests verify Unix paths,
  directive placement, declared walltime, locked test selection, two-lane
  dependencies, tamper detection and duplicate/partial-submission receipts.
- All 17 existing-pipeline matrix configurations are accepted by the real `train.py` parser.
- Windows registry writes now close their temporary file before replacement;
  this was needed to register the planned configurations on this workstation.

No real dataset, pretrained backbones, remote scheduler or CUDA-enabled test runtime was
available in this checkout. Read-only nvidia-smi detected an RTX4060 Laptop with
8188 MiB VRAM; installed test PyTorch is CPU-only. **No real-input campaign audit, full-backbone GPU
smoke, submission, real-data training run, GPU speed measurement or performance improvement
has been verified here.** In particular, expanded-history windows may require
a newly audited common cohort; the current worker refuses population changes
or overlap. Nicole must review her actual resource profile and run smokes.

The full designs distinguish 17 existing-pipeline configurations and a compact price-only prototype from 11 designs
requiring new implementation/evidence. All remain planned in the registry.
Existing result records are preserved. Evidence URLs use main-branch paths as
required by the reporting skill and become public Research Home evidence only
after the focused PR merges; the branch files are reviewable before merge.

Revision checks also cover atomic checkpoint recovery after a simulated quota
failure; validation-best stage handoff; strict text-only warmup across both
price routes; pre-attention contrastive anchors; resource-error classification;
smoke-only rendering and evidence-bound full training.

The [reproducible synthetic CPU check](compact-synthetic-cpu.json) used a
16,931-parameter model with a 72,761-byte state. It is an engineering exercise
on random tensors, not a real-data run or accuracy result. Reproduce with
`python scripts/improvement/benchmark_compact.py --device cpu --out FRESH.json`.
No CUDA speed, T4 deployment or model non-inferiority has been measured.
