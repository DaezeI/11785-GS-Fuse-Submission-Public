# What changed in compact GS-Fuse

**The forward model is compact price-only.** The optional summary residual remains an experimental branch, not a required dependency. This is a replacement architecture evaluated on the same return forecasting task, not a compressed R3 checkpoint or a controlled removal of one R3 component.

| | Published R3 H200 | Compact whole-path baseline |
| --- | --- | --- |
| Price encoder | Pretrained MOMENT-1-large; configured raw-price residual and learnable last value | From-scratch 4→32 pointwise projection and four left-padded residual 1D convolutions (kernel 3, dilations 1/2/4/8); concatenate last and mean activations |
| Text encoder | Pretrained Llama-3.2-1B | None by default. Optional train-only vocabulary (up to 16k tokens, max 512 per summary), 64-dim trainable embeddings, masked mean excluding PAD, and a trainable projection |
| Fusion/readout | Cross-attention and Granger-supervised softmax gating | Two-layer price MLP emits all 560 outputs directly. Optional independent text MLP adds an OHLC-path residual to a frozen price model. No cross-attention, Granger gate, or autoregressive decoding |
| Training | Multi-stage setup and a magnitude plus direction objective | Fresh price training, then an optional frozen-price residual run. SmoothL1 over all normalized 140×4 return targets; checkpoint selected by validation whole-path MAE, epoch 0 eligible |

Implementation: [compact price blocks](../../model/compact_return.py), [summary encoder](../../model/compact_raw_text.py), [whole-path heads and training](../../scripts/improvement/train_compact_path.py). Contrast: [R3 run configuration](../../results/adaeze-screened-r3-h200-v2/provenance.json), [historical compact-design audit](../../docs/experiments/AUDIT_FIRST_AND_COMPACT_V2.md).

## What stays fixed and what changes next

- **Task:** 35 past five-minute OHLC bars → 140 future bars × 4 channels; each value is a basis-point log return from the last observed close. Whole-path MAE weights every event, bar and channel equally. Close at bars 35/70/140 is secondary. Canonical screened C1 and the original timing policy remain fixed; article publication times are unverified.
- **No text:** Price-only instantiates no text encoder. In the optional residual model, omitting token IDs or supplying all PAD returns the same frozen price forecast. That is not R3's zeroed-article-token control; zeroing R3 article inputs does not remove its fusion route.
- **Context:** The four convolutions give each activation a nominal 31-bar receptive field. The final activation sees at most the last 31 bars; mean pooling covers all supplied input positions but does not explicitly model their long-range temporal order. The proposed L35/L140 price-context ablation keeps the architecture and forecast origin fixed, pairs both arms on the same re-audited train/validation events, and never scores the protected test set. Wider dilations or a different readout would be separate experiments.
- **Evidence and compute:** The measured implementations had 53,136 parameters price-only and 1,101,792 with the optional residual (1,048,656 trainable). On the RTX 4060, the three validation runs per mode logged peak PyTorch-allocated CUDA memory of about 19.6 and 66.0 MiB respectively. These are not total board use or matched R3 throughput measurements. The compact code needs no foundation weights or cached embeddings; [the local runbook](RAW_PATH_RUNBOOK.md) gives its reference setup. A portable environment must still stage and verify the same data.
- **Interpretation:** [Locked test](RAW_PATH_TEST_RESULTS.md) did not confirm a summary benefit or a win over R3. The optional text path uses archived summaries, not original articles. Follow-up choices are validation-only; no retuning on the locked test results.

For the teammate setup, see [the Windows/Colab/Linux runbook](COMPACT_PORTABLE_RUNBOOK.md).
