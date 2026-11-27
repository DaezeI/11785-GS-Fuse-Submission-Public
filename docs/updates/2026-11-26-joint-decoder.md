# Shared nonlinear price/text decoder

Implementation only, November 26. No new text training, forecasting result or held-out scoring is recorded by this change.

## Motivation and design

The completed `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add` has separate nonlinear heads, then sums their two-output predictions. Its function is `f_price(X) + g_text(T)`. The text correction is independent of the supplied price context even though each branch is nonlinear. This statement applies to that additive architecture, not the original GS-Fuse cross-attention/decoder.

The user requested a joint MLP as a bounded alternative to a larger early-fusion redesign. The opt-in implementation instead computes:

```
h_price = concat(last CNN state, mean CNN state)       # 64 dimensions
h_text  = GELU(Linear(pooled_text))                    # 32 dimensions
delta   = joint_MLP(concat(h_price, h_text))           # 96 -> 32 -> 2
forecast = frozen_price_forecast + delta
```

`MLP1-Residual` has one GELU hidden layer in `joint_MLP`; `MLP2-Residual` has two. The joint nonlinearity precedes the two-output correction and receives both modalities. It can therefore express a text effect that changes with price context. This is encoded-feature fusion. The implementation does not add token alignment, cross-attention or a claim that a particular fusion timing is optimal.

The final correction layer starts at zero, so the initial forecast equals the loaded price baseline. `freeze_price()` freezes the price encoder and original price head; the text projection and full joint decoder remain trainable under the existing `text_head` optimizer group. Bypassing text with `text=None` returns the original price prediction after head learning. Supplying an all-zero feature vector **does not guarantee** the same result, since the joint decoder can learn a price-dependent correction and has biases.

## Interfaces and checkpoint provenance

`CompactReturn(text_dim=4096, output_dim=2, fusion="joint_mlp", joint_hidden_layers=1)` constructs `MLP1-Residual`. Depth 2 constructs `MLP2-Residual`. Other frozen embedding dimensions are accepted by the module; using a different encoder requires its own architecture ID and experimental record.

`scripts/run_fed_spx_text.py` now accepts `--fusion joint_mlp --joint-hidden-layers 1` (or 2). It records the actual architecture in metrics, checkpoints and prediction NPZ files. The default remains the existing Add architecture. The shared scorer already supports `from-text-run --label`; use the exact new architecture ID rather than its legacy default when converting a joint run.

The existing exact-8B runner still enforces the original #55 frozen CNN checkpoint hashes and input fingerprints. It must not silently accept different rolling-price checkpoints. The model class accepts the rolling-price `state_dict` under the same price-parameter keys; using those weights in a new text comparison requires a separately identified frozen bundle and protocol. This change does not choose that new experiment.

The runner now saves full-training Smooth L1 each epoch, each optimizer-step loss/gradient norm, and the final model/optimizer/order-generator state. Its inherited epoch default is retained for compatibility, **not endorsed as an adequate budget for an unrun joint experiment**.

## Required comparison before a text-value claim

A new joint decoder adds a price-conditioned trainable readout as well as text conditioning. Comparing it only to a frozen price baseline would confound extra readout capacity with text value. A valid comparison needs a separately trained price-only readout using the same joint-head structure with zero text features, matched training rows/budget/selection, plus correctly matched and reassigned text conditions. An inference-only zero-text intervention is not that retrained baseline. Representation quality and the small text-training population remain unresolved by adding this MLP.

## Verification scope

Focused CPU tests check that the text effect can depend on price context, both hidden-depth options update their trainable projection/decoder, the frozen price tensors and text-bypass predictions remain unchanged, legacy Add parameter keys/shapes remain compatible, and the training factory selects the requested decoder. They establish implementation behavior, not forecasting quality. No early-fusion or new joint-decoder GPU benchmark has run.

The 11 focused tests passed. A separate CPU check loaded the selected GRID15 rolling-price checkpoint and used eight real validation price histories: both joint depths exactly reproduced the loaded price forecast at their zero-correction starting state. The one-/two-hidden-layer trainable heads contain 134,274/135,330 parameters with 4,096-dimensional text features. This is a checkpoint-wiring check, not a trained text result. The cached 171-release text population is not enlarged by this implementation.
