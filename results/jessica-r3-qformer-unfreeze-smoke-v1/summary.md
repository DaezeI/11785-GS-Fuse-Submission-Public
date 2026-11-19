# Q-Former Stage 3 smoke

Jessica owned the experiment. Nicole ran it on one H200 on her personal GPU
resource with a 30-minute wall-time cap.

The smoke asks whether Q-Former weights change in Stage 3 after PR #28. It uses
commit `7e510a35e9eb5c3740dc975384a557a37decfdc9` and the frozen
`c1-screened-return-v1` snapshot with content SHA-256
`049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`.

The return recipe uses `category4_1400_strict_before`, the raw time-series
residual, learnable last value, direction weight 1, and task-based gate target.
The run uses Q-Former with 16 queries and 2 layers, a softmax gate,
MOMENT-1-large and Llama-3.2-1B, the `text_inst_llama1b_tf32.pt` cache,
TF32 without 4-bit, and seed 0. The batch size
is 2. Each stage has one epoch. The smoke uses 16 training events, 8 events
per evaluation split, and 32/16 sliding windows for training/evaluation.

The CPU check compares the Stage 2 and Stage 3 checkpoints. A pass requires
`cond_proj.weight` to move from zero to a finite positive norm and requires
`layers.0.ln_self.weight` to move away from its all-ones initialization.
The small subset supports an execution check. A forecasting comparison would
require a matched full cross-attention R3 reference and held-out evaluation.

## Verified result

The job completed in 2 minutes 43 seconds. The frozen-input hashes matched,
all three stages completed, and the selected Stage 3 checkpoint reloaded.
The Stage 2 projection norm was zero and the Stage 3 norm was
0.02358576. Stage 2 LayerNorm weights were all one. The largest
absolute Stage 3 change was 4.011e-05. The CPU check passed.

[Weight check](weight-check.json) · [Checkpoint checker](check-checkpoints.py) · [Provenance](provenance.json)
