---
name: gsfuse-naming
description: Use whenever a GS-Fuse answer, note, PR, result, code comment or handoff refers to a model, forecaster, baseline, control, gate, data set, cohort, split or comparison. Enforces the IDs in docs/NAMING.md instead of phrases like "the model" or "the data".
---

# Name GS-Fuse models, data and comparisons by ID

1. Read `docs/NAMING.md` on the current branch before writing.
2. Name every forecaster by its architecture ID (`<TextEncoder>+<PriceEncoder>~<Reader>~<Fusion>`),
   every data set by its cohort ID (`<SOURCE>-<CONTENT>[-<EVIDENCE>]-<CLOCK>-v<N>`), and every
   number by its comparison label `<ArchitectureID> @ train:<CohortID> → score:<CohortID> <split>`.
3. Never write "the model", "our model", "the gate", "the data", "the matched set" or "the
   control" without the ID. Name a gate by its fusion value (`Gate-Learned`, `Gate-Granger`,
   `Gate-Const0.5`, `MoEGate`).
4. Use "control" only for the same trained architecture with an input or component neutralized
   (`[text=zero]`, `[text=shuffled-<seed>]`, `[gate=<g>]`). Independent reference forecasts are
   baselines.
5. If a design or cohort has no ID yet, propose one in the same format, add it to
   `docs/NAMING.md` in the same change, and say so. Never reuse an ID for a changed definition;
   bump its version.
6. When reading older notes that predate the rule, translate their wording into IDs in your answer
   and point out any ambiguity instead of guessing.
