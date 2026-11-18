# Final results: start here

The experiment register in [EXPERIMENTS.md](../EXPERIMENTS.md) is the canonical
index. This folder holds the evidence, figures and tables linked from that index.
Keep code, small result records and writing in Git; keep large artifacts in the
team's agreed storage. Overleaf is for final polishing and submission exports.

## Current evidence

See the [research decision brief](../docs/RESEARCH_DECISION_BRIEF.md) for
comparison validity and the proposed next batch.

- [Screened price baseline](screened-price-v1/summary.md): full training and paired
  evaluations completed; seed 0; 2,236 test examples. First verified result pack.
- [Seed-1 screened price repeat](screened-price-seed1-v1/summary.md): all three forecast lengths again show no clear advantage for correctly matched articles over shuffled articles.
- [Pre-event volatility analysis](price-volatility-v1/summary.md): saved subgroup findings plus CPU block-length sensitivity; no clear matched-text benefit in any volatility group.
- Adaeze owns the separate R0–R3 return campaign. Await her completed result pack;
  submission or a smoke test is not a full result.
- Jessica owns Q-Former; Joye owns MoE. Historical runs remain historical.
  New architecture comparisons need the same R3 reference, data, seed, budget
  and hardware/environment. This price checkpoint is not that R3 reference.

## Every result pack

Use [TEMPLATE.md](TEMPLATE.md). Include the question, intended change, comparison,
fixed settings, owner, actual state, code/data/checkpoint identities, numeric
metrics and counts, uncertainty, controls, finding, limits, and artifact custodian.
Save negative and failed runs too. Link every pack from EXPERIMENTS.md.

Use fresh run versions. Corrections must explain what changed; never silently
replace a result or mix metrics from different units, checkpoints or populations.
Do not commit raw articles, model weights, credentials or private access records.

See [the delivery workflow](../docs/FINAL_DELIVERY_WORKFLOW.md) for owners,
GPU handoffs, writing, backups and suggested dates. The old midterm files remain
unchanged under report/overleaf; new writing lives under report/final.
