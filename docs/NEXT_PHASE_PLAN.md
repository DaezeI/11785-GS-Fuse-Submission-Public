# Current phase: controlled experiments, analysis and final writing

The frozen screened dataset is ready, and Nicole's price reference and full
paired evaluations have completed. Routine data preparation is finished.
The work now is to turn existing runs into defensible findings, complete the
missing controlled comparisons, and write the paper and presentation.
The final report is not finished merely because the infrastructure is ready.

## Where each thing belongs

| Item | Canonical location | Other copies |
|---|---|---|
| Code, configs, split manifests, dataset fingerprints | GitHub | Execution copies pinned to a commit |
| Experiment purpose, owner, intended change, state | GitHub `EXPERIMENTS.md` | WhatsApp may link to the record |
| Small metrics, provenance, findings, generated tables/figures | GitHub `results/<run-id>/` | Versioned exports/backups in the new Drive area |
| Final paper source and presentation outline | GitHub `report/final/` | Drive exports; Overleaf for final polishing |
| Checkpoints and full predictions | Execution storage plus a verified team-accessible backup | Record checksums and custodian; never put large weights in Git |
| GPU account, provider, reservation, keys and private paths | Nicole's local operator skill/handoff | No public copy |

The new Drive area is **Final Deliverables — Team 23 — December 2026** in the
existing team folder. Leave older files intact as historical context. The
Drive Start Here doc is an entry point, not a second copy of the experiment
register or paper source. Bring substantive Overleaf changes back into Git.

## What to do next, in order

1. **Inventory finished results.** Collect each owner's committed result pack
   and identify which question it answers. Reading their records comes before
   asking them to rerun anything. Adaeze owns R0–R3, Jessica Q-Former, Joye MoE;
   Nicole owns screened-data provenance and paired text analysis.
2. **Preserve the completed outputs.** A backup means a second copy of the
   trained checkpoint and predictions with matching checksums and verified
   access. The existing small ZIP is only a metric/provenance backup; the full
   large-artifact backup is still pending.
3. **Check comparison validity.** List completed, missing and incompatible
   comparisons. Keep historical split results separate. Standardized-price MSE
   cannot be ranked directly against return-basis-point MSE. New architectures
   need the same R3 reference, inputs, budget, seed and hardware/environment.
4. **Analyze before expanding.** Start with the existing per-event predictions:
   aligned metrics, persistence/reference comparisons, date dependence and
   bootstrap block-length sensitivity. Add training seeds or new runs only
   for a stated missing comparison, coordinated with its owner.
5. **Write alongside analysis.** Every paper/slide claim links to a result pack.
   Generate paper and presentation tables from the same saved numbers. Confirm
   exact course dates/template/authors and rehearse before early December.

## Insights that must survive a new chat

- Correct and shuffled articles have nearly identical MSE for the completed
  price checkpoint. This is a measured single-seed result, not proof that
  financial text is generally useless.
- The two article-free conditions are different inference interventions; they
  are not separately trained time-series-only models.
- Price versus return compares a redesign package. R0→R1, R1→R2 and R2→R3
  isolate the specified return components; architecture comparisons are another
  question and need their own matching reference.
- The date screen excludes suspect articles but does not verify retained
  publication times. Preserve that limitation and all earlier dataset versions.
- Other owners' submitted campaigns must not be duplicated, canceled or relabeled
  by accident. Nicole's completed price reference is not an R3 checkpoint.

## Nicole's personal GPU resource

Only Nicole operates this additional resource for the team. A teammate sends
her a committed GitHub config/script plus exact commit, question, reference,
intended change, frozen dataset, seed, precision, budget and expected outputs.
Nicole launches it in a fresh run directory and returns the resulting metrics,
provenance and plain-language finding. Teammates do not need her login.
Keep provider/account/reservation details local; hardware models may be recorded
for scientific reproducibility. The local operator instructions carry the
resource limits. No new run is authorized just by appearing in this plan.

## Optional automation: keep it small

The completed campaign already ran a fixed verification → smoke → training →
evaluation chain that stopped on failure. Scheduler jobs can run without a
chat repeatedly checking them. This is execution automation, not an autonomous
researcher deciding the next hypothesis.

A useful next addition is a lightweight results dashboard generated from
committed result packs: run state and last update, purpose, owner, dataset/task,
metrics with units/counts, uncertainty and links to figures/paper evidence.
Group compatible comparisons; do not create a leaderboard mixing tasks or splits.
A static HTML report refreshed after an evaluation is enough initially.

This dashboard is a proposal, not an implemented service. Do not build a new
database/platform, continuously poll with an LLM, or launch open-ended experiments
before the result inventory is complete. Broad literature/deep research is not
the first step; use targeted research when an actual analysis or writing gap
needs it. The next chat should start with the local handoff, current main and
saved evidence, then propose a bounded run list or small dashboard if useful.
