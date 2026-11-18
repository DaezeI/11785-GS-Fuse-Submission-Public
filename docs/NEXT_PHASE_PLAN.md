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

- Correct and shuffled articles had nearly identical MSE on two screened
  price checkpoints. The [seed-0 last-price comparison](../results/screened-price-v1/persistence.md)
  also found lower error at all three horizons with the flat reference.
- The [two-seed trained price-only comparison](../results/screened-price-only-v1/summary.md)
  found that the news-trained package beat the independent price-only
  forecasters and that persistence beat both. The correct-vs-shuffled
  comparison tests article matching separately.
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
resource limits. Verify current ownership, execution state, and matching inputs before starting a follow-up.

## Research home and execution

The [research home](https://gs-fuse-research.twiggydonkey4.chatgpt.site) reads the
versioned [research registry](../research_home/records.json) from GitHub main.
Use [the continuation guide](RESEARCH_HOME.md) to run a bounded comparison,
collect verified evidence, and update the same page from a new record.
Nicole authorizes reasonable follow-up experiments within the project scope.
Preserve frozen inputs, keep teammates' existing campaigns intact, and use
scheduler dependencies and scripts for execution and collection.
