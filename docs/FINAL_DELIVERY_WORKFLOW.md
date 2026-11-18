# Final paper and presentation workflow

GitHub holds the versioned code, experiment register, small result packs and
writing. Drive holds team handoff material, presentation exports and large
artifact backups. Overleaf is used for final formatting, compiling and polishing;
bring edits back into Git before recording a submission version.

The new Drive area is **Final Deliverables — Team 23 — December 2026** inside
the existing team folder. Older files are historical context and are preserved.
Keep the private operator's access information outside the public repository.

## What is ready

The frozen screened C1 dataset is prepared and audited. Nicole's full price
baseline and its paired controls completed. See [results](../results/README.md)
for the first measured finding. Routine data preparation does not need repeating.
Unknown retained publication times remain a scientific limitation.

## Owners and experiment handoffs

- Nicole: data provenance, paired text analysis, result consolidation; runs
  teammates' committed experiments using her privately managed GPU access.
- Adaeze: return objectives and R0–R3 component campaign.
- Jessica: Q-Former comparisons and query/attention diagnostics.
- Joye: MoE comparisons and routing/expert diagnostics.

Other teammates do not need Nicole's cluster login. Send a GitHub branch/commit
with a runnable config/script, a question, the intended single change, matching
reference, frozen data identity, seed, precision, budget, resource needs and
expected outputs. Nicole records execution in a fresh run version and returns
a result pack. Do not send mutable settings only through WhatsApp.

ABCI architecture experiments require an R3 reference under the same environment
and hardware. The completed price model is not a matching R3 reference. Existing
Delta campaigns remain separately owned; coordinate before duplicating them.

## One record per experiment

Before training: add its purpose, intended change and fixed settings to
[EXPERIMENTS.md](../EXPERIMENTS.md). After execution: link a result pack using
[the template](../results/TEMPLATE.md). Include failed and negative outcomes.
Save exact numeric metrics, units and counts, uncertainty, code/data/checkpoint
hashes, control definitions, limitations, and the artifact custodian.

Do not compare historical-split and screened-C1 scores directly, or mix
standardized-price MSE with basis-point return MSE. R0→R1→R2→R3 are component
comparisons; the price-to-return redesign is a package comparison. Inference
article-free controls are not independently trained price-only models.

## From results to writing

1. Link verified results in the register. Tables and figures should be generated
   from their saved machine-readable metrics, not typed independently into slides.
2. Write new paper text under `report/final`; preserve `report/overleaf` as midterm
   history. Link each claim to a result pack or a source citation.
3. Build the talk from `report/final/presentation-outline.md` and the same figures.
4. Before final polishing, check the course's exact dates, format and author list.
   Nicole reported a presentation in the first week of December; exact dates
   here are not independently confirmed.
5. Tag/export an agreed Git version for Overleaf polishing. Bring substantive
   edits back to Git and archive the submitted PDF/deck plus its commit identity.

Suggested internal targets, subject to team agreement: November 23 for a first
results inventory; November 27 for the main comparison table and paper outline;
November 30 for a rehearsable presentation draft. These are planning targets,
not verified course deadlines or new assignments accepted by teammates.

## Artifact preservation

The current completed checkpoint and predictions remain in Nicole's execution
storage. Small metrics/provenance have a local evidence copy. A second full,
team-accessible artifact backup is still pending. Record an archive manifest,
checksums and access verification when that backup completes; do not call
artifacts backed up merely because a Drive folder exists. Private account/key
material is never part of a backup or public result pack.
