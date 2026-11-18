# Experiment register and reporting rule

This is the index for experiments the team intends to cite in the paper. Keep
planned runs, queued runs, completed runs, and measured findings distinct.
Large data, predictions, and checkpoints stay on Delta; link their shared paths
and commit only small run records or summaries here.

## Before a run

Record the question the run answers, the comparison run, the one intended
change, and what stays fixed: manifest and timing policy, event population,
price and article snapshot hashes, code version, model/backbones, training
budget, seed, metrics, and evaluation controls. Publish changed data as a new
version, preserving the prior copy and its audit. Save the exact launch command and a fresh output path.
Run the same evaluation population and metric definitions for both sides.

One-change comparisons support claims about that change. A bundled redesign is
also useful, but label it as a package comparison and do not credit its gain to
one component. A data screen is a validity decision; do not present an
unscreened run with suspect article dates as a clean benchmark merely to make
an ablation. If the population changes, state that the scores are not paired.

## After a run

Add a result note or update the linked handoff with: status, owner, code commit,
manifest, price, and article snapshot hashes, training seed and resolved settings, job ID, exact
command, checkpoint, output path, paired evaluation IDs, metrics with counts and
uncertainty, observed finding, interpretation, limitations, and next action.
Use `PENDING` for unknown fields. A queued job has no result. A smoke run checks
execution, not model quality. Do not copy a result from one checkpoint or data
population into another run's row.

## Current comparison index

| Run or comparison | Purpose | Intended change | Status and evidence |
|---|---|---|---|
| Screened C1 price baseline | Establish the price-task reference and matched text controls | Uses the November 13 screened C1 manifest | Completed on H200 with full paired controls on 2,236 test events. See [the verified result pack](results/screened-price-v1/summary.md); real and shuffled article MSE are nearly identical for this checkpoint. |
| Screened C1 return redesign | Test Adaeze's return design as a package on the same screened population | Return target, Huber/direction task loss, gate target, and last-value shortcut change together | [PR #8](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/8) merged the implementation to `main` at `f0ac8ef`. The [runtime smoke and metrics replay](analysis/text_value/ADAEZE_RETURN_SPEED_SMOKE.md) validate execution on 16 training and eight validation/test events only. Full frozen-input training remains pending; do not attribute a future package difference to one component. |
| Return component ablations R0→R3 | Identify which return-design choices matter | Add direction penalty, then full-task gate target, then last-value shortcut, one at a time | [Design handoff](docs/updates/2026-11-13-return-ablations.md) and [Adaeze execution runbook](docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md) use the verified frozen snapshot and audit. Adaeze has submitted the campaign; collect her verified completed result pack before adding model findings. Adaeze's older jobs `24050724` and `24051590` were last verified user-held with zero elapsed time, use the pre-merge protocol, and are not R0–R3 results. |

The original unscreened C1 manifest and historical outputs remain preserved for
audit context. The screened subset excludes 883 suspect articles; publication
times for retained articles are still unverified.

## Entry template for future work

Copy this block into a focused result note linked from the index:

```text
Question / hypothesis:
Comparison run and exact one change (or explicitly named package):
Held fixed (manifest/hash, population, code, model, budget, seed, metrics):
Owner and status (planned / queued / running / completed / failed):
Command, code commit, environment, job ID, checkpoint, output path:
Validation selection rule and test evaluation population:
Metrics, eligible counts, paired uncertainty, controls:
Observed finding (include negative or inconclusive results):
Interpretation, limitations, and next action:
```

## Final-delivery assets

Use [results/README.md](results/README.md) for linked metric sources, result-pack
templates and figures; use [report/final](report/final/README.md) for versioned
final writing. The [delivery workflow](docs/FINAL_DELIVERY_WORKFLOW.md) explains
owner handoffs and how these feed the paper and presentation.
