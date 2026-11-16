# Experiment register and reporting rule

This is the index for experiments the team intends to cite in the paper. Keep
planned runs, queued runs, completed runs, and measured findings distinct.
Large data, predictions, and checkpoints stay on Delta; link their shared paths
and commit only small run records or summaries here.

## Before a run

Record the question the run answers, the comparison run, the one intended
change, and what stays fixed: manifest and timing policy, event population,
price file, code version, model/backbones, training budget, seed, metrics, and
evaluation controls. Save the exact launch command and a fresh output path.
Run the same evaluation population and metric definitions for both sides.

One-change comparisons support claims about that change. A bundled redesign is
also useful, but label it as a package comparison and do not credit its gain to
one component. A data screen is a validity decision; do not present an
unscreened run with suspect article dates as a clean benchmark merely to make
an ablation. If the population changes, state that the scores are not paired.

## After a run

Add a result note or update the linked handoff with: status, owner, code commit,
manifest and price hashes, training seed and resolved settings, job ID, exact
command, checkpoint, output path, paired evaluation IDs, metrics with counts and
uncertainty, observed finding, interpretation, limitations, and next action.
Use `PENDING` for unknown fields. A queued job has no result. A smoke run checks
execution, not model quality. Do not copy a result from one checkpoint or data
population into another run's row.

## Current comparison index

| Run or comparison | Purpose | Intended change | Status and evidence |
|---|---|---|---|
| Screened C1 price baseline | Establish the price-task reference and matched text controls | Uses the November 13 screened C1 manifest | No full result. On November 15, Nicole's three Delta model jobs were verified as pending with zero run time; leave them unchanged pending owner confirmation. Expanse preparation is unexecuted because the November 12 exchange remains under review. See [project status](PROJECT_STATUS.md). |
| Screened C1 return redesign | Test Adaeze's return design as a package on the same screened population | Return target, Huber/direction task loss, gate target, and last-value shortcut change together | Code in [draft return PR](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/8); GPU smoke and full run pending. Do not attribute a future difference to one component. |
| Return component ablations | Identify which return-design choices matter | Change direction penalty, gate target, and last-value shortcut separately while holding screened data and budget fixed | Run setup pending on the return branch; no result yet. |

The original unscreened C1 manifest and historical outputs remain preserved for
audit context. The screened subset excludes 883 suspect articles; publication
times for retained articles are still unverified.

## Execution migration is not an experiment

Moving the same screened-C1 baseline from Delta A100 to Expanse V100 is an
execution change that must be recorded, not silently folded into a result.
Before comparison, record the Expanse project, code commit, environment, GPU,
precision/quantization, cache signature, batch size, seed, manifest and price
hashes, and resolved training budget. A V100 memory-driven configuration change
creates a different run configuration. Do not attribute a metric difference to
the data screen, text, gate, or return task if hardware and precision also
changed. The prepared scripts in [`scripts/expanse`](scripts/expanse/README.md)
are plans only: the exchange is under review and no Expanse job has been run.
The current request is for 100 GPU-hours, but more may be requested; record the
hours actually approved and used rather than treating 100 as a permanent cap.

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
