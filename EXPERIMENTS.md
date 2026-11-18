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
| Screened C1 price baseline | Establish the price-task reference and matched text controls | Uses the November 13 screened C1 manifest | Training and paired jobs queued; see [project status](PROJECT_STATUS.md). No full result yet. |
| Screened C1 return redesign | Test Adaeze's return design as a package on the same screened population | Return target, Huber/direction task loss, gate target, and last-value shortcut change together | Merged to `main`. Smoke job `24055019` completed `0:0` (16 events, 3 stages, verified checkpoint/config). Full training job `24055023` (R3) submitted and queued on Delta (`gpuA100x4`, pending with estimated start `2026-11-18 16:36:05`). Paired control jobs `24055030`–`24055031` pending on dependency. No full result yet; see [launch update](docs/updates/2026-11-17-adaeze-r0-r3-launch.md). |
| Return component ablations R0→R3 | Identify which return-design choices matter | Add direction penalty, then full-task gate target, then last-value shortcut, one at a time | Merged to `main`. Full training for R0 (`24055020`), R1 (`24055021`), and R2 (`24055022`) completed `0:0` with verified 1.8 GB checkpoints on 2,302 validation and 2,236 test events. Test return MAEs are 80.92 bp (R0), 81.08 bp (R1), and 81.08 bp (R2); test DHRs are 50.5%–52.9%. Differences between R0, R1, and R2 are minimal ($\Delta\text{MAE} < 0.16\text{ bp}$, R1 vs R2 difference $< 0.003\text{ bp}$). R3 full training (`24055023`) and paired control jobs (`24055024`–`24055031`) are queued on Delta (`gpuA100x4`). See [R0–R2 results handoff](docs/updates/2026-11-18-adaeze-r0-r2-results.md). |

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
