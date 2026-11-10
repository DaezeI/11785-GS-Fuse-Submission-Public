# Project status

Updated November 10, 2026. Maintained by Nicole for team coordination.

Read this page for current progress and handoffs. Read
[TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) for the shared experiment protocol and
ownership. Dated updates preserve the history of what changed.

## Current position

**C1 is the agreed main split.** The latest inspected code still needs to connect
that manifest to every training stage and to evaluation. The immediate research
goal is to run the corrected article-free comparison on an agreed full-model
checkpoint, then produce comparable C1 results.

| Work | Status and evidence | Next handoff |
|---|---|---|
| C1 selection and window audit | C1 agreed. Independent bar-overlap checks passed under two timing assumptions. Actual publication times still need verification. | Nicole implements/tests shared C1 loading. |
| Nicole's initial analysis and controls | Merged into `main` in [PR 1](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/1). CPU analysis and 25 focused tests documented. | Evaluate the new controls with complete inputs and a verified checkpoint. |
| Q-Former/shared text padding | Jessica's correction is included in `main` at [the November 9 merge](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/commit/96cc6d7900dedd8b867b39a951045fe3804aa778). Full-model performance effects remain to be measured. | Jessica measures padding attention, query similarity, and matched forecast changes. |
| Delta accounts | Team chat reports account details delivered. Jessica reports successful SSH access. Nicole reports receiving her username and entering the identity portal. | Nicole verifies her Delta shell access and shared-file access. |
| Delta environment and inputs | Joye reports an A100/PyTorch/CUDA test and uploaded data/models. Setup and job scripts are on `feat/moe-qformer-wandb-cache`. Complete GS-Fuse execution and current file inventory still need verification. | Joye records a working small job, exact inputs/checkpoint, and outputs. |
| Reference checkpoint | The feature-branch guide says original midterm trained weights were never copied off the old host and describes recreating them. Later chat reports uploaded models without identifying a matching trained checkpoint. | Inspect shared files and training logs to identify a completed usable checkpoint. |
| Delta jobs | Adaeze reports pending jobs. Joye's November 6 commit records an earlier cache-job timeout. Current queue and completion state have not been inspected in this update. | Job owners inspect queue reasons and accounting/logs. |
| Original gate / MoE | Ownership remains Adaeze / Joye. Latest reviewed changes contain setup work and padding fixes. Further diagnostics may exist locally or on Delta. | Owners publish scripts/results with run identity when available. |

## Next work, in dependency order

1. **Establish one runnable checkpoint diagnostic.** Verify shared articles,
   prices, backbones, trained weights, model configuration, and environment.
   Record the exact code commit and training history. Run a small evaluation
   before expanding the sample size.
2. **Complete C1 loading in parallel.** Nicole connects training and evaluation
   to the committed manifest, including Stage-1 price windows, training-only
   preprocessing, auxiliary report pools, and emitted-window checks.
3. **Run the initial corrected text control when a checkpoint is available.**
   Keep model and price examples fixed across correct, repeated shuffled, and
   explicit article-free inputs. Save individual predictions and IDs. Label the
   checkpoint's actual training protocol.
4. **Produce the common C1 baseline and matched diagnostics.** Joye coordinates
   compute, Nicole verifies data, and all owners use the same scoring protocol.
   Final C1 claims depend on verified C1 training history.

Steps 1 and 2 proceed in parallel. Gate, expert, and query code inspection can
also proceed while compute is queued. See the
[November 10 update](docs/updates/2026-11-10.md) for recent history, explanations,
and proposed completion criteria.

## Branches and review

- Develop on personal/task branches and open focused PRs. Merge ready changes
  into `main`, following the November 5 meeting guidance supplied by Nicole.
- Current `main` contains Nicole's tools, team instructions, C1 lists, and
  Jessica's padding correction.
- Joye's Delta scripts and cache-job updates remain on
  `feat/moe-qformer-wandb-cache`. That branch also needs current shared changes
  incorporated before it becomes the common corrected execution setup.
- Coordinate changes to the shared Delta checkout and Python environment. Use
  separate working directories and outputs for experiments.

## How to maintain these notes

Nicole coordinates the summary. Each work owner supplies changes and evidence
through a small documentation PR or their experiment PR.

- Update this page when a blocker is resolved, an input becomes available, or a
  result is ready for someone else to use.
- Add a dated note under `docs/updates/` for a meeting or significant project
  update. Preserve earlier notes and link corrections in later entries.
- Record owner, status, next step, dependency, and evidence. Use the run checklist
  in `TEAM_WORKFLOW.md` for commands, versions, W&B links, and output locations.
- Identify the source of a status claim. Code inspection, an executed result,
  a teammate's report, and a proposed action have different evidential scopes.
- Summarize technical decisions and handoffs. Keep personal contact details,
  account secrets, private recordings, and raw chat transcripts in their original
  private locations.

## Update history

- [November 10 — progress since the November 5 TA meeting](docs/updates/2026-11-10.md)
- [November 4 — C1 decision and shared workflow](TEAM_WORKFLOW.md)
