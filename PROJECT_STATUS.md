# Project status

Updated November 12, 2026, after checking the submitted C1 jobs and article dates on Delta.
This page is the current team overview. [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md)
has the owner-by-owner handoff and experiment checklist; dated updates preserve
earlier status without rewriting it.

## Where we stand

**C1 is the agreed main split, and its training/evaluation code is in `main`.**
Nicole's [PR #4](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/4)
connected the manifest to all three training stages and paired text controls.
The real-input audit emitted 6,167 train, 2,500 validation, and 2,921 test
events with no exclusions or shared input-plus-target bars across partitions.
Five new C1 regression tests and 25 existing focused text-control tests passed
in the tested Delta environment. See the [protocol and exact audit](analysis/text_value/C1_PROTOCOL.md).

Nicole completed a 16-training-event, three-stage GPU smoke run and paired
correct/shuffled/two article-free controls on eight test events. Those
[derived results](analysis/text_value/C1_SMOKE_RESULTS.md) validate execution
only. They do **not** answer whether correctly matched articles improve a
properly trained model.

| Work | Current evidence | Next step |
|---|---|---|
| C1 implementation | Merged in PR #4; loader audit and focused tests passed. | Use `main` for new work and keep explicit timing/data provenance. |
| Full C1 text comparison | [Sensitivity run plan](analysis/text_value/C1_FULL_SENSITIVITY_PLAN.md) records training job `23812725` and two `afterok` full-test paired jobs `23812735` and `23812734`. On November 12 the still-pending training job was placed on a reversible user hold while article dates are audited; its dependent jobs remain pending. The separate 256-event smoke-checkpoint diagnostic `23812736` completed with exit 0. | Complete the [article-date audit](analysis/text_value/ARTICLE_DATE_AUDIT.md), version any corrected C1 inputs, rerun window checks, then release or replace the held job. A larger evaluation on the 16-event checkpoint is not the next scientific test. |
| Publication-time validity | The manifest has 12 categories but no per-article release timestamps. Five C1 test summaries labeled December 16, 2023 contain Dec 19–21 stories (four Reuters datelines and one matched publisher page). The price CSV timezone and bar-label convention remain unverified. The new run explicitly uses `category4_1400_strict_before` as a sensitivity assumption. | Complete the [article-date audit](analysis/text_value/ARTICLE_DATE_AUDIT.md), document verified corrections, and agree on an availability policy before calling any full result publication-time-safe. |
| Jessica's padding fix | Merged to `main` in [commit `96cc6d7`](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/commit/96cc6d7900dedd8b867b39a951045fe3804aa778) with three CPU tests. | Ask for links to any Q-Former analysis missing from GitHub before assigning further work. |
| Gate, MoE, Q-Former diagnostics | Adaeze, Joye, and Jessica retain their existing streams. The inspected November 11 team job scripts used the earlier split; their outputs are historical diagnostics, not the C1 baseline. | Keep each checkpoint/split identity explicit; use the [owner handoff](TEAM_WORKFLOW.md#9-immediate-next-actions-agent-handoff-for-wednesday-november-11). |

The full C1 training and paired jobs are queued work, **not completed results**.
Slurm jobs continue independently of a laptop or SSH session. The shared
account balance and queue can change; query them before planning more compute.
Large checkpoints, raw articles/prices, and full prediction NPZ files stay on
Delta. GitHub holds code, commands, hashes, and shareable derived records.

## Alternative GPU resource

On November 12, Nicole submitted an ACCESS exchange request for **100 SDSC
Expanse GPU-hours and 200 GB of Expanse Projects Storage** (5,585 ACCESS
credits). The portal showed the request **Under Review**. It is a single pool
for the team project, not a per-person allocation; the existing 972 remaining
Delta GPU-hours were left unchanged. The [course migration note](https://piazza.com/class/mo4ko44arnc3sg/post/659)
allows other ACCESS resources, and the [ACCESS exchange guide](https://allocations.access-ci.org/how-to)
explains provider approval.

Do not treat the request as Expanse access yet. After approval, check each
teammate's Expanse access in ACCESS Users + Roles and complete any account
provisioning. Transfer only authorized inputs and model files through Globus,
verify hashes and permissions, then run a short Expanse Slurm memory/environment
check before submitting full C1 training and dependent paired controls. Standard
Expanse GPUs are 32 GB V100s, so the A100-tested configuration may need an
adjustment. Keep the Delta jobs queued until an alternative run is ready.

## Next team actions

Check the Delta queue and Expanse exchange status, choose an owner and policy
for article/price timing, and have each diagnostic owner link an artifact or
record a concrete blocker. Adaeze can debug gate instrumentation on Nicole's
16-event checkpoint, but matched C1 gate claims need a completed full C1
checkpoint. Joye's historical jobs should not be relabelled as C1 merely
because C1 code later merged to `main`.

## History and maintenance

- [November 10 progress snapshot](docs/updates/2026-11-10.md) records what was
  known before Delta access and C1 integration were verified. It is historical.
- [November 10 C1 implementation handoff](docs/updates/2026-11-10-c1-implementation.md)
  records the audit and smoke work completed that day.
- [November 4 workflow](TEAM_WORKFLOW.md) records the C1 decision, ownership,
  and detailed reproducibility requirements, with its current status updated.

Nicole coordinates this summary; each owner supplies evidence through their
own PR or result note. Update this page when a queued run completes, the
timing policy changes, or a new checkpoint becomes the team's reference.
