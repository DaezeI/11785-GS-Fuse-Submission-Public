# Project status

Updated November 16, 2026, after PR #8 merged the return-task implementation,
runtime fixes, frozen-input audit, and R0–R3 protocol to `main`.
The [dataset and run handoff](docs/updates/2026-11-13-return-ablations.md) lists
the Delta path, fingerprint, and commands for the controlled return comparisons.
This page is the current team overview. [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md)
has the owner-by-owner handoff and experiment checklist; dated updates preserve
earlier status without rewriting it.

**For all new team C1 runs, use
[`purged_split_c1_date_screen_v1.json`](splits/manifests/purged_split_c1_date_screen_v1.json).**
It excludes 883 suspect validation/test articles and retains 6,167 train,
2,302 validation, and 2,236 test events. The original C1 manifest is a
historical reference. Retained news publication times remain unverified; report
that limitation when interpreting text gains. The team is proceeding with
model experiments after the prepared-input audit, without a separate
announcement-timing comparison.

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
| Full C1 text comparison | Old unscreened jobs `23812725`, `23812734`, and `23812735` were canceled. Screened input audit `23835057` **passed** (10,705 events, no exclusions, zero shared bars across splits). Training `23840368` is pending GPU scheduling; paired jobs `23840369`/`23840370` are pending on its completion. No full run has started. See the [audit result](analysis/text_value/c1_date_screen_v1_audit_summary.json) and [handoff](analysis/text_value/ARTICLE_DATE_AUDIT.md). | Monitor training and paired comparisons. |
| Return-task integration and controls | [PR #8](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/8) merged to `main` at `f0ac8ef`. Exact head `e0daa50` passed compilation, 30 affected tests, and 25 required text-value checks. Adaeze's smoke `24045929` and replay `24046557` completed, and frozen-input audit `24050520` passed. These validate execution and inputs, not full-model quality. | Launch new return work only from updated `main`, the verified frozen snapshot, and the merged R0–R3 protocol. Do not use eight-event scores as model-quality or text-value evidence. |
| Publication-time validity | Five source-checked test records have corrected calendar dates but no verified release times. The [evidence ledger](analysis/text_value/c1_article_date_policy_v1.csv) excludes those five and 878 more Reuters candidates; 10,705 retained records still have unknown release times. The screen retains 6,167 / 2,302 / 2,236 train/validation/test events. | Label the dependent run a date-screened sensitivity comparison. Do not claim publication-time-safe C1. |
| Jessica's padding fix | Merged to `main` in [commit `96cc6d7`](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/commit/96cc6d7900dedd8b867b39a951045fe3804aa778) with three CPU tests. | Ask for links to any Q-Former analysis missing from GitHub before assigning further work. |
| Gate, MoE, Q-Former diagnostics | Adaeze, Joye, and Jessica retain their existing streams. The inspected November 11 team job scripts used the earlier split; their outputs are historical diagnostics, not the C1 baseline. | Keep each checkpoint/split identity explicit; use the [owner handoff](TEAM_WORKFLOW.md#9-immediate-next-actions-agent-handoff-for-wednesday-november-11). |

The full C1 training and paired jobs are queued work, **not completed results**.
At the November 16 check, `23840368` remained pending for priority and
`23840369`/`23840370` remained pending on dependency; Adaeze did not alter them.
The prepared [eight-announcement BLS check](analysis/text_value/GOVERNMENT_CHECKED.md)
has source and price-order evidence, but its paired model comparison `23841902`
was canceled on November 13 at the user's request. It produced no model result.
Slurm jobs continue independently of a laptop or SSH session. The shared
account balance and queue can change; query them before planning more compute.
Large checkpoints, raw articles/prices, and full prediction NPZ files stay on
Delta. GitHub holds code, commands, hashes, and shareable derived records.

Adaeze's separate return jobs `24050724` and `24051590` were submitted from
pre-merge commits, then user-held with zero elapsed time after the frozen-input
R0–R3 integration became the source of truth. They produced no result and must
not be released or described as R0–R3 runs. The merged R0–R3 GPU jobs remain
planned, not submitted.

## Declined Expanse exchange

ACCESS declined the proposed request for 100 SDSC Expanse GPU-hours and 200 GB
of storage on November 16. No Expanse allocation, transfer, or model run
resulted. Do not describe Expanse as an active execution path or resubmit the
same queue-wait justification without a materially different provider-appropriate
case and team agreement. Delta remains the active resource.

## Next team actions

Reconfirm and cancel only Adaeze's two superseded jobs, then follow the
[Adaeze R0–R3 Delta runbook](docs/ADAEZE_R0_R3_DELTA_RUNBOOK.md). Create its
permanent Delta checkout from updated `main`, use the verified frozen snapshot,
and submit the merged protocol only after recording exact job provenance.
Continue monitoring Nicole's separate price job and its dependent controls.
Matched gate claims still require a completed full C1 checkpoint. Joye's
historical jobs must not be relabelled as C1 merely because later C1 code
merged to `main`.

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
