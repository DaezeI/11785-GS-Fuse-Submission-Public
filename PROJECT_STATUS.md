# Project status

Updated November 15, 2026, after Adaeze began preparing an Expanse alternative
because of the Delta scheduling backlog and inspected the pending exchange.
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
| Full C1 text comparison | Old unscreened jobs `23812725`, `23812734`, and `23812735` were canceled. Screened input audit `23835057` **passed** (10,705 events, no exclusions, zero shared bars across splits). On November 15, `squeue` and `sacct` verified that Nicole's training job `23840368` and paired jobs `23840369`/`23840370` were still pending and had used zero run time. No full result is recorded. See the [audit result](analysis/text_value/c1_date_screen_v1_audit_summary.json) and [handoff](analysis/text_value/ARTICLE_DATE_AUDIT.md). | Leave these owner-specific jobs unchanged until Nicole confirms whether they should continue. Prepare Expanse as an alternative, but submit nothing before approval and account provisioning. |
| Publication-time validity | Five source-checked test records have corrected calendar dates but no verified release times. The [evidence ledger](analysis/text_value/c1_article_date_policy_v1.csv) excludes those five and 878 more Reuters candidates; 10,705 retained records still have unknown release times. The screen retains 6,167 / 2,302 / 2,236 train/validation/test events. | Label the dependent run a date-screened sensitivity comparison. Do not claim publication-time-safe C1. |
| Jessica's padding fix | Merged to `main` in [commit `96cc6d7`](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/commit/96cc6d7900dedd8b867b39a951045fe3804aa778) with three CPU tests. | Ask for links to any Q-Former analysis missing from GitHub before assigning further work. |
| Gate, MoE, Q-Former diagnostics | Adaeze, Joye, and Jessica retain their existing streams. The inspected November 11 team job scripts used the earlier split; their outputs are historical diagnostics, not the C1 baseline. | Keep each checkpoint/split identity explicit; use the [owner handoff](TEAM_WORKFLOW.md#9-immediate-next-actions-agent-handoff-for-wednesday-november-11). |

The full C1 training and paired jobs have no recorded completed result. They
were verified as pending with zero elapsed time on November 15. They belong to
Nicole's screened-C1 stream and must not be canceled or changed without owner
confirmation. Adaeze's Expanse preparation is an alternative execution path;
it does not supersede or release those jobs.
The prepared [eight-announcement BLS check](analysis/text_value/GOVERNMENT_CHECKED.md)
has source and price-order evidence, but its paired model comparison `23841902`
was canceled on November 13 at the user's request. It produced no model result.
Slurm jobs continue independently of a laptop or SSH session. The shared
account balance and queue can change; query them before planning more compute.
Large checkpoints, raw articles/prices, and full prediction NPZ files stay on
Delta. GitHub holds code, commands, hashes, and shareable derived records.

## Planned Expanse migration

On November 12, Nicole submitted an ACCESS exchange request for **100 SDSC
Expanse GPU-hours and 200 GB of Expanse Projects Storage** (5,585 ACCESS
credits). Direct inspection of the ACCESS project History tab on November 15
still showed the exchange **Under Review**. The Credits + Resources graph shows
the requested 100 GPU-hours and 200 GB, but the page explicitly says those
balances reflect the pending request. They are not an active award yet. It is a
single pool for the team project, not a per-person allocation; the existing
Delta allocation is separate. The [course migration note](https://piazza.com/class/mo4ko44arnc3sg/post/659)
allows other ACCESS resources, and the [ACCESS exchange guide](https://allocations.access-ci.org/how-to)
explains provider approval.

The requested 100 GPU-hours are the current proposed starting allocation, not
a documented permanent project maximum. The team may request or exchange for
additional hours later. Plan the first approved allocation carefully, but do
not describe 100 hours as a fixed scientific limit until ACCESS confirms the
available expansion path.

The Expanse portal reports no account while the exchange remains pending. Do
not treat this as active access or transfer data yet. After approval, verify
individual provisioning without committing private account records, plus the
assigned project code, storage paths, and the 32 GB V100 environment. Then run
the login check and bounded `gpu-debug` validation in
[`scripts/expanse`](scripts/expanse/README.md) before any model job. The
A100-tested precision/cache configuration must not be assumed compatible with
V100 merely because paths or filenames match.

## Next team actions

Leave the three verified-pending Delta jobs unchanged until Nicole confirms
their intended disposition, wait for the Expanse exchange to be approved, and
finish the parameterized migration preparation.
Once approved, record the project code, provisioned users, storage paths,
environment, hashes, and a short V100 smoke result before full training. Also
choose an owner and policy for article/price timing, and have each diagnostic
owner link an artifact or record a concrete blocker. Adaeze can debug gate instrumentation on Nicole's
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
