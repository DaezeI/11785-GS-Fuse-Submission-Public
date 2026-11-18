# Project status

**Current phase: controlled experiments, result analysis and final writing.**
Routine data preparation is complete. The [research home](https://gs-fuse-research.twiggydonkey4.chatgpt.site) reads the latest [research records](research_home/records.json) after updates reach GitHub main. Start with the
[next-phase plan](docs/NEXT_PHASE_PLAN.md); the experiment register and linked
result packs are the current evidence index.

**November 18 price evidence.** Both [screened price runs](results/README.md)
found nearly identical error with matched and shuffled articles. A [paired
CPU comparison](results/screened-price-v1/persistence.md) found that carrying
forward last prices beat seed 0 at all three horizons. The [decision
brief](docs/RESEARCH_DECISION_BRIEF.md) records the approved matched price-only
training comparisons and timing limits.

**Latest result update:** Nicole's screened-price baseline and both full paired
evaluations completed on H200. The [result pack](results/screened-price-v1/summary.md)
contains the measured finding and provenance. Adaeze's frozen-input return campaign
completed full training for R0, R1, and R2 on Delta on November 18 with verified checkpoints
and metrics. See the [R0–R2 results handoff](docs/updates/2026-11-18-adaeze-r0-r2-results.md)
and [campaign launch update](docs/updates/2026-11-17-adaeze-r0-r3-launch.md) for Delta paths,
fingerprints, and remaining queued jobs.

Updated November 18, 2026, after PR #8 merged the return-task implementation,
runtime fixes, frozen-input audit, and R0–R3 protocol to `main`, and full training
for R0, R1, and R2 completed on Delta.
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
| Full C1 text comparison | Nicole's full screened-price reference and both paired evaluations completed on her personal GPU resource. Both evaluations covered all 2,236 test examples with three shuffle draws. Her superseded Delta duplicates were canceled without starting. See the [result pack](results/screened-price-v1/summary.md). | Interpret the paired findings, preserve the large artifacts, and add missing reference analyses before broader claims. |
| Return-task integration and R0–R3 campaign | PRs #8 and #11–#13 merged to `main`. Frozen date-screened full training for R0 (`24055020`), R1 (`24055021`), and R2 (`24055022`) **completed `0:0`** (~7h 35m each), producing verified 1.8 GB checkpoints on 2,302 validation and 2,236 test events. Test MAE is 80.92 bp (R0), 81.08 bp (R1), and 81.08 bp (R2); test DHR is 50.5%–52.9%. Differences between R0–R2 are negligible ($\Delta\text{MAE} < 0.16\text{ bp}$, R1 vs R2 $\Delta\text{MAE} < 0.003\text{ bp}$). R3 full training (`24055023`) and paired control evaluations (`24055024`–`24055031`) are queued on Delta (`gpuA100x4`). See [R0–R2 results handoff](docs/updates/2026-11-18-adaeze-r0-r2-results.md). | Monitor queue for R3 training dispatch and paired zero-token/bypass evaluations. |
| Publication-time validity | Five source-checked test records have corrected calendar dates but no verified release times. The [evidence ledger](analysis/text_value/c1_article_date_policy_v1.csv) excludes those five and 878 more Reuters candidates; 10,705 retained records still have unknown release times. The screen retains 6,167 / 2,302 / 2,236 train/validation/test events. | Label the dependent run a date-screened sensitivity comparison. Do not claim publication-time-safe C1. |
| Jessica's padding fix | Merged to `main` in [commit `96cc6d7`](https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/commit/96cc6d7900dedd8b867b39a951045fe3804aa778) with three CPU tests. | Ask for links to any Q-Former analysis missing from GitHub before assigning further work. |
| Gate, MoE, Q-Former diagnostics | Adaeze, Joye, and Jessica retain their existing streams. The inspected November 11 team job scripts used the earlier split; their outputs are historical diagnostics, not the C1 baseline. | Keep each checkpoint/split identity explicit; use the [owner handoff](TEAM_WORKFLOW.md#9-immediate-next-actions-agent-handoff-for-wednesday-november-11). |

Historical queue observation before Nicole migrated the price reference:
at the November 16 check, `23840368` remained pending for priority and
`23840369`/`23840370` remained pending on dependency; Adaeze did not alter them.
The prepared [eight-announcement BLS check](analysis/text_value/GOVERNMENT_CHECKED.md)
has source and price-order evidence, but its paired model comparison `23841902`
was canceled on November 13 at the user's request. It produced no model result.
Slurm jobs continue independently of a laptop or SSH session. The shared
account balance and queue can change; query them before planning more compute.
Large checkpoints, raw articles/prices, and full prediction NPZ files stay on
Delta. GitHub holds code, commands, hashes, and shareable derived records.

Adaeze's superseded return jobs `24050724` and `24051590` were submitted from
pre-merge commits, then cancelled as intended; they produced no result and must
not be released or described as R0–R3 runs. Adaeze's frozen-input R0–R3 campaign
was submitted directly from `main` into `/work/hdd/bjjg/gs-fuse/runs/adaeze/return-r0-r3-v1`
using the guarded launcher. The single R3 smoke job `24055019` completed `0:0` in 45m23s.
Full training jobs for R0 (`24055020`), R1 (`24055021`), and R2 (`24055022`)
completed successfully with exit `0:0` (~7h 35m each) on November 18, generating
`ablation-r0-v1`, `ablation-r1-v1`, and `ablation-r2-v1`. R3 full training (`24055023`)
remains queued in `gpuA100x4` awaiting node availability. Dependent paired zero-token
controls for R0, R1, and R2 (`24055024`, `24055026`, `24055028`) were released from
dependency and are queued in priority order. Pending jobs are not results; smoke
metrics are execution checks, not full-C1 benchmark performance.

## Declined Expanse exchange

ACCESS declined the proposed request for 100 SDSC Expanse GPU-hours and 200 GB
of storage on November 16. No Expanse allocation, transfer, or model run
resulted. Do not describe Expanse as an active execution path or resubmit the
same queue-wait justification without a materially different provider-appropriate
case and team agreement. Delta remains the team's shared execution resource. Nicole's personal GPU
resource is an additional privately managed execution route.

## Next team actions

Follow [docs/NEXT_PHASE_PLAN.md](docs/NEXT_PHASE_PLAN.md): consolidate existing
owner-verified results, preserve large artifacts, identify missing matched
comparisons, and use the same numeric sources for paper and presentation.
Adaeze's return campaign is already submitted; leave remaining queued jobs
(`24055023`–`24055031`) untouched in the Delta queue so their accumulated
priority age is preserved. Monitor Delta queue turnover. Jessica's historical runs
remain historical. New Q-Former/MoE comparisons require a matching R3 reference,
not Nicole's completed price checkpoint. Joye's historical jobs must not be
relabelled as C1 merely because later C1 code merged to `main`. Routine data prep does not need
repeating. Provider/access details for Nicole's personal GPU resource stay local.

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
