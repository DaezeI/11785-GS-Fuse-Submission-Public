# Exact frozen 8B validation completed — November 26

Nicole completed `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add` on 171 training / 149 validation releases. MAE was 16.058/20.408 bp at 20/30 minutes versus the frozen CNN at 16.057/20.392. Paired comparisons with CNN, zero features and past-only shuffled text were inconclusive on ALL. All three seeds and frozen-price checks passed; the one-H200 job exited 0 in 7m46s. This is exploratory validation; test remains reserved pending the team freeze. [Result, metrics, plots and provenance](results/fed-spx-text-llama31-8b-v1/summary.md). Earlier running/held notes below are historical.

# Exact frozen 8B run started — November 26

Nicole’s exact `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add` is running on one H200 after all 12 pinned files and train/validation inputs passed verification. The existing held job was released with its fixed three-hour cap. No accuracy result or test score is claimed. [Protocol and verified model files](experiments/fed-spx-text-llama31-8b-v1/README.md).

# Meta access approved — November 26

Meta approval for Nicole’s exact `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add` was verified on the refreshed model page. The existing job stays on user hold pending personal download authentication and pinned-weight staging/verification. No GPU training or test scoring has run. Frozen CNN files, train/validation inputs and the shared scorer are ready. Earlier pending-review notes below are historical. [Exact protocol](experiments/fed-spx-text-llama31-8b-v1/README.md).

# Exact 8B input readiness — November 26

Nicole verified and staged all three frozen CNN checkpoints and their saved forecasts for `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add`. The real-input audit confirms 171/149 text rows, matching validation IDs and targets to numerical precision, and zero cross-split shared bars. The alignment table's canonical CRLF line endings were restored to match the existing frozen fingerprint. The shared paired scorer passes 15 tests. The exact one-GPU job remains user-held: Meta's access request is submitted and awaiting manual review. No GPU training, full 8B smoke, or test scoring has run. [Protocol and audit](experiments/fed-spx-text-llama31-8b-v1/README.md).

# Paired scorer ready — November 25

Adaeze packaged the shared Fed/SPX [paired scorer and forecast format](results/fed-spx-paired-scorer-v1/README.md) (#54), with the frozen `none+CNN~none~none`, `none+DLinear~none~none` and `none+Ridge~none~none` validation forecasts on `FED-PXTX-ALL-REL5-v2`. It reproduces `fed-spx-matched-control-v2` to within 2e-9 bp. No new model was trained; test was not loaded.

# Frozen CNN bundle staged — November 25

Adaeze staged the frozen `fed-spx-endpoints-price-v1` CNN bundle for Nicole's exact 8B arm (#55) in Drive `01 Results and Evidence/fed-spx-endpoints-price-v1-frozen-cnn.zip`. All three checkpoint hashes match the published metrics. The job stays on hold: the exact Llama-3.1-8B weights are still missing. See the [8B pilot protocol](experiments/fed-spx-text-llama31-8b-v1/README.md).

# Exact 8B submission update — November 24

Nicole's exact 8B job is submitted on user hold, with no GPU usage or training started. Release awaits the exact frozen CNN files and Llama-3.1-8B weights. Shared Delta searches found neither. Private scheduler receipt and recovery steps are in Nicole's handoff. The prepared runner and protocol merged in PR #63 at 013ca5c; test remains reserved.

# Person 4 restart — November 24, 2026

Fed/SPX dependencies merged in PR #62. Nicole owns #55: `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add` on `FED-PXTX-ALL-REL5-v2` (171 train after exclusions / 149 val). The [exact 8B pilot](experiments/fed-spx-text-llama31-8b-v1/README.md) is prepared; the frozen CNN checkpoint files and exact 8B weights are missing from her inspected workspace. No training job or test scoring is claimed. Adaeze owns the paired scorer package (#54). Older campaign prose below is historical context.

# Project status

**November 23 short-horizon Fed/SPX update.** Adaeze completed the compact
price-only validation run for 20- and 30-minute Close returns. The comparison
with zero return was inconclusive at both horizons. The result pack includes
the real-input window audit and historical text-version checks. A matched text
run needs verified versions on a common cohort. [Result pack](results/fed-spx-endpoints-price-v1/summary.md)

**November 23 Fed/SPX text evidence.** Exact-version audits now cover all 1,185
price-eligible Fed events: about 160 train and 121 validation rows are eligible once the cohort is
rebuilt with the Board re-check, and evidence classes stay labeled. A broader U.S. macro cohort has
2,158 candidate training events under a document-based evidence rule. No training ran.
[Update and next steps](docs/updates/2026-11-23-fed-spx-text-evidence.md)

**November 22 Fed/SPX handoff:** Adaeze completed two price-only validation
checks on Joye's official-Fed alignment v1. On 215 main
validation rows, last-close persistence had lower OHLC path MAE than ridge
at all three horizons. The [result and matching protocol](results/fed-spx-price-baselines-v1/summary.md)
describes the matched price-only comparison needed to measure the effect of
text. DLinear and the compact CNN await evaluation on this split. The Fed/SPX
held-out scores are pending.

**Latest Adaeze direction:** prioritize a local prices-plus-raw-news model with
a small trainable text encoder and no Llama feature-export dependency. The
[continuation handoff](experiments/improvement-v1/CONTINUE_WITHOUT_NICOLE.md)
records the verified laptop CUDA setup, public data link and remaining work.
Raw-text encoding is pending; current compact multimodal code uses precomputed
vectors. No real-data compact accuracy run has completed.

**New planned improvement campaign:** Adaeze prepared a separate
[controlled campaign](docs/experiments/IMPROVEMENT_CAMPAIGN_V1.md) for Nicole's
personal GPU resource: corrected-code reference, text LoRA, longer price
histories, text budgets and optional fusion variants. The revised default is five one-hour execution smokes; full training requires
resource/quota/audit readiness. A compact laptop price/text prototype loads
no Llama or MOMENT. See [the revised design](docs/experiments/AUDIT_FIRST_AND_COMPACT_V2.md).
Eleven additional designs require evidence or new code.
No jobs or new model results are implied. Existing completed results and
Delta jobs retain their recorded identities and status.

**Current phase: controlled experiments, result analysis and final writing.**
Routine data preparation is complete. The [research home](https://gs-fuse-research.twiggydonkey4.chatgpt.site) reads the latest [research records](research_home/records.json) after updates reach GitHub main. Start with the
[next-phase plan](docs/NEXT_PHASE_PLAN.md); the experiment register and linked
result packs are the current evidence index.

**November 18 R3 result.** Nicole ran Adaeze’s exact R3 configuration on her
personal GPU resource. Training and both full article checks passed. Correct
and shuffled articles had virtually identical error, and zero return had
slightly lower mean absolute error. The [R3 result pack](results/adaeze-screened-r3-h200-v2/summary.md)
separates this H200 result from Adaeze’s pending Delta campaign.

**November 18 price evidence.** Both independently trained price-only runs
completed. The [matched comparison](results/screened-price-only-v1/summary.md)
shows a gain for the news-trained package over the corresponding price-only
reference in both seeds. The last-price forecast had the lowest test error
at all three horizons. Earlier matched versus shuffled results were nearly
equal. See the [6 p.m. meeting brief](docs/RESEARCH_DECISION_BRIEF.md#november-18-group-meeting).

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
