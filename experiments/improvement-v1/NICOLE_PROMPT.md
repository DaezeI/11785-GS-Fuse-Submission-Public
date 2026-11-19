# Prompt to send to Nicole's agent

Please execute Adaeze's controlled improvement campaign from branch
`adaeze/controlled-improvement-campaign` in
`https://github.com/DaezeI/11785-GS-Fuse-Submission-Public`.
Adaeze remains the experiment owner; Nicole is the executor. Use your local
research companion/GPU skills and private access handoff. Read AGENTS.md,
docs/experiments/IMPROVEMENT_CAMPAIGN_V1.md and the committed
experiments/improvement-v1/matrix.json first.

1. Fetch the branch into a separate checkout and pin its full commit. Inspect
   live scheduler state and owner records first; avoid duplicating completed
   or active work. Keep current Delta jobs untouched. Use Nicole's personal
   GPU resource, with a GLOBAL maximum of two concurrent single-GPU jobs.
   No provider, account, reservation, credentials or private access details
   go into public records.
2. Review the code diff and run the focused CPU tests listed in VALIDATION.md.
   The branch repairs LoRA freeze/optimizer wiring, stale feature caches,
   projection gradients and adapter save/reload; old R3 is not a matched
   reference. Require the fresh `ref` on this corrected code. Confirm
   Llama-3.2-1B/MOMENT versions and the frozen snapshot with content SHA-256
   `049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`.
3. Create a PRIVATE profile with actual absolute paths, environment setup,
   PBS single-GPU resources and a six-hour walltime (schema in the design).
   Before any GPU submission, run the real-input audits for L=35/140/280
   in an appropriate CPU allocation. If an expanded history overlaps splits
   or changes the population, record that failure and prepare a new versioned
   common-cohort design for every arm; do not submit the current full screen
   or silently change its committed inputs.
   Keep the checkout clean and outputs outside it. Render the DEFAULT
   seed-0 validation screen with `scripts/improvement/campaign.py render`.
   It includes ref, lora8, context140, context280, tokens2048 and the
   lora8-context140 interaction. Review the generated package, install it
   at the printed resource_dir and submit it ONCE with submit.py and the
   available-slot count verified from the live queue. The two dependency
   lanes bound concurrency. The six-job screen cap is 36 GPU-hours.
4. Each task must pass its actual L/H emitted-window audit, retain exactly
   6167/2302/2236 events with zero cross-split shared bars, and pass its own
   three-stage smoke/checkpoint reload before full training. Confirm LoRA
   parameters change in Stage 3, sampled base weights stay fixed, and
   adapted train/eval never reuse static embeddings. Inspect actual MOMENT
   long-context patch behavior. Never silently alter precision, batch size,
   losses, context length, population or cache policy to recover a failure.
   Record failed attempts and stop a systematically failing lane.
5. Screen only on validation: primary metric H=140 endpoint CLOSE-return
   MAE in bp, secondary H=35/70, DHR with eligible counts and all-channel
   errors. Evaluate matched, three shuffled assignments, zero tokens,
   fusion bypass and zero-return/training-majority references. Keep legacy
   trainer-produced test summaries quarantined during selection. Account
   for repeated dates with paired date-block intervals. Random shuffles
   can draw future donors and are diagnostic, not a timing certificate.
6. Promote at most two candidates with >=1% validation MAE improvement
   over fresh ref. Save an immutable selection JSON with exact variants,
   evidence hashes and rule before test scoring. If none qualify, report
   the negative result and stop broad search. For promoted variants plus
   ref run seeds 1 and 2 (at most six further six-hour jobs), then evaluate
   all locked seeds including seed 0 on test. Reuse verified seed-0 screen
   checkpoints for test evaluation rather than retraining them. Apply the
   design's uncertainty and multiple-comparison rules. No best-seed claims.
7. The other executable arms are optional follow-ups; the eleven gated
   designs need evidence/implementation before any submission. Prioritize
   the CPU ridge readout and publication/version/price-clock ledger if
   semantic text gains remain absent. Do not relaunch the canceled
   eight-announcement job. The retained reports are date-screened, with
   unverified intraday availability; keep that limitation in every result.
8. Follow through on collection and standard delivery: verify accounting,
   sample alignment, checkpoint/artifact hashes, environment and exact
   settings; save metrics, uncertainty, a figure and a plain-language
   finding (including negative/failed runs); update each stable registry
   record and open focused result PRs. Verify published main/Research Home,
   then back up the reviewed aggregate ZIP to Drive's Final Deliverables /
   01 Results and Evidence and read back the receipt. Keep large/private
   artifacts on agreed compute storage. Report job IDs, current status,
   completed checks, result links and any concrete blocker without labeling
   a submitted job or smoke as a benchmark result.
