# GS-Fuse research home

Open the [GS-Fuse Research Home](https://gs-fuse-research.twiggydonkey4.chatgpt.site) for the current findings and the latest reported run status. This is the fresh-chat entry point for Nicole's GS-Fuse work. Read the repository's [agent instructions](../AGENTS.md), [team workflow](../TEAM_WORKFLOW.md), [project status](../PROJECT_STATUS.md), [experiment register](../EXPERIMENTS.md), and [results guide](../results/README.md). For a submitted run, recover its private campaign record and job IDs from the top of the local handoff. For private access and continuity, Nicole's local `PROJECT_HANDOFF.md` and `nicole-gpu-research` skill supplement these public records. Current user instructions take precedence over dated plans.

## Latest research decision

The [November 18 decision brief](RESEARCH_DECISION_BRIEF.md) reviews the two
screened price runs and teammates' available evidence. It includes a new
[last-price comparison](../results/screened-price-v1/persistence.md), three
follow-up experiments, and the completed [two-seed price-only comparison](../results/screened-price-only-v1/summary.md).

## Continue the work

1. Inspect the checkout and current records, including the research registry at [research_home/records.json](../research_home/records.json). Identify owners, existing evidence, open changes, and missing information. Separate the project's authorized scope from what has actually run or been verified.
2. Before scheduling, check live scheduler state and owner records; continue an existing campaign where appropriate instead of duplicating it. Preserve frozen inputs and respect teammate ownership. Nicole's personal GPU resource is limited to two concurrent single-H200 jobs. Cluster access and account details stay local.
3. Nicole authorizes routine, reasonable experiments within this project without separate approval for every batch. Define a bounded question and compute budget, use matched controls and record provenance. Avoid open-ended compute. Follow [EXPERIMENTS.md](../EXPERIMENTS.md), including for failed or negative results.
4. Follow the [screened price execution recipe](../scripts/research_cycle/README.md) when repeating the price comparison. For other questions, use the owner’s approved configuration and write a result pack in the same format. Update the registry from verified evidence using `python scripts/research_home.py upsert /path/to/public-record.json`. The live home reads the registry from GitHub main, so result updates need no change to the page or its deployment.
5. Report status, finding, evidence, limits, decision, and the next action in plain language. Distinguish queued work from results, exploratory analysis from confirmation, and inference interventions from retrained baselines. Preserve the limitation that retained article release times remain unverified.

## Publish a teammate's experiment

Pull GitHub main and open the repository in Codex. Ask Codex to prepare a
config, run an experiment, or analyze results as you normally would. The
[repository instructions](../AGENTS.md#report-experiment-work-by-default) tell
Codex to prepare the Research Home update as part of the task. They apply to
Delta and Nicole's personal GPU resource. When Nicole runs a teammate's
config, the teammate remains the experiment owner.

The same PR includes the verified record and available evidence. After it
merges to main, reload the Research Home. Contributors working without Codex
can use the commands in the
[shared skill](../.agents/skills/gsfuse-result-report/SKILL.md).

## Finish a completed experiment

A request to run an experiment through completion or collect its results
includes delivery to the standard project destinations. Start as soon as
the run and its required evaluations are verified.

1. Save the curated result pack under `results/<run-id>/` and publish it
   with the experiment register and registry update in a focused GitHub PR.
   Read back the merged record on `main`. The existing Research Home reads
   that record on refresh.
2. Archive the reviewed result pack from the published commit. For example,

   ```sh
   git archive --format=zip --output "$RESULT_ZIP" "$PUBLISHED_COMMIT" -- "results/$RUN_ID"
   unzip -tq "$RESULT_ZIP"
   ```

   Upload the aggregate ZIP to **Final Deliverables → 01 Results and
   Evidence**. Reuse a verified backup of the same version when it already
   exists. Read back the uploaded file in the folder and record its link
   and size in the run handoff. The ZIP contains the shareable summary,
   metrics, figures and scientific provenance. Checkpoints, per-event
   predictions, raw data and operational logs stay in their existing
   approved storage.
3. Hand back the finding and the verified GitHub and Drive links. If either
   destination is blocked, complete the other and record the remaining
   step and reason. Keep a verified local ZIP ready for the upload.

The scheduler can finish training and write outputs while Codex is away.
Collection and publication run in the active task. When a task stops
after launch, resume from the recorded job and allow the verified
completion to trigger this delivery during the next requested check.

## Where work lives

- **GitHub:** canonical versioned code, configs, research records, results, figures, and paper source. [EXPERIMENTS.md](../EXPERIMENTS.md) indexes runs; [results](../results/README.md) contains evidence packs; [report/final](../report/final) holds paper and presentation source.
- **Drive Final Deliverables:** exports, team handoffs, and backups. Verify full checkpoint and prediction backups separately from small result exports.
- **Overleaf:** final polishing; return substantive edits to GitHub.

For public descriptions of additional compute, use **Nicole's personal GPU resource**. Hardware models may appear in scientific provenance; provider, account, reservation and access details remain private.

## Maintain the research home

The browser loads the latest [GitHub registry](../research_home/records.json) and displays its update date. If GitHub is unreachable, it labels the bundled fallback. Check changes with `python scripts/research_home.py validate`. Use a focused PR for new records, figures and evidence. Update the Site source only for interface changes. Copy the files under `research_home/web/` and the current registry to the static Site’s `dist/` directory. Use the existing Site identity in the local handoff to package and publish it. Keep the current audience.
