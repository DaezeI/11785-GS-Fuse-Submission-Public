# GS-Fuse research home

Open the [GS-Fuse Research Home](https://gs-fuse-research.twiggydonkey4.chatgpt.site) for the current findings and the latest reported run status. This is the fresh-chat entry point for Nicole's GS-Fuse work. Read the repository's [agent instructions](../AGENTS.md), [team workflow](../TEAM_WORKFLOW.md), [project status](../PROJECT_STATUS.md), [experiment register](../EXPERIMENTS.md), and [results guide](../results/README.md). For a submitted run, recover its private campaign record and job IDs from the top of the local handoff. For private access and continuity, Nicole's local `PROJECT_HANDOFF.md` and `nicole-gpu-research` skill supplement these public records. Current user instructions take precedence over dated plans.

## Latest research decision

The [November 18 decision brief](RESEARCH_DECISION_BRIEF.md) reviews the two
screened price runs and teammates' available evidence. It includes a new
[last-price comparison](../results/screened-price-v1/persistence.md), three
follow-up experiments, and the approved price-only batch.

## Continue the work

1. Inspect the checkout and current records, including the research registry at [research_home/records.json](../research_home/records.json). Identify owners, existing evidence, open changes, and missing information. Separate the project's authorized scope from what has actually run or been verified.
2. Before scheduling, check live scheduler state and owner records; continue an existing campaign where appropriate instead of duplicating it. Preserve frozen inputs and respect teammate ownership. Nicole's personal GPU resource is limited to two concurrent single-H200 jobs. Cluster access and account details stay local.
3. Nicole authorizes routine, reasonable experiments within this project without separate approval for every batch. Define a bounded question and compute budget, use matched controls and record provenance. Avoid open-ended compute. Follow [EXPERIMENTS.md](../EXPERIMENTS.md), including for failed or negative results.
4. Follow the [screened price execution recipe](../scripts/research_cycle/README.md) when repeating the price comparison. For other questions, use the owner’s approved configuration and write a result pack in the same format. Update the registry from verified evidence using `python scripts/research_home.py upsert /path/to/public-record.json`. The live home reads the registry from GitHub main, so result updates need no change to the page or its deployment.
5. Report status, finding, evidence, limits, decision, and the next action in plain language. Distinguish queued work from results, exploratory analysis from confirmation, and inference interventions from retrained baselines. Preserve the limitation that retained article release times remain unverified.

## Where work lives

- **GitHub:** canonical versioned code, configs, research records, results, figures, and paper source. [EXPERIMENTS.md](../EXPERIMENTS.md) indexes runs; [results](../results/README.md) contains evidence packs; [report/final](../report/final) holds paper and presentation source.
- **Drive Final Deliverables:** exports, team handoffs, and backups. Verify full checkpoint and prediction backups separately from small result exports.
- **Overleaf:** final polishing; return substantive edits to GitHub.

For public descriptions of additional compute, use **Nicole's personal GPU resource**. Hardware models may appear in scientific provenance; provider, account, reservation and access details remain private.

## Maintain the research home

The browser loads the latest [GitHub registry](../research_home/records.json) and displays its update date. If GitHub is unreachable, it labels the bundled fallback. Check changes with `python scripts/research_home.py validate`. Use a focused PR for new records, figures and evidence. Update the Site source only for interface changes. Copy the files under `research_home/web/` and the current registry to the static Site’s `dist/` directory. Use the existing Site identity in the local handoff to package and publish it. Keep the current audience.
