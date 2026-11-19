---
name: gsfuse-result-report
description: Use when a GS-Fuse contributor wants to register an experiment, report a verified status, or add completed results and figures to the team's Research Home through a GitHub PR.
---

# Report a GS-Fuse experiment

Read the repository's AGENTS.md and the current record in
`research_home/records.json`. Use `results/TEMPLATE.md` for scientific
evidence and `docs/RESEARCH_HOME.md` for the team workflow.

1. Identify the owner, stable run ID, question, reference run, exact commit
   and config, and what changed. Reuse the run ID for status updates. Use a
   new ID for a new attempt. If Nicole will run a teammate's experiment,
   collect their committed config and keep the teammate identified as owner.
2. Check the owner's evidence before setting the state. A submitted job has
   no finding. For a completed run, verify the evaluation, data version and
   sample counts, metric units, and agreement with the figure. Preserve
   uncertainty and limits. Keep analyses of the same checkpoint linked.
3. Save shareable evidence under `results/<run-id>/`. Include a plain-language
   summary, numeric metrics, a figure with alt text, and configuration and
   provenance. Keep raw inputs, large model files, and account information in
   the agreed storage. In public records use **Nicole's personal GPU resource**
   when referring to Nicole's additional compute.
4. Reuse a collector's `candidate-record.json`, or use
   `results/screened-price-seed1-v1/candidate-record.json` as the shape for a
   new record. Set the real owner, question, configuration, state, finding,
   limitations, evidence URLs, related run IDs, figure and timestamp.
   Pending runs can use `null` for the figure and should link to the committed
   config or protocol. Use main-branch URLs for evidence and raw GitHub URLs
   for figures. Include their files in the same PR so the links work at merge.
5. From the repository root, apply and validate the record:

   ```sh
   python3 scripts/research_home.py upsert /path/to/candidate-record.json
   python3 scripts/research_home.py validate
   git diff --check
   ```

   Put the resulting registry change and its evidence in one focused PR. If the
   registry changed while the PR was open, reapply the candidate on the latest
   main before merging so other runs survive. The same Research Home reads
   GitHub main after merge.
6. Finish the [standard result delivery](../../../docs/RESEARCH_HOME.md#finish-a-completed-experiment)
   for a verified run. Verify the registry on GitHub main, then archive the
   reviewed result directory and upload the aggregate ZIP to **Final
   Deliverables → 01 Results and Evidence**. Read back the receipt and share
   its link. Record any blocked destination and finish the remaining steps.
   When pausing after submission, leave pending delivery in the run handoff
   so collection can resume without a separate reporting request.

This skill handles reporting and handoff. Launch or cancel a GPU job only
within the run owner's authorization and documented compute workflow.
