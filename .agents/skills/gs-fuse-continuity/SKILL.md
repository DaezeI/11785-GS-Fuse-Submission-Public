---
name: gs-fuse-continuity
description: Resume, maintain, and close CMU GS-Fuse research and development sessions across new ChatGPT chats. Use for requests such as "Continue GS-Fuse", "start GS-Fuse session", "end GS-Fuse session", or questions about GS-Fuse project status, documentation, experiments, handoffs, GitHub repository, Research Home, and result evidence. Fetch and follow the current repository guidance rather than relying on a frozen project summary. Preserve local edits and scientific integrity.
---

# GS-Fuse continuity

Project: https://github.com/DaezeI/11785-GS-Fuse-Submission-Public

## Source of truth and access

1. Retrieve the **current** default branch (`origin/main` for Git tooling) before claiming the repository is up to date. Start by reading `AGENTS.md`, `TEAM_WORKFLOW.md`, `docs/RESEARCH_HOME.md`, and `.agents/skills/gsfuse-result-report/SKILL.md`; also review `PROJECT_STATUS.md`, `research_home/records.json`, recent `docs/updates/`, and task-specific protocols as applicable. Apply the repository's result-reporting skill by default, subject to its current instructions.
2. Treat repository files, GitHub PRs, and authoritative current project records as more current than any remembered snapshot in this skill or an earlier chat. Check timestamps, branches, commits and whether the work is merged. Do not confuse a result registry entry with clearance of an underlying dataset.
3. In a repository-connected computer environment, check `git status` **before** modifying anything; fetch origin; never discard local edits, switch dirty worktrees, or overwrite unpublished data. For changes, create a focused branch/worktree based on the latest `origin/main` following repo conventions. Inspect the diff and tests before commits or PRs.
4. In a browser-only chat, read the public repo through available browsing tools and clearly distinguish remotely verified information from unavailable local Windows status. A GitHub link does **not** give access to a Windows checkout, run outputs, or private accounts. Do not claim a Git action, backup, or upload occurred unless verified. Suggest Work/repository-connected tooling for actual operations, or provide one Windows PowerShell 5.1-compatible copy-paste block if necessary. Do not assume GitHub CLI (`gh`) is installed.
5. Do not recreate or re-merge already merged work. Avoid redundant requests for details already supplied or recorded. When records conflict, report the discrepancy and inspect the actual current state.

## Continue GS-Fuse

- Reconcile current instructions, Git status, Research Home registry, recent project updates, open PRs and any user-supplied handoff.
- State the verified baseline, live blockers, and next authorized unblocked task briefly; then **do the task** using available tools rather than only proposing a plan.
- Distinguish exploratory validation, screened C1 evidence, and held-out test evidence; obey dataset permission, timestamp, leakage, and evaluation restrictions in the current protocol.
- Do not launch training, retune on an exhausted validation set, access held-out tests, move unpublished raw data, or publish logs/checkpoints without explicit task authorization and protocol justification.

## Record material work continuously

- Preserve meaningful research, experiment outcomes (including failures), implementation, debugging, documentation, planning, decisions, unsuccessful approaches, verification, limitations and unresolved issues.
- Follow **existing** record destinations: `results/<run-id>/` and `research_home/records.json` for reportable results, `EXPERIMENTS.md` if mandated, `PROJECT_STATUS.md` for current state, dated `docs/updates/` for nonduplicative durable development/history, and Git commits/PRs for code/document changes. Inspect actual repository requirements before choosing the destination.
- Record rationale, evidence, versions, run IDs, relevant environment and data lineage, constraints, change identifiers, and exact follow-up where material. Do not create an arbitrary "experiment" for ordinary planning/debugging or a redundant separate session-log system.
- Respect sanitized-evidence and private-artifact constraints. Verify backups using destination readback when possible; user confirmation without readback is not independently verified.

## End GS-Fuse session

- Compare activity in the current session with durable project records; update the proper existing files for material changes, without inventing results or unnecessary writes.
- Apply current result-reporting skill where relevant; reconcile Git status, branch/commit/PR status, validation, pending artifacts, backups, blockers and owner-specific follow-ups.
- Provide a **short handoff** with: verified state; work done and exact refs; unsuccessful attempts and important decisions; limitations and permissions; uncommitted or unpublished work; next unblocked task and blocked tasks. Keep detailed history in project records, not only in the chat.
- Never silently push, merge, upload, delete, rerun training, or access held-out evaluation. State anything not verifiable as pending.

## Response conventions

Be concise but precise. Avoid rehashing an entire experiment when only its status changed. When a user cannot connect a computer, make PowerShell commands compatible with Windows PowerShell 5.1, guard missing tools, use one block, and preserve all existing work. When a new chat lacks tools needed to update the repository, explain that limitation, provide grounded next steps, and do not promise background work.
