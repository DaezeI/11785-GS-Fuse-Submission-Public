# Revised prompt to send to Nicole's agent

Continue Adaeze's campaign from branch adaeze/controlled-improvement-campaign,
PR #34 in DaezeI/11785-GS-Fuse-Submission-Public. Read AGENTS.md,
docs/experiments/AUDIT_FIRST_AND_COMPACT_V2.md and
experiments/improvement-v1/matrix.json. This supersedes
the previous immediate six-full-run launch. Adaeze owns this campaign; keep
other owners' completed/active experiments separate. Use your local research
and GPU skills and private handoff.

1. Find the exact newer Q-Former failure: job, code/config, traceback and
   accounting. Adaeze tentatively recalls "storage quota exceeded"; unverified.
   The earlier Q-Former smoke passed. The linked c1-article-readout-v1 was a
   successful CPU run, now merged; reuse it rather than duplicating H35.
   Frozen weights still compute attention, so historical padding attention
   does not contradict a frozen Q-Former. Bypass is a different claim to audit.
2. Distinguish EDQUOT, ENOSPC, CUDA OOM, RAM and walltime. Run resources.py
   read-only diagnostics and inspect actual user/group/project/inode quotas;
   df alone is insufficient. Inspect output, token/model caches, temp and W&B
   paths. Plan approved storage and backups before retrying. Delete no results
   or other owners' artifacts. Q-Former training does not unfreeze Llama's 1B
   base; LoRA requires an explicit flag. No blanket model/dataset replacement.
3. Pin a separate clean checkout and existing frozen snapshot. Review/test
   the corrected stage-best checkpoint handoff, explicit Stage-2 price inputs
   and contrastive-anchor switch. Run real-input L/H audits before GPU work;
   require unchanged 6167/2302/2236 IDs/targets and no shared bars. A changed
   cohort needs a versioned design for all comparators. Times remain unverified.
4. After resource diagnosis, render the DEFAULT five seed-0 SMOKES only:
   ref, no-contrastive, no-invariance, clean-anchor, text-warmup. One-hour PBS
   profile, one lane initially, maximum five GPU-hours. These 16/8/8 checks
   verify execution, not accuracy. No automatic full-run promotion. The GLOBAL
   ceiling remains two single-GPU jobs only if all active work/storage fits.
5. Defer LoRA/context search. Bind actual resource, quota, audit, stage-handoff
   and gradient evidence to the final commit/snapshot in a private readiness
   receipt. For the first full comparison choose ref plus ONE hypothesis,
   two six-hour caps; render explicitly --phase full --readiness. Do not tick
   readiness boxes without evidence. Validation selects; quarantine test
   summaries and preserve negative/failed attempts. Leave Delta jobs untouched.
6. Advance the compact laptop track: model/compact_return.py and
   scripts/improvement/train_compact.py load no Llama/MOMENT. Adaeze's target
   is RTX 4060 Laptop 8 GB, with T4 16 GB secondary. Measure actual speed,
   VRAM, artifacts and aligned error against zero return and references.
   Current compact trainer is price-only. A tiny pooled-text correction is
   designed/model-supported but still needs feature export/control training.
   Do not promise equal quality or call synthetic checks market results.
7. Recover raw article timestamps/versions and the price clock in parallel.
   Audit the upstream FNSPID UTC conversion issue without assuming our copy
   inherited it. Keep existing screened C1 for conditional comparisons; a new
   verified-time cohort needs versioning, repurging and matched retraining.
   New data is necessary only if source evidence cannot support the question.
   Do not relaunch the canceled eight-announcement job.
8. Use the reporting skill: exact commit/config/hashes/job, resource diagnosis,
   counts, metrics/uncertainty, failures, limitations, figure, registry update,
   focused result PR, main/Research Home readback and reviewed aggregate ZIP
   backup/readback in Final Deliverables / 01 Results and Evidence. Large
   artifacts stay on agreed compute storage. Keep provider/account/access
   details private. Return verified diagnosis before claiming a fix worked.
