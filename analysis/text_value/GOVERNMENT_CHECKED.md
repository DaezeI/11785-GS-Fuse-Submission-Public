# Checked government-announcement comparison

**Status:** eight held-out test announcements passed source, text-identity, and
price-order checks. GPU text comparisons wait for the date-screened C1
checkpoint from Slurm job `23840368`; there are no model results yet.

The exact [eight-record evidence set](government_announcements_checked_v1.json)
contains four BLS CPI and four BLS Employment Situation announcements from
August 2023 to March 2024. Each linked BLS archive states an 08:30 Eastern
embargo release. For each stored generated summary, the report period and two
distinctive headline figures match its official release; SHA-256 pins the
checked text. The [Delta audit](government_announcements_checked_v1_audit.json)
confirmed all eight IDs remain in the date-screened C1 **test** partition,
with no missing files or changed hashes. Several other inspected summaries were
not selected because their report identity or figures were ambiguous or
inconsistent. This check establishes identity and headline agreement, not
perfect accuracy of every generated sentence.

For every selected announcement, the 35-bar input ends at 16:15 on the prior
trading day and the first of 140 target bars is labeled 09:30 on the release
day, one hour after the official release. The SP500 CSV has no timezone field;
this comparison interprets its 09:30 market-open labels as New York local time.
That assumption and the absence of verified publication times for the broad
training corpus limit any causal claim.

The [paired job](../../scripts/delta/nicole_c1_government_checked_paired.sbatch)
uses the **same eight IDs and price windows** for real text, three seeded
within-category shuffled texts, and two separately labeled no-text modes.
Each category has four cases, permitting a complete derangement within type.
The runner fails if an ID is missing or duplicated and records the evidence
file hash, selected indices, donor IDs, checkpoint hash, and price/manifest
hashes. Its `afterok` dependency on training must be recorded when submitted.
The checkpoint itself is trained on the larger date-screened *exploratory*
corpus; the checked-set evaluation is more trustworthy on release timing but
small and is not a clean-training benchmark.
