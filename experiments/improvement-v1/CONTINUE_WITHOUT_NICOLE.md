# Continue independently on Adaeze's laptop

Latest user direction: minimize Nicole's involvement, use the local GPU, and
continue from another account. Do not request Llama feature exports or launch
remote jobs as the default next step. This document supersedes the earlier
recommendation to wait for Nicole's pooled-feature export. It is a handoff,
not a claim that the raw-text training route is implemented.

## Paste this into the next coding chat

Continue work in this existing checkout:

```text
C:\Users\Adaeze Ilodianya\Documents\Fall 2026\11785 Intro to Deep Learning\GS-FUSE-main\11785-GS-Fuse-Submission-Public
```

Repository: https://github.com/DaezeI/11785-GS-Fuse-Submission-Public

Branch: `adaeze/controlled-improvement-campaign`

PR: https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/pull/34

Read AGENTS.md and the required workflow/skill files. Then read this handoff,
`docs/experiments/AUDIT_FIRST_AND_COMPACT_V2.md`, and the compact model/trainer.
Preserve unrelated edits. Implement and verify the local raw-text route below;
do not merely produce another plan. Keep replies concise to conserve usage.

### Objective and implementation still needed

Train a genuinely multimodal prices-plus-news model on my RTX 4060 Laptop
(8,188 MiB VRAM), without Llama, MOMENT, cached Llama outputs, or Nicole's compute.
Price-only remains a comparison, not the final objective. Do not promise equal
accuracy before experiments. Switching encoders requires fresh training and
matched evaluation; it does not automatically require a different dataset.

Reuse the compact causal price network. Add an initial small trainable text
encoder (for example masked mean word embeddings plus a projection, starting
with vocabulary 16,000, embedding dimension 64 and maximum 512 tokens). Fit
vocabulary and every learned preprocessing parameter on training articles only;
use explicit PAD/UNK handling, exclude PAD from pooling, and handle empty text.
This is a starting design, not a measured optimum. Train text embeddings through
the forecasting loss; do not cache their changing outputs. Token IDs may be
cached with tokenizer/content hashes. Save tokenizer and preprocessing with
checkpoints. A pretrained small encoder can be a later controlled alternative.

First implement a loader from raw article text plus price series, respecting
the screened C1 protocol and existing return definitions. Avoid loading any
foundation-model weights. Add a PowerShell entrypoint using the working GPU
environment, fresh output directories, bounded checkpoints and explicit CUDA
checks. Existing run_local.ps1 currently expects a portable feature pack; do not
present it as a working raw-text command.

Prespecify three initial comparisons, seeds 0/1/2 after a one-epoch smoke:

1. Existing compact price-only baseline, trained locally.
2. A trainable text residual on that exact frozen price checkpoint. Freeze
   only the price reference here; the new text encoder must receive gradients.
3. Jointly trained price/text model, clearly labeled a different training
   package rather than proof of one component's effect.

Keep L35 and H35/70/140 initially, train-only scaling, the current endpoint
close-return task and validation-H140-MAE checkpoint selection. Prespecify a
bounded epoch/patience budget. Check zero-return and correctly paired price-only
references and matched/shuffled text on validation. Shuffle within the existing
eligible split/category/time rules and report donor coverage. Lock configuration
before final test evaluation; never select changes using test scores. Report
runtime, peak GPU memory, disk usage and parameters alongside accuracy. Do not
run the old remote campaign or canceled timing jobs.

### Data: remaining dependency, investigate before asking anyone

No raw events/series were found in this checkout or the bounded searches of the
course folder and Downloads. This does not establish that none exist elsewhere.
Check existing local/team locations before asking someone to recreate data.

The upstream README advertises a dataset download:
https://github.com/lakebodhi/GS-FUSE#dataset-access
https://drive.google.com/file/d/1fH436rkOHVYIG2JPROrFTXXemnY7JPdB/view?usp=sharing

The README link was verified; download access, archive size and equivalence to
the team's snapshot were NOT verified. Inspect metadata/free space before a
large download. Keep raw data private and ignored by Git. Prefer existing
available copies. Ask Adaeze only for unresolved access/location if needed;
do not make Nicole's export the default dependency.

Canonical screened manifest:
`splits/manifests/purged_split_c1_date_screen_v1.json`
(6,167 train / 2,302 validation / 2,236 test).

Canonical return snapshot content SHA-256:
`049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf`.
Its previously recorded shared path is
`/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1`;
that path is not a local Windows path or evidence of current access.

Inspect existing snapshot construction/audit code and reconstruct only if the
download contains the required sources. Audit content, event coverage and price
window separation. Do not label a different reconstruction as the immutable
snapshot or silently change the population. If provenance cannot match, mark
the new input version and its comparisons exploratory and state the blocker to
canonical C1 evaluation. Preserve an untouched source copy.

Missing intraday publication timestamps remain unresolved. Date screening
removed 883 suspect records but did not certify the retained articles as
available at 08:30/14:00. A smaller model does not fix this. Timestamp recovery
is separate work; no invented timestamps or blanket timezone shifts.

### What already works, and what does not

- `model/compact_return.py`: working compact price network and optional residual
  head accepting already-computed text vectors.
- `scripts/improvement/train_compact.py`: price training and optional vector
  residual training; `compact_features.py` exports existing frozen Llama vectors.
  These do NOT yet implement trainable raw-article encoding.
- `LOCAL_GPU_RUNBOOK.md`: optional legacy pooled-vector route, dependent on an
  export. Preserve it as an option, not the preferred independent path.
- Synthetic CUDA forward/backward and benchmark passed. No real-data compact
  accuracy experiment has run. Existing synthetic results do not validate news
  usefulness or include raw-text encoding costs.
- Existing focused checks passed before this handoff: 66 CPU tests and 25
  required text-value checks. Add meaningful checks for the new loader, masking,
  train-only vocabulary and encoder gradients; run affected checks after edits.
- Register newly runnable experiments as planned using the required skill,
  preserving existing record IDs/owners/results. Never promote designs to results.
- Storage-quota failure is only a tentative recollection; no exact failed-run
  log was supplied. Disk quota and CUDA out-of-memory require different fixes.

### Local environment, already repaired without changing the old environment

Use explicitly:

```powershell
& '.\.venv-local-gpu\Scripts\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Verified: PyTorch 2.5.1+cu124, CUDA True, RTX 4060 Laptop. This repository-local
environment bridges existing CUDA packages through a .pth file and a working
Python 3.12 runtime; keep both referenced locations intact. The original
`C:\Users\Adaeze Ilodianya\Documents\Fall 2026\.venv_gpu` contains CUDA PyTorch,
but its python launcher references a missing base Python. `.venv-campaign` is
CPU-only. Do not reinstall everything or mistake that CPU environment for the
working GPU environment. Neither local environment is tracked by Git.

Finish with exact PowerShell commands to prepare inputs, run a smoke, then run
the local comparisons. Execute what available data permits, commit source/docs
on this branch, update PR #34 when appropriate, and distinguish measured results
from pending work. No messages or compute requests to Nicole are authorized.
