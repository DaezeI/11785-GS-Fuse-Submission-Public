# Expanse approval checklist for teammates

**Status: dormant contingency only.** ACCESS declined the November 12 Expanse
exchange on November 16, 2026. No Expanse account, storage, file copy, or job
resulted. Do not follow this checklist unless the team later receives a real
approved award and confirms that this plan is relevant to it.

The reviewer said a move based on long waits elsewhere could not be accepted
while all ACCESS resources were in high demand. Do not resubmit the same request
without a materially different, provider-appropriate reason and team agreement.

## What not to change

- Do not cancel or alter Delta jobs `23840368`, `23840369`, or `23840370`.
  They belong to Nicole's screened-C1 work and were still waiting with zero
  running time when checked on November 15.
- Do not use the old unscreened C1 event list for a new full run.
- Do not describe the screened event list as publication-time certified.
  Publication times for the retained articles are still unknown.
- Do not start full training before the short Expanse checks below succeed.
- Do not commit usernames, passwords, verification codes, or private account
  records to GitHub.

## Step 1: confirm that approval and personal access are real

1. In ACCESS, confirm that the exchange says **Approved**.
2. Confirm that Expanse GPU and storage appear as active resources, not pending
   balances.
3. Confirm that your own Expanse account opens successfully.
4. Connect to Expanse and run:

   ```sh
   expanse-client user -r expanse_gpu
   ```

5. Record the approved project name, hours, storage, and end date in a GitHub
   handoff. Do not record private login information.

These checks do not use GPU hours.

## Step 2: use the prepared branch

Use branch `adaeze/expanse-migration`. It contains:

- the current project and experiment status;
- the verified Delta source locations and file fingerprints;
- a login/environment check;
- a ten-minute GPU check;
- small practice-training instructions;
- full screened-C1 training instructions; and
- the two required no-article comparisons.

Read [`scripts/expanse/README.md`](../scripts/expanse/README.md) before using
the files. Replace every marked placeholder with the real approved project and
storage locations. Never guess these values.

## Step 3: confirm storage before copying

The following Delta sources were verified on November 15:

| Item | Delta source | Reported size |
|---|---|---:|
| MOMENT-1-large | `/work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large` | 2.6 GB |
| Llama-3.2-1B | `/work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B` | 2.4 GB |
| Saved text information | `/work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt` | 212 MB |
| SP500 prices | `/work/hdd/bjjg/gs-fuse/data/series/SP500.csv` | 22 MB |
| Articles/events | `/work/hdd/bjjg/gs-fuse/data/event` | **Size still unknown** |

The measured subtotal is about 5.2 GB. Before copying, include the articles,
software environment, saved models, predictions, logs, and extra safety space
in the storage estimate. The current request is for 200 GB, but it has not yet
been proven sufficient.

## Step 4: copy and verify the inputs

Copy only after approval, personal access, project storage, and permissions are
confirmed. Keep the raw Delta inputs unchanged.

After copying, verify these two fingerprints:

```text
Screened event list:
1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1

SP500 price file:
6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
```

Matching fingerprints mean those copied files are identical to the verified
Delta sources. Stop if either one differs.

Copying files does not use GPU hours, but it does use the Expanse storage
allocation.

## Step 5: perform the free account check

Run:

```sh
bash scripts/expanse/check_login_environment.sh
```

This confirms the account, project, and available software information. It
does not start a GPU or train the model. Save the output in the run handoff.

## Step 6: perform the ten-minute GPU check

Fill in the approved project and software-setup values described in
[`scripts/expanse/README.md`](../scripts/expanse/README.md), then use
`expanse_gpu_debug.sbatch`.

This check should use at most about 0.17 GPU-hours. It confirms that the GPU is
visible and that the project software can use it. Stop if it fails.

## Step 7: run the small practice experiment

Use `expanse_c1_smoke.sbatch`. It trains on only 16 events and checks eight.
It is a practice run, not a scientific result.

Before continuing, record:

- whether all three training stages finished;
- how long the job took;
- the highest GPU memory use;
- the saved-model location;
- the exact code version and settings; and
- every warning or error.

Do not start full training until the team reviews this record. Expanse uses a
different GPU from Delta, so a memory-saving or number-format change creates a
different experiment setup that must be documented.

## Step 8: run the full baseline and matched comparisons

After the practice run is approved:

1. Run the full screened-C1 price baseline with `expanse_c1_full.sbatch`.
2. After it succeeds, run `expanse_c1_paired.sbatch` once with zeroed article
   information and once with the article-fusion bypass.
3. Each comparison also evaluates correct articles and three fixed shuffled
   article assignments on the same examples.
4. Keep the return redesign, gate, MoE, and Q-Former experiments separate until
   this reference result is complete, unless the team records a new priority.

The declined request was for 100 GPU-hours. A future request or award may be
different; record only the amount actually approved and used.

## Required handoff after every run

Record these facts in GitHub:

- owner and purpose;
- status: planned, waiting, running, completed, or failed;
- code version and exact command;
- Expanse project and software setup, excluding private login details;
- event-list and price-file fingerprints;
- model and saved-text versions;
- training settings and whether the run used a fixed random starting setting;
- job number, time used, saved-model location, and output location;
- checks performed and their results;
- measured findings, including negative or failed findings;
- limitations and the next action.

Use **PENDING** for anything unknown. A waiting or practice job is not a final
result.
