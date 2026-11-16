# Expanse migration plan

**Dormant contingency only.** ACCESS declined the November 12 Expanse exchange
on November 16, 2026. The reviewer said a move based on long waits elsewhere
could not be accepted while all ACCESS resources were in high demand. No
Expanse account, project, storage, transfer, or model result exists. Do not use
or submit these scripts unless a future approved award makes them relevant.

For a plain-language teammate handoff, start with
[`docs/EXPANSE_APPROVAL_CHECKLIST.md`](../../docs/EXPANSE_APPROVAL_CHECKLIST.md).

The existing Delta jobs remain separate. Do not alter or cancel them without
their owner's confirmation.

## Required configuration

Set these only after Expanse supplies real values:

```sh
export EXPANSE_PROJECT=REPLACE_WITH_APPROVED_PROJECT
export REPO_ROOT=REPLACE_WITH_FIXED_EXPANSE_CHECKOUT
export RUN_ROOT=REPLACE_WITH_TEAM_READABLE_RUN_DIRECTORY
export EVENT_DIR=REPLACE_WITH_AUTHORIZED_EVENT_DIRECTORY
export SERIES_DIR=REPLACE_WITH_SERIES_DIRECTORY
export MOMENT_PATH=REPLACE_WITH_MOMENT_DIRECTORY
export TEXT_MODEL_PATH=REPLACE_WITH_LLAMA_DIRECTORY
export TEXT_CACHE=REPLACE_WITH_VERIFIED_TEXT_CACHE
export ENV_ACTIVATE=REPLACE_WITH_ENVIRONMENT_BIN_ACTIVATE
```

`RUN_ROOT` must be a fresh, team-readable owner/run directory. Keep the checkout
fixed while jobs depend on it, set group permissions, and retain `umask 007`.
The scripts derive the screened manifest from `REPO_ROOT` and verify these
committed hashes before execution:

- manifest: `1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1`
- SP500 price CSV: `6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a`

## Verified Delta source inventory — November 15

Adaeze checked the shared Delta files without launching a GPU job. The two
hashes above matched exactly. Sizes reported by Delta were:

| Item | Shared source | Size |
|---|---|---:|
| MOMENT-1-large | `/work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large` | 2.6 GB |
| Llama-3.2-1B | `/work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B` | 2.4 GB |
| Text cache | `/work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt` | 212 MB |
| SP500 CSV | `/work/hdd/bjjg/gs-fuse/data/series/SP500.csv` | 22 MB |
| Article/event tree | `/work/hdd/bjjg/gs-fuse/data/event` | **PENDING** |

The measured subtotal is about 5.2 GB. Do not claim that the requested 200 GB
is sufficient until the article tree, environment, code checkout, checkpoints,
predictions, logs, and safety margin are included. No files have been copied.

The default `PRECISION_MODE=4bit` is a memory-conservative V100 smoke setting,
not the prior A100 configuration. `PRECISION_MODE=fp32` adds `--no-4bit`.
Expanse V100 does not provide A100 TF32 behavior, so the scripts do not enable
`--tf32`. Freeze and record one successfully tested precision/cache profile
before the full run; changing it prevents a one-factor scientific comparison.

## Gates before full training

- Delta jobs `23840368`–`23840370` were verified as pending with zero elapsed
  time on November 15. They belong to Nicole's screened-C1 stream; leave them
  unchanged until she confirms whether they should continue.
- Confirm the exchange is approved and each intended user is provisioned.
- Record the real Expanse project code, group-readable paths, allocation end
  date, and storage inventory without committing usernames or account records.
- Verify environment, model/cache compatibility, and peak V100 memory in the
  bounded smoke job.
- Record the training seed limitation: current `train.py` has no seed CLI.
  Either add and test an explicit seed in a separate focused change or mark the
  resolved seed behavior as a limitation before a reportable full run.
- Estimate the baseline and paired-control GPU budget before spending from the
  current shared 100-hour request. Additional hours may be requested later, so
  record the amount actually approved rather than treating 100 as a hard cap.

## Approval-to-run order

1. On a login node, run `bash scripts/expanse/check_login_environment.sh`.
2. From a fresh `$RUN_ROOT/logs` directory, submit the ten-minute GPU check:

   ```sh
   sbatch --account="$EXPANSE_PROJECT" --export=ALL \
     "$REPO_ROOT/scripts/expanse/expanse_gpu_debug.sbatch"
   ```

3. After reviewing the GPU/environment record, submit the bounded three-stage
   smoke job. Do not proceed if it fails or exceeds V100 memory:

   ```sh
   sbatch --account="$EXPANSE_PROJECT" --export=ALL \
     "$REPO_ROOT/scripts/expanse/expanse_c1_smoke.sbatch"
   ```

4. Evaluate the smoke checkpoint on eight events by setting `CHECKPOINT`,
   `MAX_EVENTS=8`, and a distinct `PAIRED_TAG=smoke` for each mode. This is an
   execution check, not a result.
5. Only after review, submit the full training job and two dependent controls:

   ```sh
   training_job=$(sbatch --parsable --account="$EXPANSE_PROJECT" --export=ALL \
     "$REPO_ROOT/scripts/expanse/expanse_c1_full.sbatch")
   sbatch --account="$EXPANSE_PROJECT" --dependency="afterok:$training_job" \
     --export=ALL,MODE=zeroed_article_tokens \
     "$REPO_ROOT/scripts/expanse/expanse_c1_paired.sbatch"
   sbatch --account="$EXPANSE_PROJECT" --dependency="afterok:$training_job" \
     --export=ALL,MODE=fusion_bypass_ts_inst \
     "$REPO_ROOT/scripts/expanse/expanse_c1_paired.sbatch"
   ```

These commands are documentation, not executed work. Record actual job IDs,
commit, environment, hashes, resolved settings, paths, W&B run, checks, result,
limitations, and next action in a focused result note.
