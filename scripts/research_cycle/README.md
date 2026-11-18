# Repeat the screened price experiment

This recipe checks whether the same model gives a similar text-control result
when trained with a different seed. The run uses the existing read-only input
snapshot and the original pinned code. Each of its four jobs uses one H200, in
sequence, on Nicole’s personal GPU resource.

Before launching, consult the private GPU skill and campaign records. Check
for existing jobs, capacity, and an already-used run ID. Seed 1 is already
submitted. To prepare another approved-in-scope seed, render a new launch
directory using the locally verified scheduler settings:

```sh
python scripts/research_cycle/launch_price.py --seed "$TRAINING_SEED" \
  --root "$PRIVATE_RESOURCE_ROOT" --project "$SCHEDULER_PROJECT" \
  --queue "$SCHEDULER_QUEUE" --resource-type "$SINGLE_H200_TYPE" \
  --python-module "$PYTHON_MODULE" --output "$NEW_LOCAL_PACKAGE"
```

Rendering does not submit jobs. Install the reviewed package as
`scripts/price-seed<N>-v1` inside the private resource project root. Run its
`submit.sh` there after verifying settings and capacity. The launcher writes
each job ID into its private chain record as the scheduler accepts it. If
submission is interrupted, inspect that chain record and continue the
remaining stages. Each subsequent job depends on the previous job succeeding.

## Collect, analyze, and update the home

The GPU operator verifies the serial smoke, full training and both paired-control
jobs finished successfully; checks the checkpoint and report checksums; and
collects the two aggregate `*.analysis.json` reports and `resolved_config.json`
into a private directory. Predictions and model weights stay in private storage.
The execution uses the frozen screened snapshot and original pinned code with
a chosen training seed and evaluation shuffle seeds 23, 24, 25. The two article-free
controls are inference interventions, not independently trained baselines.

After verification, generate the public result and a candidate registry record:

```sh
python scripts/research_cycle/export_price.py \
  --zero "$PRIVATE_EVIDENCE/zeroed_article_tokens.analysis.json" \
  --bypass "$PRIVATE_EVIDENCE/fusion_bypass_ts_inst.analysis.json" \
  --config "$PRIVATE_EVIDENCE/resolved_config.json" \
  --fingerprints "$PRIVATE_EVIDENCE/input-fingerprints.json" \
  --completion "$PRIVATE_EVIDENCE/completion.json" \
  --checkpoint-sha "$CHECKPOINT_SHA256" --seed 1 \
  --output results/screened-price-seed1-v1
```

The exporter rejects mismatched checkpoints, code commit, seed, event counts,
shuffle seeds, control definitions, missing provenance and disagreement in the
shared real/shuffled metrics. It does not independently query scheduler
accounting, hash a remote checkpoint, or certify the frozen input; those are
operator verification steps. Review all output before merging the result pack and registry update into GitHub main. The home reads the new record on reload. The exporter’s
candidate-record URLs point to the intended main-branch paths and will resolve
after publication. Intervals are conditional on a checkpoint and supplied
shuffles, not uncertainty across training seeds.

For a reproducibility test only, use `--seed 0 --historical-replay` with the
previous run's reports and config and a separate temporary output directory.
That mode is labeled historical and cannot produce a seed-1 candidate record.

For one local collection → export → registry upsert after all jobs finish, use
`collect_price.py` with an SSH alias configured outside the repository, an
absolute private resource root, the four scheduler IDs in smoke/full/zero/bypass
order, and a fresh private local directory:

```sh
python scripts/research_cycle/collect_price.py --host "$SSH_ALIAS" \
  --remote-root "$PRIVATE_RESOURCE_ROOT" \
  --jobs "$SMOKE_JOB" "$FULL_JOB" "$ZERO_JOB" "$BYPASS_JOB" \
  --seed 1 \
  --private-evidence "$PRIVATE_EVIDENCE"
```

The script checks scheduler accounting and artifact checksums on the resource,
collects only small files, exports the public pack and upserts the local
registry. SSH identity and configuration must already be set locally; no
credentials are stored by the script. `--ssh-config` selects a local SSH configuration file; `--ssh-option` may be repeated. Use `--seed N` for another positive training seed using the same price recipe (the default is 1). To test
the upsert without changing the real registry, supply `--registry` with a
copy of the registry outside the repository. Nothing is committed or published.
The fresh evidence directory remains after failures to preserve diagnostics;
use a new directory after resolving the cause.
