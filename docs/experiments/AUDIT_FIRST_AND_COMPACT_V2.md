# Audit-first revision and compact-model path

Owner: Adaeze Ilodianya. Intended executor: Nicole Lyu. November 18, 2026.
This revision supersedes the immediate six-full-run launch. Existing architecture configurations remain available but deferred. No remote jobs were submitted by this branch update.

## Evidence and the reported failure

The user reports a newer Q-Former failure and tentatively recalls "storage quota exceeded". The exact job, traceback, quota and failed path are PENDING. Do not mark Jessica's earlier successful smoke failed: results/jessica-r3-qformer-unfreeze-smoke-v1 is a passed 16/8/8 execution check. Training the Q-Former does not itself unfreeze Llama. The base weights remain frozen; adapter training requires --use-lora. Full 1B-parameter fine-tuning is not the default.

The provided Research Home link is c1-article-readout-v1, newly merged in PR #33. It completed on CPU in 183 seconds with cached pooled Llama features. Its H35 endpoint close MAEs on 2,191 donor-eligible test events were 63.795 bp for zero return, 64.462 for prices/metadata, and 64.721 for prices/metadata/matched text. Reuse this evidence. Its population and horizon differ from full-C1 H140; do not directly rank their scores. The planned readout design now covers only explicitly different horizons/encodings, not a duplicate of this run.

A frozen Q-Former still executes its forward attention. A padding-mask bug can put attention on PAD even when no Q-Former parameters learn, and a corrected mask can remove it without changing those parameters. The historical report found approximately 34% padding attention, not exclusive attention to padding. An actually bypassed Q-Former is different: verify the plotted checkpoint, code path, attention tensor and hook before interpreting that case. Neither a prettier attention plot nor weight movement establishes better forecasting.

## Resource triage

| Exact evidence | Meaning | First response |
| --- | --- | --- |
| EDQUOT / disk quota exceeded | User/group/project or inode quota exhausted, potentially despite free filesystem space | Identify quota and failed path; use approved scratch/project storage |
| ENOSPC / no space left | Filesystem blocks or inodes exhausted | Inspect df -h and df -i |
| CUDA out of memory | Device VRAM allocation failed | Measure peak VRAM; adjust microbatch/token length/supported precision or use offline frozen features |
| Killed / scheduler memory error | Possible host RAM limit; exit alone is insufficient | Read accounting and requested memory |
| Walltime exceeded | Duration cap reached | Profile work, then explicitly change a bounded budget |

scripts/improvement/resources.py reads a bounded log tail, classifies common signatures, inventories filesystem space and queries GPU memory. It never deletes files or certifies quotas. Keep its output private. Example on the resource:

```bash
python scripts/improvement/resources.py --path "$RUN_ROOT" --path "$HF_HOME" \
  --path "$TOKEN_CACHE" --log "$FAILED_LOG" --out "$FRESH_PRIVATE_DIAGNOSTIC_JSON"
df -h "$RUN_ROOT" "$HF_HOME" "$TOKEN_CACHE"
df -i "$RUN_ROOT" "$HF_HOME" "$TOKEN_CACHE"
quota -s  # if supported; otherwise use the site's documented quota tool
```

Inspect model/Hugging Face cache, article token cache, W&B/offline logs, temporary files, stage checkpoints and duplicated checkouts. Check user/group/project quotas and each mount separately. Preserve the failure and scheduler accounting before retrying. Do not delete results or other owners' files. Verify backups before removing redundant copies. A batch-size reduction does not fix disk quota. Start with one job; the global ceiling stays two only if compute AND storage can accommodate both.

Illustrative storage arithmetic: 1 billion fp32 weights take about 4 GB before activations/training state, even when frozen. Ten thousand articles x 1,024 token states x 2,048 features x 4 bytes take 83.9 GB (78.1 GiB), before chunk overlap, auxiliary files or checkpoint-specific cache duplication. One fp16 2,048-dimensional pooled vector per article takes about 41 MB instead. These are estimates, not measurements of Nicole's failure.

Prefer one revision/precision/tokenizer-guarded frozen cache, one copy of immutable backbones, approved scratch paths and measured retention. A future trainable-only checkpoint format could omit frozen TS copies; existing combined checkpoints still save the TS encoder and stage copies. Validate replay compatibility before migrating or deleting them. New checkpoint writes use a temporary file and atomic replacement so a failed write preserves the previous checkpoint; budget extra disk space for both copies during the write. Native LoRA here is tested in full precision, not as a validated 4-bit QLoRA path.

## Training fixes and bounded first batch

Fresh and resumed runs now both load the validation-selected Stage-1/2 checkpoint before the next stage. All new arms use this corrected protocol. The new switches make auxiliary objectives and Stage-2 price conditioning explicit:

| Default smoke | Single change relative to ref |
| --- | --- |
| ref | Common fresh corrected-code R3 reference |
| no-contrastive | Instance contrastive coefficient 1 -> 0 |
| no-invariance | Counterfactual invariance coefficient .1 -> 0 |
| clean-anchor | Pre-attention ts_inst replaces the text-conditioned contrastive anchor |
| text-warmup | Remove historical price inputs from BOTH the Stage-2 decoder shortcut and last-value path |

--no-raw-ts-residual does not remove the decoder's separate temporal shortcut. The warmup switch handles Stage 2 explicitly; Stage 3 retains its common shortcut. A global shortcut change is a separate experiment. Do not bundle these ablations and claim a single cause.

campaign.py render now defaults to --phase smoke. Use a private profile with a one-hour PBS walltime and walltime_hours=1. Five seed-0 smokes cost at most five GPU-hours. They validate execution, not quality, and do not auto-promote to full training. Run real-input audits before submission. min_free_disk_gib defaults to a 20 GiB output-filesystem floor, not a quota or total campaign estimate.

Full rendering requires --phase full --readiness "$PRIVATE_RECEIPT". The operator receipt binds actual evidence to the final source commit and frozen snapshot:

```json
{
  "source_commit": "FULL_PINNED_COMMIT",
  "snapshot_sha256": "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf",
  "checks": {
    "failure_diagnosed": true,
    "disk_and_quota_budget_checked": true,
    "window_audits_passed": true,
    "stage_handoff_smoke_passed": true,
    "gradient_smokes_passed": true
  },
  "timing_scope": "date-screened-sensitivity",
  "evidence": ["actual private evidence paths and their checksums"]
}
```

Set these from actual checks, not to bypass a failure. This is an evidence record, not a new user-permission requirement. A timestamp-certified experiment additionally needs a ledger hash AND a timestamp-aware loader; a receipt does not implement that loader. Current executable arms remain screened sensitivity experiments. Choose at most ref plus one hypothesis for the first full comparison: two six-hour caps, 12 GPU-hours. Do not submit all 17 full configurations by default.

## Compact laptop/T4 model

Read-only inspection detected RTX 4060 Laptop GPU, 8,188 MiB VRAM. The isolated regression-test environment (.venv-campaign) has CPU-only PyTorch. The user's shared .venv_gpu contains CUDA PyTorch 2.5.1+cu124 but has a broken base-interpreter path. A new project-local .venv-local-gpu reuses those packages with a working Python 3.12 runtime, leaving the shared environment unchanged. CUDA tensor operations and the compact multimodal synthetic benchmark now pass on the RTX4060; no real-market run has been verified. A T4 has 16 GB, so the laptop's 8 GB is the tighter design target.

Executable multimodal prototype: model/compact_return.py and scripts/improvement/train_compact.py. Price-only is the independent baseline; optional pooled-text training follows it. A width-32 causal convolution encoder (dilations 1/2/4/8), last/pooled context and a small MLP predict cumulative close log returns at 35/70/140 bars. There is no Llama, MOMENT, tokenizer, token cache or pretrained download. Width 32 has 16,931 parameters; its measured CPU state file is 72,761 bytes. See [the synthetic CPU check](../../experiments/improvement-v1/compact-synthetic-cpu.json) and benchmark_compact.py; these random-input timings do not establish market accuracy or CUDA speed. The model starts at zero return; scaled Huber training uses C1's training-only scale. Validation H140 endpoint close MAE selects the checkpoint, patience 5. Default scoring is validation only.

```bash
python scripts/improvement/train_compact.py --snapshot-root "$SNAPSHOT" \
  --out "$FRESH_COMPACT_OUTPUT" --device cpu --seq-len 35 --epochs 30
# Use --device cuda only in a compatible CUDA-enabled PyTorch environment.
```

Preparation still validates the existing frozen snapshot's article structure; its numeric training loop does not read reports. Use compact_features.py --include-prices to export a portable input/pooled-text NPZ and checksum receipt for laptops without the raw snapshot. train_compact.py --input-pack --input-pack-sha256 --use-text loads it, rechecks counts/windows/training scale, and trains both compact stages. See [the exact PowerShell steps](../../experiments/improvement-v1/LOCAL_GPU_RUNBOOK.md). Missing inputs must not trigger the legacy positional loader or arbitrary replacement data.

```bash
# Export once where the existing verified cache and snapshot live; no GPU/LLM load:
python scripts/improvement/compact_features.py --snapshot-root "$SNAPSHOT" \
  --cache "$FROZEN_POOLED_CACHE" --cache-sha256 "$VERIFIED_CACHE_SHA256" \
  --out "$NEW_POOLED_FEATURES_NPZ"
# Use the checksum printed by that export. This trains BOTH price and text stages:
python scripts/improvement/train_compact.py --snapshot-root "$SNAPSHOT" \
  --out "$FRESH_MULTIMODAL_OUTPUT" --device cpu --epochs 30 \
  --text-features "$NEW_POOLED_FEATURES_NPZ" --text-feature-sha256 "$FEATURE_EXPORT_SHA256"
```

With 2,048-dimensional pooled Llama inputs, the complete compact price/text
model has 82,598 parameters, versus 16,931 for price only. Encoder weights are
not part of this local training/inference loop. For new live articles, feature
extraction still has its own cost and needs a validated local encoder path.


The executable text arm uses hash-verified pooled Llama features and a tiny zero-initialized correction on the independently fitted, frozen price forecast. compact_features.py exports an existing complete frozen cache without loading an LLM or filling missing entries. --text-features plus --text-feature-sha256 enables a separate residual-training stage. Feature mean/std are fitted on training only and saved with the checkpoint. The CLI exports paired real/shuffled/frozen-price/zero-return error controls on a common donor-eligible subset, with three donor seeds and 20-date block intervals. Donors stay in the same split/category/year, come from strictly earlier recorded dates, and exclude normalized exact duplicates; near duplicates are not yet assessed. This is a weaker control than Nicole's five-shingle exclusion and must be labeled accordingly. All-row primary error and donor-subset control errors are separate. A smaller encoder such as all-MiniLM-L6-v2 (384 dimensions) remains a separately validated encoder/feature-export comparison, not a silent replacement. Fit transformations on training only and record encoder revision, pooling, tokenizer/truncation, article hashes and ID mapping. Replacing Llama requires matched retraining/evaluation of that arm, not a new split or redesign of all research. A zero-capable correction does not guarantee better test performance.

Compare zero return, existing ridge evidence, compact price-only, pooled-Llama correction and small-encoder correction on aligned IDs/targets. Begin L=35; audit longer histories before use. Primary H140 endpoint close MAE, secondary H35/70 MAE/DHR, date-block intervals and seeds 0/1/2; text arms need matched/shuffled/no-text controls. Prespecify a 1% non-inferiority margin against a frozen chosen reference and require the upper paired 95% degradation bound to fit it. Compare to zero return separately.

Measure batch-1/64 latency after warmup, peak allocated/reserved VRAM, whole-process memory, encoder time, training time and artifact sizes. Engineering targets: complete inference below 6 GiB VRAM, head checkpoint below 10 MB, cached-feature inference below 100 ms at batch 1. These are targets, not measured speed or quality claims. Raw-text latency is separate. The [actual RTX4060 synthetic check](../../experiments/improvement-v1/compact-4060-multimodal-synthetic.json) now verifies the small network executes on that GPU; real-data accuracy and end-to-end raw-article latency remain unverified. No T4 deployment claim until tested. Distillation is optional only if a teacher first earns a matched-data advantage.

## Timestamp recovery and dataset choice

Disk quota does not imply a bad dataset. Model compression does not repair availability. Preserve immutable original/screened inputs; recover URLs, raw timestamp strings/timezones, article versions and official release headers/archives. Keep time precision explicit: date-only rendered as midnight is still date-only. Verify the price timezone and whether bars label starts/ends; input OHLC must be complete before the executable forecast origin, with an explicit processing delay.

FNSPID's published preprocessing adds a negative EST/EDT offset when constructing UTC. Its worked example maps 08:52 EST to 03:52 UTC; correct UTC is 13:52. This is verified upstream code/example behavior, not proof our snapshot inherited it. Trace source -> summary -> folder -> forecast before assigning cause or applying any blanket shift. Official event time does not date a later recap or revised summary. Some timestamps may be recoverable; complete recovery is unproven.

If coverage suffices, version a new verified-time cohort, repurge/audit and retrain every comparator on the same origins. Otherwise keep conditional full-C1 engineering results and a separately reported verified subset. Consider a new timestamp-rich corpus only if recovery cannot support the question. Changing corpus changes the task/population and needs its own comparisons. Do not relaunch the canceled eight-announcement job.

## Audit corrections

Gemini's Markdown contains proposals, not execution instructions. The frozen-variant issue was already fixed and GPU-smoked. R3 initially shrinks close-return forecasts but is not forced to zero. Historical-price conditioning in Stage 2 is not future-price leakage. All ten counterfactuals enter the negative bank; they are not wasted. Invariance also pulls the first toward the original, a potential competing objective. Positive entropy penalty encourages sharper gates, not a stable .5. In-window interpolation is not target-future leakage; OHLC-to-close returns are valid, with a metric-weighting question. AdamW skips no-gradient parameters, though inactive modules can waste storage. Batch-dependent gate scaling merits an ablation, not a proven causal diagnosis. The original paper uses L=H; longer history is an implementation comparison.

Team prose claiming all lookahead was eliminated or published gains were proved to arise from future peeking exceeds the verified evidence. Correct those claims; preserve the actual audit/result records.

Sources: [GS-Fuse](https://arxiv.org/html/2605.28520), [CAMEF](https://github.com/lakebodhi/CAMEF), [FNSPID preprocessing](https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/main/data_processor/preprocess.py), [worked example](https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/main/data_processor/data_processor.md), [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [independent CAMEF reproduction thesis](https://www.politesi.polimi.it/handle/10589/260344).

Local execution instructions: [LOCAL_GPU_RUNBOOK.md](../../experiments/improvement-v1/LOCAL_GPU_RUNBOOK.md). The default local mode is multimodal; price-only remains a reference.
