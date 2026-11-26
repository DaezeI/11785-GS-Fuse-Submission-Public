# Nicole's exact 8B text arm (#55)

Status: prepared; execution awaits the exact Llama-3.1-8B weights. The frozen price checkpoints were staged on Drive on November 25. No GPU training or held-out scoring has run.

Architecture: `Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add`.
The frozen `none+CNN~none~none` was trained on `FED-PX-REL5-v1` (978 training releases).
The additive text correction trains on `FED-PXTX-ALL-REL5-v2`, excluding the two documented truncated training texts, leaving 171 releases. Validation uses all 149 eligible releases. Test remains reserved.

## Fixed pilot

- Full float32 Llama-3.1-8B, no quantization; masked mean pooling of the final hidden state, including special tokens. First 2048 tokens per release. Encoder weights never train.
- Verified `model_text` from the v2 JSONL files. Full JSONL and per-release hashes are checked. No test JSONL is opened.
- Train-only text centering/scaling. Existing frozen CNN scales and weights are reused without refitting; all three checkpoint hashes must match the published reference.
- Price CNN seeds 0, 1, 2; text-head seeds match each price seed. Only the text head trains. AdamW, learning rate .001, weight decay .001, batch 64, gradient clip 1, 30 epochs plus epoch 0. Choose the epoch by weighted 20-minute validation MAE. No hyperparameter search.
- Predict Close returns at 20 and 30 minutes from the fixed decision-time Close. Preserve the upstream timing, event splits and price audit.
- Explicit text bypass must reproduce the loaded frozen CNN exactly before and after training; verify every frozen price tensor remains unchanged. Zero features are a separate input intervention because the text head can learn biases.
- Save matched, zero-feature, and three frozen shuffled-donor forecasts (seeds 11, 23, 37). Donors must be other same-split releases, from another release instant, available by the recipient cutoff. Exclude rows with no past donor from the paired shuffled comparison and report the count.
- Unit weight per release instant; MAE and direction, plus paired 5-/10-date-block intervals. Validation is exploratory and already informed the frozen CNN selection. ALL evidence includes inferred classes. Final test evidence class and freeze decision remain with the team.

## Required artifacts

1. `compact-price-seed-{0,1,2}.pt`, `metrics.json`, `val-paired.npz` from `fed-spx-endpoints-price-v1`. Obtain the actual frozen checkpoint bytes; do not retrain substitutes. **Staged November 25:** Drive `01 Results and Evidence/fed-spx-endpoints-price-v1-frozen-cnn.zip` (228,762 bytes, SHA-256 `9b0eab4daaa3183095f14ad28d539730ac087b8acae9dfe8550e9b12bcd3358d`) holds the three checkpoints, `metrics.json`, `val-paired.npz`, `population.csv` and `SHA256SUMS.txt`. Before packaging, all three checkpoint SHA-256s matched the published `results/fed-spx-endpoints-price-v1/metrics.json`, the local `metrics.json` was byte-identical to it, and `population.csv` matched its recorded hash. `val-paired.npz` has no published hash; it came from the same run folder. Verify `SHA256SUMS.txt` after unzipping.
2. Complete local full-precision Llama-3.1-8B weights and tokenizer, with the source identity verified when staged.
3. Frozen SP500.csv and v2 train.jsonl/val.jsonl. Do not regenerate texts from today's pages.

```sh
python scripts/run_fed_spx_text.py --text-dir /path/to/verified-text \
 --price-csv /path/to/SP500.csv --frozen-dir /path/to/frozen-cnn \
 --llama-dir /path/to/Llama-3.1-8B --out-dir /path/to/fresh-output \
 --max-tokens 2048 --epochs 30 --preflight-only
```

Run preflight before reserving a GPU; remove `--preflight-only` for training after all inputs pass. The private one-H200 launcher has a three-hour cap. Account access, storage paths and scheduler receipts stay in Nicole's private handoff.

## Verification and limits

Upstream Fed integration passed 40 focused tests and validated 48 research records before PR #62 merged. The new unit tests exercise frozen-price parity after head learning and the no-future donor rule. Real-input preflight stopped at the missing CNN/8B artifacts; the CNN bundle is now staged, and the 8B weights remain missing; no end-to-end 8B smoke or result is claimed. Score with the [paired scorer](../../results/fed-spx-paired-scorer-v1/README.md) (#54): convert `val-paired.npz` with `from-text-run`, then run `score` with the packaged `none+CNN~none~none` baselines and `--text-arm`. Retain raw per-release outputs. An inference control is not an independently trained baseline.
