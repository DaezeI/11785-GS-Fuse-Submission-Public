#!/bin/bash
#SBATCH --job-name=nicole-c1-return-full
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1/logs/return-full-%j.out

set -euo pipefail
umask 007
RUN_ROOT=/work/hdd/bjjg/gs-fuse/runs/nicole/return-v1
VARIANT=${VARIANT:-R3}
case "$VARIANT" in
  R0) RUN_KIND=ablation-r0-v1; VARIANT_ARGS=(--direction-weight 0 --gate-target-objective magnitude --no-learnable-last-value) ;;
  R1) RUN_KIND=ablation-r1-v1; VARIANT_ARGS=(--direction-weight 1 --gate-target-objective magnitude --no-learnable-last-value) ;;
  R2) RUN_KIND=ablation-r2-v1; VARIANT_ARGS=(--direction-weight 1 --gate-target-objective task --no-learnable-last-value) ;;
  R3) RUN_KIND=full-v1; VARIANT_ARGS=(--direction-weight 1 --gate-target-objective task --learnable-last-value) ;;
  *) echo "VARIANT must be R0, R1, R2, or R3" >&2; exit 2 ;;
esac
OUTPUT="$RUN_ROOT/$RUN_KIND"
test ! -e "$OUTPUT"
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd "$RUN_ROOT/repo"
: "${EXPECTED_CODE_COMMIT:?set the one code commit used for all variants}"
test "$(git rev-parse HEAD)" = "$EXPECTED_CODE_COMMIT"
test -z "$(git status --porcelain)"
export HF_HOME=/work/hdd/bjjg/gs-fuse/nicole/c1-nov11/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SNAPSHOT=/work/hdd/bjjg/gs-fuse/data/snapshots/c1-screened-return-v1
: "${C1_SNAPSHOT_SHA256:?set the frozen input content_sha256 from snapshot.json}"
python scripts/freeze_c1_inputs.py --verify "$SNAPSHOT" --expect-content-sha256 "$C1_SNAPSHOT_SHA256"
MANIFEST="$SNAPSHOT/manifest.json"
PRICE="$SNAPSHOT/series/SP500.csv"
test "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" = 1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1
test "$(sha256sum "$PRICE" | cut -d' ' -f1)" = 6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a
git rev-parse HEAD

python -u train.py \
  --forecast-task return_bp \
  --raw-ts-residual "${VARIANT_ARGS[@]}" \
  --c1-manifest "$MANIFEST" \
  --c1-timing-policy category4_1400_strict_before \
  --event-dir "$SNAPSHOT/event" \
  --series-dir "$SNAPSHOT/series" \
  --moment-path /work/hdd/bjjg/gs-fuse/pretrained/MOMENT-1-large \
  --text-model-path /work/hdd/bjjg/gs-fuse/pretrained/Llama-3.2-1B \
  --text-cache /work/hdd/bjjg/gs-fuse/cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 8 --sw-stride 10 \
  --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 \
  --fusion-type crossattn --gate-type softmax \
  --output-path "$OUTPUT" \
  --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
  --wandb-name "nicole-c1-return-$VARIANT-v1"
