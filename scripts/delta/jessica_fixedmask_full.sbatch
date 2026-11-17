#!/bin/bash
# Q-Former / baseline rerun with the fixed text padding mask (PR #2, #5).
# Historical diagnosis on the ORIGINAL positional split (not C1), matched to
# Joye's midterm runs: Llama-3.2-1B fp32 + TF32, batch 8, 2/2/2 epochs, stride 1.
# Submit from the checkout root:
#   sbatch --job-name=jessica-qformer  --export=ALL,VARIANT=qformer  scripts/delta/jessica_fixedmask_full.sbatch
#   sbatch --job-name=jessica-baseline --export=ALL,VARIANT=baseline scripts/delta/jessica_fixedmask_full.sbatch
#SBATCH --job-name=jessica-fixedmask
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bjjg/zchen58/gsfuse/logs/full-%x-%j.out

set -euo pipefail
umask 007
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /projects/bjjg/zchen58/gsfuse
export HF_HOME=/projects/bjjg/zchen58/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# data/event, data/series, data/cache_llama_emb, pretrained/ and cache/ are
# symlinks to /projects/bjjg/gs-fuse/repo/GS-FUSE (Joye's midterm inputs).
case "${VARIANT:?set VARIANT}" in
  baseline) flags="--fusion-type crossattn --gate-type softmax" ;;
  qformer)  flags="--fusion-type qformer --gate-type softmax" ;;
  *) echo "unknown VARIANT: $VARIANT"; exit 1 ;;
esac

python -u train.py \
  --moment-path pretrained/MOMENT-1-large \
  --text-model-path pretrained/Llama-3.2-1B \
  --text-cache cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 8 \
  $flags \
  --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 \
  --sw-stride 1 --interpret-samples 200 \
  --output-path outputs/fixedmask/$VARIANT \
  --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse \
  --wandb-name jessica-historical-fixedmask-$VARIANT
