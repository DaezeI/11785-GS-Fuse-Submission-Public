#!/bin/bash
# Execution check for the fixed padding mask with the Q-Former (64 train / 32 eval
# events, original split). Not a benchmark. Reports qformer_mean_pad_attention_mass.
#SBATCH --job-name=jessica-smoke
#SBATCH --account=bjjg-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bjjg/zchen58/gsfuse/logs/smoke-%j.out

set -euo pipefail
umask 007
module load pytorch-conda/2.8
source /projects/bjjg/gs-fuse/env/gsfuse/bin/activate
cd /projects/bjjg/zchen58/gsfuse
export HF_HOME=/projects/bjjg/zchen58/hf-cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -u smoke_train.py \
  --moment-path pretrained/MOMENT-1-large \
  --text-model-path pretrained/Llama-3.2-1B \
  --text-cache cache/text_inst_llama1b_tf32.pt \
  --no-4bit --tf32 --batch-size 8 \
  --fusion-type qformer --gate-type softmax \
  --n-event-train 64 --n-event-eval 32 --skip-checkpoints \
  --output-path outputs/smoke/qformer
