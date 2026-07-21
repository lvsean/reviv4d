#!/bin/bash
# Train the RIGHT-hand motion VQ-VAE tokenizer on a single machine.
# (The config sets domain=lhand_joint and rhand=True.)
#
# Usage:
#     bash train_scripts/hand_tok_train_rhand.sh [NUM_GPUS]
#
# NUM_GPUS defaults to all visible GPUs. Run from the repository root
# inside the `reviv` conda environment (see environment.yaml).
set -euo pipefail

NUM_GPUS=${1:-$(python -c "import torch; print(torch.cuda.device_count())")}
CONFIG=cfgs/default/tokenization/vqvae/hand/hand_1k_60_rhand.yaml

echo "START TIME: $(date)"

NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    run_training_vqvae.py \
    --config "$CONFIG"

echo "END TIME: $(date)"
