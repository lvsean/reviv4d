#!/bin/bash
# Train the body motion VQ-VAE tokenizer on a single machine.
#
# Usage:
#     bash train_scripts/body_tok_train.sh [NUM_GPUS]
#
# NUM_GPUS defaults to all visible GPUs. Run from the repository root
# inside the `reviv` conda environment (see environment.yaml).
#
# Default config = the released 21-joint body tokenizer (reviv_tok_body.pth).
# To train the experimental 60-joint fullbody tokenizer instead, point CONFIG at
# cfgs/default/tokenization/vqvae/body/fullbody_motion_60_2048_2_3.yaml.
set -euo pipefail

NUM_GPUS=${1:-$(python -c "import torch; print(torch.cuda.device_count())")}
CONFIG=cfgs/default/tokenization/vqvae/body/body_motion_60_2048_2_3.yaml

echo "START TIME: $(date)"

NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    run_training_vqvae.py \
    --config "$CONFIG"

echo "END TIME: $(date)"
