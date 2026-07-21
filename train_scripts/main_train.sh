#!/bin/bash
# Train the main ReViV any-to-any model.
#
# Single machine:
#     bash train_scripts/main_train.sh [NUM_GPUS]
#
# Multi-node (run once per node, with the same MASTER_ADDR pointing at rank-0):
#     NNODES=4 NODE_RANK=0 MASTER_ADDR=host0 bash train_scripts/main_train.sh 4
#     NNODES=4 NODE_RANK=1 MASTER_ADDR=host0 bash train_scripts/main_train.sh 4
#     ...
#
# NUM_GPUS defaults to all visible GPUs. Run from the repository root
# inside the `reviv` conda environment (see environment.yaml).
# Note: the reference model was trained on 256 GPUs; when scaling down,
# adjust the batch size / learning rate in the config accordingly.
set -euo pipefail

NUM_GPUS=${1:-$(python -c "import torch; print(torch.cuda.device_count())")}
CONFIG=cfgs/default/reviv/models/main/reviv_base_500b_2048.yaml

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12345}

echo "START TIME: $(date)"

if [ "$NNODES" -gt 1 ]; then
    RDZV_ARGS="--nnodes $NNODES --node_rank $NODE_RANK \
        --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT"
else
    RDZV_ARGS="--standalone"
fi

NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun \
    $RDZV_ARGS \
    --nproc_per_node "$NUM_GPUS" \
    run_training_reviv.py \
    --config "$CONFIG"

echo "END TIME: $(date)"
