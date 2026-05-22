#!/bin/bash
# FSDP Distributed Training Launch Script
#
# Usage:
#   bash scripts/run_fsdp.sh -c config/LLIE_TinyRestormer_LOLv2_Ubuntu_Baseline.json -p train
#
# Environment variables:
#   NPROC_PER_NODE  - Number of GPUs per node (default: auto-detect via nvidia-smi)
#   MASTER_PORT     - Master port for distributed communication (default: 29500)
#   NNODES          - Number of nodes (default: 1)
#   NODE_RANK       - Rank of this node (default: 0)
#   MASTER_ADDR     - Master address for multi-node training (default: localhost)

set -e

# NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}

# if [ "$NPROC_PER_NODE" -eq 0 ]; then
#     echo "Error: No GPUs detected. Please set NPROC_PER_NODE manually."
#     exit 1
# fi

NPROC_PER_NODE=3

echo "Launching FSDP training with ${NPROC_PER_NODE} GPU(s)..."

CUDA_VISIBLE_DEVICES=0,1,3 torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=${NNODES:-1} \
    --node_rank=${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29500} \
    main.py "$@"
