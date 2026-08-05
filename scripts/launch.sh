#!/bin/bash
# Launches ats.cli.train under torchrun for single- or multi-node training.
# Usage: scripts/launch.sh --config configs/7b.yaml --use-moe --use-mla
# Env vars (all optional, with defaults for single-node single-process use):
#   NUM_NODES, GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, JOB_ID
set -e
: "${NUM_NODES:=1}"
: "${GPUS_PER_NODE:=8}"
: "${MASTER_PORT:=29500}"

torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${JOB_ID:-0}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT}" \
  -m ats.cli.train "$@"
