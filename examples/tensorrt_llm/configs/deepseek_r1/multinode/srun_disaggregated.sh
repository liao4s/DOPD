#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is one of the only variables that must be set currently, most of the rest may
# just work out of the box if following the steps in the README.
IMAGE="${IMAGE:-""}"

# Set to mount current host directory to /mnt inside the container as an example,
# but you may freely customize the mounts based on your cluster. A common practice
# is to mount paths to NFS storage for common scripts, model weights, etc.
# NOTE: This can be a comma separated list of multiple mounts as well.
DEFAULT_MOUNT="${PWD}:/mnt"
MOUNTS="${MOUNTS:-${DEFAULT_MOUNT}}"

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}

NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-4}
PREFILL_ENGINE_CONFIG="${PREFILL_ENGINE_CONFIG:-/mnt/engine_configs/wide_ep_prefill.yaml}"

NUM_DECODE_NODES=${NUM_DECODE_NODES:-4}
DECODE_ENGINE_CONFIG="${DECODE_ENGINE_CONFIG:-/mnt/engine_configs/wide_ep_decode.yaml}"

# Automate settings of certain variables for convenience, but you are free
# to manually set these for more control as well.
ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
export HEAD_NODE="${SLURMD_NODENAME}"
export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

if [[ -z ${IMAGE} ]]; then
  echo "ERROR: You need to set the IMAGE environment variable to the " \
       "Dynamo+TRTLLM docker image or .sqsh file from 'enroot import' " \
       "See how to build one from source here: " \
       "https://github.com/ai-dynamo/dynamo/tree/main/examples/tensorrt_llm#build-docker"
  exit 1
fi

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
echo "Launching frontend services in background."
srun \
  --overlap \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/start_frontend_services.sh &

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
echo "Launching multi-node prefill worker in background."
TASK=prefill \
ENGINE_CONFIG=${PREFILL_ENGINE_CONFIG} \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE,TASK,ENGINE_CONFIG \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodes "${NUM_PREFILL_NODES}" \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/start_trtllm_worker.sh &

echo "Launching multi-node decode worker in background."
TASK=decode \
ENGINE_CONFIG=${DECODE_ENGINE_CONFIG} \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE,TASK,ENGINE_CONFIG \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodes "${NUM_DECODE_NODES}" \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/start_trtllm_worker.sh &
