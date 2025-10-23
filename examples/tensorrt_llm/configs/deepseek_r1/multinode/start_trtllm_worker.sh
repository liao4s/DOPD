#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ -z ${MODEL_PATH} ]]; then
    echo "ERROR: MODEL_PATH was not set."
    echo "ERROR: MODEL_PATH must be set to either the HuggingFace ID or locally " \
         "downloaded path to the model weights. Since Deepseek R1 is large, it is " \
         "recommended to pre-download them to a shared location and provide the path."
    exit 1
fi

if [[ -z ${SERVED_MODEL_NAME} ]]; then
    echo "WARNING: SERVED_MODEL_NAME was not set. It will be derived from MODEL_PATH."
fi



if [[ -z ${ENGINE_CONFIG} ]]; then
    echo "ERROR: ENGINE_CONFIG was not set."
    echo "ERROR: ENGINE_CONFIG must be set to a valid Dynamo+TRTLLM engine config file."
    exit 1
fi

EXTRA_ARGS=""
if [[ -n ${TASK} ]]; then
  EXTRA_ARGS+="--task ${TASK}"
fi

# NOTE: When this script is run directly from srun, the environment variables
# for TRTLLM KV cache are not set. So we need to set them here.
# Related issue: https://github.com/ai-dynamo/dynamo/issues/1743
if [[ -z ${TRTLLM_USE_UCX_KVCACHE} ]] && [[ -z ${TRTLLM_USE_NIXL_KVCACHE} ]]; then
    export TRTLLM_USE_UCX_KVCACHE=1
fi

# NOTE: trtllm_inc.py is a standalone python script that launches a Dynamo+TRTLLM
# worker and registers itself with the runtime. It is currently easier to wrap
# this standalone script with `trtllm-llmapi-launch` for MPI handling purposes,
# but this may be refactored into 'dynamo serve' in the future.
trtllm-llmapi-launch \
  python3 /workspace/launch/dynamo-run/src/subprocess/trtllm_inc.py \
    --model-path "${MODEL_PATH}" \
    --model-name "${SERVED_MODEL_NAME}" \
    --extra-engine-args "${ENGINE_CONFIG}" \
    ${EXTRA_ARGS}
