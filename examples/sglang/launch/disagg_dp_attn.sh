#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run ingress
dynamo run in=http out=dyn --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
python3 components/worker.py \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000 &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=2,3 python3 components/decode_worker.py \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --port 31000