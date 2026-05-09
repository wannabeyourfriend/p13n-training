#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME="qwen3-14b-us-profile-mar31"
PORT="${PORT:-8003}"
GPU="${GPU:-3}"
LORA_DIR="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen3_14b_rollout-us-profile-mar31_n1240/final"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES="$GPU" /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  /home/2025user/zhou/hf_models/Qwen3-14B \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "${MODEL_NAME}=${LORA_DIR}" \
  --max-model-len 16384 \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string \
  --override-generation-config '{"enable_thinking": false}' \
  "$@"
