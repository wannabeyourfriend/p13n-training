#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME="qwen3-14b-oracle-profile-only"
PORT="${PORT:-8002}"
GPU="${GPU:-2}"
LORA_DIR="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen3_14b_rollout-oracle-profile-only_n1240/final"
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
