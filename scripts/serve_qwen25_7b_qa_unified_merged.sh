#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8400}"
GPU="${GPU:-0}"
MODEL_PATH="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen25_7b_instruct_qa_unified_lr2e-4_merged"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES="$GPU" /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  "$MODEL_PATH" \
  --served-model-name qwen25-7b-qa-unified-merged \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string \
  "$@"
