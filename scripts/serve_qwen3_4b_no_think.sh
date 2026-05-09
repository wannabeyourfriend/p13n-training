#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-/home/2025user/zhou/hf_models/Qwen3-4B}"
PORT="${PORT:-8000}"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
/home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --chat-template-content-format string \
  --override-generation-config '{"enable_thinking": false}' \
  "$@"
