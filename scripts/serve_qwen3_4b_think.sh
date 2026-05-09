#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-unsloth/Qwen3-4B}"
PORT="${PORT:-8000}"
vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  "$@"
