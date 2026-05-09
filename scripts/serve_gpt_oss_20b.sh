#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-openai/gpt-oss-20b}"
PORT="${PORT:-8000}"
vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  "$@"
