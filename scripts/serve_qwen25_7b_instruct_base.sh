#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8004}"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0 /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  /home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.5 \
  "$@"
