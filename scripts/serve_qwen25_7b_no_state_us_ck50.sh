#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8001}"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=1 /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  /home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct \
  --port "$PORT" \
  --enable-lora \
  --lora-modules no-state-us=/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen25_7b_instruct_rollout-no-state-us_n1240_bs32/checkpoint-50 \
  --max-model-len 16384 \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.75 \
  "$@"
