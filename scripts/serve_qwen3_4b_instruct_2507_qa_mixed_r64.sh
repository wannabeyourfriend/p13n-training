#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8600}"
GPU="${GPU:-1}"
BASE="/home/2025user/zhou/hf_models/Qwen3-4B-Instruct-2507"
ROOT="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs"
ADAPTER="$ROOT/qwen3_4b_instruct_2507_qa_mixed_lastturn_r64_lr2e-4_n1800/final"

if [ ! -d "$ADAPTER" ]; then
  echo "No LoRA dir found: $ADAPTER" >&2
  exit 1
fi

export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES="$GPU" /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  "$BASE" \
  --served-model-name qwen3-4b-instruct-2507-base \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "qwen3-4b-instruct-qa-mixed-r64-lr2e-4=$ADAPTER" \
  --max-model-len 32768 \
  --max-lora-rank 64 \
  --max-loras 1 \
  --gpu-memory-utilization 0.50 \
  --chat-template-content-format string \
  "$@"
