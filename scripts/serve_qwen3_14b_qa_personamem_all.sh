#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8100}"
GPU="${GPU:-1}"
BASE="/home/2025user/zhou/hf_models/Qwen3-14B"
ROOT="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs"
export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES="$GPU" /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  "$BASE" \
  --served-model-name qwen3-14b-base \
  --port "$PORT" \
  --enable-lora \
  --lora-modules \
    "qwen3-14b-qa-personamem-lr1e-4=$ROOT/qwen3_14b_qa_personamem_lastturn_lr1e-4_n1177/final" \
    "qwen3-14b-qa-personamem-lr2e-4=$ROOT/qwen3_14b_qa_personamem_lastturn_lr2e-4_n1177/final" \
    "qwen3-14b-qa-personamem-lr5e-4=$ROOT/qwen3_14b_qa_personamem_lastturn_lr5e-4_n1177/final" \
  --max-model-len 32768 \
  --max-lora-rank 32 \
  --max-loras 3 \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string \
  --override-generation-config '{"enable_thinking": false}' \
  "$@"
