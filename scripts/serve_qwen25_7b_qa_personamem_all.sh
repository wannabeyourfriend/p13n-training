#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8200}"
GPU="${GPU:-2}"
BASE="/home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct"
ROOT="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs"

LORA_FLAGS=()
for tag in 1e-4 2e-4 5e-4; do
  D="$ROOT/qwen25_7b_instruct_qa_personamem_lastturn_lr${tag}_n1177/final"
  if [ -d "$D" ]; then
    LORA_FLAGS+=("qwen25-7b-qa-personamem-lr${tag}=$D")
  fi
done

if [ "${#LORA_FLAGS[@]}" -eq 0 ]; then
  echo "No LoRA dirs found" >&2
  exit 1
fi

export LD_LIBRARY_PATH=/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES="$GPU" /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve \
  "$BASE" \
  --served-model-name qwen25-7b-instruct-base \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "${LORA_FLAGS[@]}" \
  --max-model-len 32768 \
  --max-lora-rank 32 \
  --max-loras 3 \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string \
  "$@"
