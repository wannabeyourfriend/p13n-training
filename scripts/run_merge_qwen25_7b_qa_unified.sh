#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE="/home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct"
LORA_A="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen25_7b_instruct_qa_personamem_lastturn_lr2e-4_n1177/final"
LORA_B="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen25_7b_instruct_qa_prefeval_gen_lastturn_lr2e-4_n623/final"
OUT="/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs/qwen25_7b_instruct_qa_unified_lr2e-4_merged"

.venv/bin/python scripts/merge_qwen25_7b_qa_unified.py \
  --base "$BASE" \
  --lora_a "$LORA_A" --name_a personamem --weight_a 0.5 \
  --lora_b "$LORA_B" --name_b prefeval   --weight_b 0.5 \
  --combination_type linear \
  --out "$OUT"
