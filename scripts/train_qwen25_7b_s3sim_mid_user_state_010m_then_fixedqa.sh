#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export $(grep -v "^#" .env | xargs -d "\n" -I{} echo {} 2>/dev/null) >/dev/null 2>&1 || true
.venv/bin/python sft_trainer.py --config configs/qwen25_7b_s3sim_mid_user_state_010m.yaml "$@"
.venv/bin/python sft_trainer.py --config configs/qwen25_7b_s3sim_mid_user_state_010m_fixedqa.yaml "$@"
