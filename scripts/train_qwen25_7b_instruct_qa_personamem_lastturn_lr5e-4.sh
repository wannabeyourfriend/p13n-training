#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export $(grep -v '^#' .env | xargs -d '\n' -I{} echo {} 2>/dev/null) >/dev/null 2>&1 || true
.venv/bin/python sft_trainer.py --config configs/qwen25_7b_instruct_qa_personamem_lastturn_lr5e-4.yaml "$@"
