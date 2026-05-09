#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export $(grep -v '^#' .env | xargs -d '\n' -I{} echo {} 2>/dev/null) >/dev/null 2>&1 || true
.venv/bin/python sft_trainer.py --config configs/qwen25_7b_instruct_rollout_no_state_us.yaml "$@"
