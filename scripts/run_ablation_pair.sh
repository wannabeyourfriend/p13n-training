#!/usr/bin/env bash
# Train the privileged and blind arms sequentially on one GPU.
# Identical recipe and item order; only the assistant targets differ.
#
#   GPU=5 bash scripts/run_ablation_pair.sh
set -uo pipefail

GPU="${GPU:-5}"
CD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CD"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for arm in privileged blind; do
  cfg="configs/abl_${arm}_r64_lr2e-4.yaml"
  log="outputs/abl_${arm}.log"
  mkdir -p outputs
  echo "[$(date +%H:%M:%S)] starting arm=$arm gpu=$GPU cfg=$cfg" | tee -a outputs/ablation_pair.log
  .venv/bin/python train.py --config "$cfg" > "$log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] arm=$arm exit=$rc" | tee -a outputs/ablation_pair.log
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ABORTING: $arm failed, see $log" | tee -a outputs/ablation_pair.log
    exit $rc
  fi
done
echo "[$(date +%H:%M:%S)] BOTH_ARMS_DONE" | tee -a outputs/ablation_pair.log
