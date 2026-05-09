#!/bin/bash
# Train one LoRA adapter per existing S3-Sim rollout dataset using the
# Unsloth/TRL trainer pattern that is closest to the original Qwen setup.
#
# Example:
#   BASE_MODEL=/path/to/Llama-3.1-8B-Instruct \
#   PY=/home/zhen/.conda/envs/sim3_env/bin/python \
#   ./scripts/run_lora_rollouts_unsloth.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/zhen/s3-sim}"
OUT_ROOT="${OUT_ROOT:-$REPO/output/checkpoints_unsloth}"
BASE_MODEL="${BASE_MODEL:-}"
PY="${PY:-/home/zhen/.conda/envs/sim3_env/bin/python}"
CONFIG_DIR="${CONFIG_DIR:-$REPO/output/lora_configs}"

if [ -z "$BASE_MODEL" ]; then
  echo "[error] set BASE_MODEL to a local Llama-3.1-8B-Instruct path or accessible HF id"
  exit 2
fi

if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python)"
fi

mkdir -p "$OUT_ROOT" "$REPO/output/logs" "$CONFIG_DIR"

write_config() {
  local run_name="$1"
  local data_file="$2"
  local config_path="$CONFIG_DIR/${run_name}.yaml"
  cat > "$config_path" <<EOF
run_name: $run_name
output_dir: $OUT_ROOT
wandb_project: sft
model: $BASE_MODEL
data: $data_file
max_turns: 12
max_seq_length: 16384
load_in_4bit: true
batch_size: 2
grad_accum: 4
epochs: 1
lr: 0.0002
warmup_ratio: 0.03
scheduler: cosine
logging_steps: 1
save_steps: 200
save_total_limit: 3
seed: 42
optim: adamw_8bit
weight_decay: 0.0
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
chat_template: "llama-3"
instruction_part: "<|start_header_id|>user<|end_header_id|>\n\n"
response_part: "<|start_header_id|>assistant<|end_header_id|>\n\n"
report_to: wandb
EOF
  printf '%s\n' "$config_path"
}

runs=(
  "full:$DATA_ROOT/rollout_gpt_4o_mini_real_world_us_1240_queires_full/sft/train_full.jsonl"
  "oracle_profile_only:$DATA_ROOT/rollout_gpt_4o_mini_real_world_us_1240_queeries_oracle_with_no_privillige/sft/train_oracle_profile_only.jsonl"
  "no_state:$DATA_ROOT/rollout_gpt_4o_mini_real_world_us_1240_queries_user_vanilla_ablated/sft/train_no_state.jsonl"
)

for spec in "${runs[@]}"; do
  name="${spec%%:*}"
  train_file="${spec#*:}"
  if [ ! -f "$train_file" ]; then
    echo "[skip] $name: missing $train_file"
    continue
  fi

  config_path="$(write_config "$name" "$train_file")"
  echo "[train] $name -> $OUT_ROOT/$name"
  "$PY" "$REPO/scripts/train_lora_unsloth_from_rollouts.py" --config "$config_path" \
    2>&1 | tee "$REPO/output/logs/${name}_lora_unsloth.log"
done

echo "[done] LoRA rollouts finished"
