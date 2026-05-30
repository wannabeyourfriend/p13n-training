# training

Multi-turn SFT trainer (Unsloth + TRL) and vLLM serving launchers for the
[Mind2Dialogue](https://github.com/wannabeyourfriend/mind2dialogue) paper.

All training logic lives in a single entry point, `train.py`. It loads a YAML
run config, fine-tunes a model with LoRA (or full fine-tuning) using
response-only loss masking, and writes a reproducible `run_meta.json` next to
the checkpoints.

## Quickstart

```bash
uv sync                                          # install dependencies
python train.py --config configs/qwen3_4b_modeB.yaml   # train
bash scripts/serve_qwen3_4b_no_think.sh          # serve via vLLM
```

Set `WANDB_API_KEY` (and any HF tokens) in `.env`.

> `sft_trainer.py` remains as a thin re-export of `train.py`, so existing
> `python sft_trainer.py --config ...` launchers continue to work unchanged.

## Layout

| | |
|---|---|
| `train.py` | single-file LoRA / full fine-tuning trainer (Unsloth backbone, TRL `SFTTrainer`, response-only loss) |
| `sft_trainer.py` | thin re-export of `train.py` for backwards compatibility |
| `configs/*.yaml` | one YAML per run (`run_name`, `model`, `data`, LoRA + optim block) |
| `scripts/train_*.sh` | per-run launchers — pin `CUDA_VISIBLE_DEVICES`, source `.env` |
| `scripts/serve_*.sh` | vLLM serving launchers (Qwen3 with / without `<think>`, GPT-OSS, …) |
| `data/*.jsonl` | training inputs (OpenAI / TRL chat schema) |
| `outputs/<run>/` | checkpoints + `run_meta.json` (data SHA-256, sample stats, full config) |

## Data

One conversation per line, OpenAI / TRL chat schema:

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Loss is computed on assistant tokens only via Unsloth's
`train_on_responses_only`, against Qwen-style `<|im_start|>user\n` /
`<|im_start|>assistant\n` markers. Override with `instruction_part` /
`response_part` for other chat templates, or set `loss_mode: full_sequence`
to train on every token.

## Config keys

```yaml
run_name: my_run            # W&B run name; checkpoints land in output_dir/run_name
model: unsloth/Qwen3-4B     # HF id or local path
data: data/train.jsonl      # .jsonl (one obj per line) or .json
output_dir: outputs

max_turns: 4                # keep only the first N assistant turns (optional)
max_seq_length: 16384       # PersonaMem-v2 needs >= 32k
load_in_4bit: true          # set false for full fine-tuning

# LoRA (omit for full fine-tuning)
lora: { r: 32, alpha: 32, dropout: 0.0 }
# full_finetuning: true     # requires load_in_4bit: false
# adapter_model: outputs/<prev>/final   # continue training an existing adapter

batch_size: 1
grad_accum: 8
epochs: 2
optim: adamw_8bit
lr: 2.0e-4
scheduler: cosine
warmup_ratio: 0.03
weight_decay: 0.0

logging_steps: 1
save_steps: 200
save_total_limit: 3
seed: 42

instruction_part: "<|im_start|>user\n"
response_part: "<|im_start|>assistant\n"
loss_mode: assistant_only   # or full_sequence

wandb_project: klab-sft

# Optional exports written after training
# save_merged_16bit: true   # merged fp16 weights (for vLLM)
# save_gguf: true           # GGUF export
# gguf_quant: q4_k_m
```

Per-step optimized-token count is logged to W&B as `tokens/step` and
`tokens/total_optimized`, and the run total is recorded in `run_meta.json` as
`total_optimized_tokens`.

## Multi-job queueing

```bash
./scripts/auto_submit_train.sh scripts/train_a.sh scripts/train_b.sh
./scripts/stop_auto_submit.sh
```

The watcher polls `nvidia-smi` and fires each launcher on the first idle GPU.
