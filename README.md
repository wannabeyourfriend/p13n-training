# training

Multi-turn SFT trainer (Unsloth + TRL) and vLLM serving launchers for the
[Mind2Dialogue](https://github.com/wannabeyourfriend/mind2dialogue) paper.

## Quickstart

```bash
uv sync                                                # install
python sft_trainer.py --config configs/qwen3_4b_modeB.yaml   # train
bash scripts/serve_qwen3_4b_no_think.sh                # serve via vLLM
```

Set `WANDB_API_KEY` (and any HF tokens) in `.env`.

## Layout

| | |
|---|---|
| `sft_trainer.py` | one-file LoRA trainer (Unsloth backbone, TRL `SFTTrainer`, response-only loss) |
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
`response_part` for other chat templates.

## Defaults

```yaml
max_seq_length: 16384       # PersonaMem-v2 needs >= 32k
load_in_4bit:   true
lora: { r: 32, alpha: 32, dropout: 0.0 }
batch_size: 1
grad_accum: 8
epochs: 2
optim: adamw_8bit
lr: 2.0e-4
scheduler: cosine
warmup_ratio: 0.03
wandb_project: klab-sft     # run name = `run_name`
```

Per-step optimized-token count is logged to W&B as `tokens/step` and
`tokens/total_optimized`.

## Multi-job queueing

```bash
./scripts/auto_submit_train.sh scripts/train_a.sh scripts/train_b.sh
./scripts/stop_auto_submit.sh
```

The watcher polls `nvidia-smi` and fires each launcher on the first
idle GPU.
