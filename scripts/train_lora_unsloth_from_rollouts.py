from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import yaml
from datasets import Dataset

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------- helpers ----------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_or_json(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def truncate_first_n_assistant(messages, n):
    if not n or n <= 0:
        return messages
    out, count = [], 0
    for m in messages:
        out.append(m)
        if m["role"] == "assistant":
            count += 1
            if count >= n:
                break
    return out


def build_dataset(data_path, max_turns):
    rows = read_jsonl_or_json(data_path)
    samples, turn_hist = [], Counter()
    for r in rows:
        msgs = r["messages"]
        msgs = truncate_first_n_assistant(msgs, max_turns)
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        if n_assist == 0:
            continue
        turn_hist[n_assist] += 1
        samples.append({"messages": msgs})
    return Dataset.from_list(samples), turn_hist


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning from rollout SFT data.")
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_name   = cfg["run_name"]
    out_dir    = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # model settings
    base_model     = cfg["model"]
    max_seq_length = cfg.get("max_seq_length", 4096)
    load_in_4bit   = cfg.get("load_in_4bit", True)
    chat_template  = cfg.get("chat_template", "llama-3")

    # LoRA settings
    lora_cfg_dict  = cfg.get("lora", {})
    lora_r         = lora_cfg_dict.get("r", 16)
    lora_alpha     = lora_cfg_dict.get("alpha", 16)
    lora_dropout   = lora_cfg_dict.get("dropout", 0.0)
    target_modules = lora_cfg_dict.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # training settings
    batch_size    = cfg.get("batch_size", 2)
    grad_accum    = cfg.get("grad_accum", 4)
    epochs        = cfg.get("epochs", 1)
    lr            = float(cfg.get("lr", 2e-4))
    warmup_ratio  = cfg.get("warmup_ratio", 0.03)
    weight_decay  = cfg.get("weight_decay", 0.0)
    logging_steps = cfg.get("logging_steps", 1)
    save_steps    = cfg.get("save_steps", 200)
    save_total    = cfg.get("save_total_limit", 3)
    seed          = cfg.get("seed", 42)
    report_to     = cfg.get("report_to", "none")
    optim         = cfg.get("optim", "adamw_8bit")
    scheduler     = cfg.get("scheduler", "cosine")
    wandb_project = cfg.get("wandb_project", "sft")

    # wandb
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", run_name)

    # load model + tokenizer via Unsloth
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect bfloat16 / float16
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # dataset
    data_path = cfg["data"]
    dataset, turn_hist = build_dataset(data_path, cfg.get("max_turns"))

    def render_batch(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for msgs in batch["messages"]
            ]
        }

    dataset = dataset.map(
        render_batch,
        batched=True,
        desc="apply_chat_template",
        load_from_cache_file=False,
    )
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "text"]
    )

    # save run metadata
    meta = {
        "run_name": run_name,
        "model": base_model,
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": file_sha256(data_path),
        "num_samples": len(dataset),
        "assistant_turn_distribution": dict(sorted(turn_hist.items())),
        "max_turns": cfg.get("max_turns"),
        "config": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # training
    from trl import SFTTrainer, SFTConfig

    steps_per_epoch = max(1, len(dataset) // (batch_size * grad_accum))
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(1, int(total_steps * warmup_ratio))

    # SFTConfig is the modern replacement for TrainingArguments with SFTTrainer.
    # completion_only_loss=True masks user/system tokens from the loss so only
    # assistant responses are trained on. This replaces the removed
    # DataCollatorForCompletionOnlyLM (dropped in TRL v1+).
    train_args = SFTConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim=optim,
        lr_scheduler_type=scheduler,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total,
        seed=seed,
        report_to=report_to,
        fp16=False,
        bf16=False,                 # Unsloth handles mixed precision internally
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=2,
        packing=False,
        completion_only_loss=True,  # assistant tokens only in loss
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset,
    )

    trainer.train()

    # save
    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Optionally save merged 16-bit weights (needed for vLLM)
    # model.save_pretrained_merged(str(out_dir / "final_merged"), tokenizer,
    #                              save_method="merged_16bit")

    # Optionally export to GGUF for Ollama
    # model.save_pretrained_gguf(str(out_dir / "gguf"), tokenizer,
    #                            quantization_method="q8_0")

    meta["trainer"] = "unsloth+SFTTrainer"
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] LoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    main()