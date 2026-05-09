import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import torch
import yaml
import datasets
from datasets import Dataset
datasets.disable_caching()

from datasets.fingerprint import Hasher
_hash_counter = {"n": 0}
def _stable_hash(value):
    _hash_counter["n"] += 1
    return f"unhashable-{_hash_counter['n']:08d}"
Hasher.hash = staticmethod(_stable_hash)

_orig_map = Dataset.map
def _map_no_mp(self, *args, **kwargs):
    kwargs["num_proc"] = None
    kwargs["load_from_cache_file"] = False
    return _orig_map(self, *args, **kwargs)
Dataset.map = _map_no_mp
from transformers import TrainerCallback, TrainingArguments
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from peft import PeftModel


_orig_to_dict = TrainingArguments.to_dict


def _unredacted_to_dict(self):
    d = _orig_to_dict(self)
    for k in ("eos_token", "pad_token"):
        if k in d:
            d[k] = getattr(self, k)
    return d


TrainingArguments.to_dict = _unredacted_to_dict


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_or_json(path):
    path = Path(path)
    text = path.read_text()
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


def resolve_device_map(cfg):
    if "device_map" in cfg:
        return cfg["device_map"]
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return {"": int(local_rank)}
    return "sequential"


class TokenCountingCollator:
    def __init__(self, base):
        self.base = base
        self.last_n = 0
        self.total = 0

    def __call__(self, features):
        batch = self.base(features)
        n = int((batch["labels"] != -100).sum())
        self.last_n = n
        self.total += n
        return batch

    def __getattr__(self, name):
        return getattr(self.base, name)


class TokenLogCallback(TrainerCallback):
    def __init__(self, collator):
        self.c = collator

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["tokens/step"] = self.c.last_n
            logs["tokens/total_optimized"] = self.c.total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg["run_name"]
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "sft"))
    os.environ["WANDB_NAME"] = run_name

    full_finetuning = bool(cfg.get("full_finetuning") or cfg.get("tuning_mode") == "full")
    if full_finetuning and cfg.get("load_in_4bit", True):
        raise ValueError("full_finetuning requires load_in_4bit: false")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=cfg.get("max_seq_length", 16384),
        load_in_4bit=cfg.get("load_in_4bit", True),
        full_finetuning=full_finetuning,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", "unsloth"),
        device_map=resolve_device_map(cfg),
        dtype=None,
    )

    adapter_model = cfg.get("adapter_model")
    if adapter_model and full_finetuning:
        raise ValueError("adapter_model is only supported for LoRA/PEFT training")
    if adapter_model:
        model = PeftModel.from_pretrained(model, adapter_model, is_trainable=True)
    elif not full_finetuning:
        lora = cfg.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 16),
            lora_dropout=lora.get("dropout", 0.0),
            target_modules=lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.get("seed", 42),
        )

    if cfg.get("chat_template"):
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])

    data_path = cfg["data"]
    dataset, turn_hist = build_dataset(data_path, cfg.get("max_turns"))

    meta = {
        "run_name": run_name,
        "model": cfg["model"],
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": file_sha256(data_path),
        "num_samples": len(dataset),
        "assistant_turn_distribution": dict(sorted(turn_hist.items())),
        "max_turns": cfg.get("max_turns"),
        "adapter_model": adapter_model,
        "loss_mode": cfg.get("loss_mode", "assistant_only"),
        "tuning_mode": "full" if full_finetuning else "lora",
        "config": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        num_train_epochs=cfg.get("epochs", 1),
        learning_rate=cfg.get("lr", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        seed=cfg.get("seed", 42),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=cfg.get("optim", "adamw_8bit"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_length=cfg.get("max_seq_length", 16384),
        dataset_num_proc=1,
        eos_token=tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        report_to="wandb",
        dataset_kwargs={"skip_prepare_dataset": False},
        ddp_find_unused_parameters=cfg.get("ddp_find_unused_parameters"),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    instr = cfg.get("instruction_part", "<|im_start|>user\n")
    resp = cfg.get("response_part", "<|im_start|>assistant\n")
    loss_mode = cfg.get("loss_mode", "assistant_only")
    if loss_mode == "assistant_only":
        trainer = train_on_responses_only(
            trainer, instruction_part=instr, response_part=resp,
        )
    elif loss_mode != "full_sequence":
        raise ValueError(f"unsupported loss_mode: {loss_mode}")

    counting = TokenCountingCollator(trainer.data_collator)
    trainer.data_collator = counting
    trainer.add_callback(TokenLogCallback(counting))

    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if cfg.get("save_merged_16bit"):
        model.save_pretrained_merged(str(out_dir / "merged_16bit"), tokenizer, save_method="merged_16bit")
    if cfg.get("save_gguf"):
        model.save_pretrained_gguf(str(out_dir / "gguf"), tokenizer, quantization_method=cfg.get("gguf_quant", "q4_k_m"))

    meta["total_optimized_tokens"] = counting.total
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
