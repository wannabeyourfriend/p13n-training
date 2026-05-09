import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--lora_a", required=True)
    p.add_argument("--lora_b", required=True)
    p.add_argument("--name_a", default="personamem")
    p.add_argument("--name_b", default="prefeval")
    p.add_argument("--weight_a", type=float, default=0.5)
    p.add_argument("--weight_b", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.add_argument("--combination_type", default="linear", choices=["linear", "ties", "dare_linear"])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")

    print(f"loading lora_a [{args.name_a}]: {args.lora_a}")
    model = PeftModel.from_pretrained(base, args.lora_a, adapter_name=args.name_a)

    print(f"loading lora_b [{args.name_b}]: {args.lora_b}")
    model.load_adapter(args.lora_b, adapter_name=args.name_b)

    merged_name = "merged"
    print(f"combining adapters [{args.combination_type}] weights=({args.weight_a},{args.weight_b}) -> '{merged_name}'")
    model.add_weighted_adapter(
        adapters=[args.name_a, args.name_b],
        weights=[args.weight_a, args.weight_b],
        adapter_name=merged_name,
        combination_type=args.combination_type,
    )
    model.set_adapter(merged_name)

    print("merging adapter into base weights")
    merged = model.merge_and_unload()

    print(f"saving full model to {out}")
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("done")


if __name__ == "__main__":
    main()
