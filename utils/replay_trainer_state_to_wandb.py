import argparse
import json
import os
from pathlib import Path

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-state", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--project", default="klab-sft")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--run-meta", default=None)
    args = ap.parse_args()

    state = json.loads(Path(args.trainer_state).read_text())
    log_history = state.get("log_history", [])
    config = {}
    if args.run_meta:
        meta = json.loads(Path(args.run_meta).read_text())
        config.update(meta.get("config", {}))
        for k in ("data_sha256", "num_samples", "assistant_turn_distribution",
                  "total_optimized_tokens"):
            if k in meta:
                config[k] = meta[k]

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        resume="never",
    )
    print(f"uploading {len(log_history)} entries to {run.url}")
    for entry in log_history:
        step = entry.get("step")
        logs = {k: v for k, v in entry.items() if k != "step"}
        wandb.log(logs, step=step)
    wandb.summary["final_loss"] = log_history[-1].get("loss") if log_history else None
    wandb.summary["total_steps"] = state.get("global_step")
    wandb.summary["epoch"] = state.get("epoch")
    wandb.finish()
    print("done")


if __name__ == "__main__":
    main()
