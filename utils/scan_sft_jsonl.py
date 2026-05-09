#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def is_sft_sample(obj):
    if not isinstance(obj, dict):
        return False
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
    return True


def scan_file(path, min_lines=100, probe=5):
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None

    valid = 0
    total = 0
    roles = set()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total += 1
                if i < probe:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    if not is_sft_sample(obj):
                        return None
                    for m in obj["messages"]:
                        roles.add(m.get("role"))
                    valid += 1
                else:
                    valid += 1
    except OSError:
        return None

    if total < min_lines:
        return None
    return {
        "path": str(path),
        "lines": total,
        "size_mb": round(size / 1e6, 2),
        "roles": sorted(r for r in roles if r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/2025user/zhou/")
    ap.add_argument("--min-lines", type=int, default=100)
    ap.add_argument("--skip", nargs="*", default=[".git", "node_modules", "wandb", ".venv", "__pycache__"])
    args = ap.parse_args()

    root = Path(args.root)
    hits = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in args.skip]
        for fn in filenames:
            if not fn.endswith(".jsonl"):
                continue
            p = Path(dirpath) / fn
            info = scan_file(p, min_lines=args.min_lines)
            if info:
                hits.append(info)
                print(f"[OK] {info['path']}  lines={info['lines']} size={info['size_mb']}MB roles={info['roles']}", flush=True)

    hits.sort(key=lambda x: -x["lines"])
    print("\n=== Summary ===", file=sys.stderr)
    print(f"Found {len(hits)} SFT jsonl files (>= {args.min_lines} lines)", file=sys.stderr)
    print(json.dumps(hits, indent=2))


if __name__ == "__main__":
    main()
