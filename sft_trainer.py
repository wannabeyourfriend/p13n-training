"""Thin re-export of the SFT trainer.

All training logic lives in ``train.py``; this module preserves backwards
compatibility for ``python sft_trainer.py --config ...`` launchers.
"""

from train import (  # noqa: F401
    TokenCountingCollator,
    build_dataset,
    file_sha256,
    load_yaml,
    main,
    read_jsonl_or_json,
    resolve_device_map,
    train,
    truncate_first_n_assistant,
)

if __name__ == "__main__":
    main()
