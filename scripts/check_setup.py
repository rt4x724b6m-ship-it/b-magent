from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"
GSM8K_DIR = PROJECT_ROOT / "data" / "gsm8k"


def main() -> int:
    ok = True
    for package in ("torch", "transformers", "accelerate", "datasets"):
        present = importlib.util.find_spec(package) is not None
        print(f"{package}: {'OK' if present else 'MISSING'}")
        ok = ok and present

    try:
        import torch

        print(f"cuda: {'OK' if torch.cuda.is_available() else 'MISSING'}")
        print(f"cuda_device_count: {torch.cuda.device_count()}")
    except Exception as exc:
        print(f"cuda_check_error: {exc}")
        ok = False

    print(f"model_path: {MODEL_PATH}")
    model_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    for filename in model_files:
        exists = (MODEL_PATH / filename).exists()
        print(f"model/{filename}: {'OK' if exists else 'MISSING'}")
        ok = ok and exists

    for split in ("train", "test"):
        path = GSM8K_DIR / f"{split}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)
            print(f"gsm8k/{split}.jsonl: OK ({line_count} lines)")
        else:
            print(f"gsm8k/{split}.jsonl: MISSING")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
