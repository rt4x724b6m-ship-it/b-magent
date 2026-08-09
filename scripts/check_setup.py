from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gemma-3-4b-it"


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
    model_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    for filename in model_files:
        exists = (MODEL_PATH / filename).exists()
        print(f"model/{filename}: {'OK' if exists else 'MISSING'}")
        ok = ok and exists
    weights_exist = (MODEL_PATH / "model.safetensors").exists() or (
        MODEL_PATH / "model.safetensors.index.json"
    ).exists()
    print(f"model/weights: {'OK' if weights_exist else 'MISSING'}")
    ok = ok and weights_exist

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
