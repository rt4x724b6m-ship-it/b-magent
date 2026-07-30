from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
VISION_DATASETS = ("mm-vet", "infographicsvqa")


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
    model_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
    for filename in model_files:
        exists = (MODEL_PATH / filename).exists()
        print(f"model/{filename}: {'OK' if exists else 'MISSING'}")
        ok = ok and exists
    weights_exist = (MODEL_PATH / "model.safetensors").exists() or (
        MODEL_PATH / "model.safetensors.index.json"
    ).exists()
    print(f"model/weights: {'OK' if weights_exist else 'MISSING'}")
    ok = ok and weights_exist

    for dataset in VISION_DATASETS:
        found = False
        for split in ("train", "test"):
            path = PROJECT_ROOT / "data" / dataset / f"{split}.jsonl"
            if not path.exists():
                continue
            found = True
            with path.open(encoding="utf-8") as handle:
                line_count = sum(1 for line in handle if line.strip())
            print(f"{dataset}/{split}.jsonl: OK ({line_count} lines)")
        if not found:
            print(f"{dataset}: MISSING (run scripts/prepare_vision_datasets.py)")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
