"""Run the base Qwen2.5-VL-7B model on a reproducible validation sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline.local_vlm_eval import (
    DEFAULT_MODEL_PATH,
    build_model,
    evaluate,
    export_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "infographicsvqa" / "validation.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "qwen_validation_500_report.json"
DEFAULT_RANDOM_SEED = 2024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the standalone local Qwen2.5-VL-7B model on validation data."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=500, help="Number of validation samples (0 = all).")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32", "auto"), default="float16"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"validation dataset not found: {args.dataset.resolve()}")
    if not args.model_path.is_dir():
        raise SystemExit(f"local model not found: {args.model_path.resolve()}")

    model = build_model(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        max_pixels=None if args.max_pixels == 0 else args.max_pixels,
    )
    try:
        report = evaluate(
            args.dataset,
            model=model,
            model_path=str(args.model_path.resolve()),
            limit=None if args.limit == 0 else args.limit,
            shuffle_seed=args.seed,
        )
        export_report(report, args.output)
    finally:
        model.unload()

    print(
        f"\nValidation samples: {report.total}\n"
        f"Normalized accuracy: {report.normalized_accuracy:.4f}\n"
        f"ANLS: {report.anls:.4f}\n"
        f"Errors: {report.errors}\n"
        f"Report: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
