from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SOURCES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "mm-vet": (("whyu/mm-vet", None),),
    "infographicsvqa": (
        ("vidore/infovqa_train", None),
        ("lmms-lab/DocVQA", "InfographicVQA"),
    ),
}


def normalized_split_name(dataset_name: str, source_split: str) -> str:
    """Preserve official splits and keep every MM-Vet sample evaluation-only."""
    normalized = source_split.lower()
    if dataset_name == "mm-vet":
        return "test"
    if normalized == "val":
        return "validation"
    return normalized


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def prepare_dataset(
    name: str,
    output_root: Path,
    limit: int | None = None,
    train_limit: int = 800,
    eval_limit: int = 100,
) -> dict[str, int]:
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' and 'Pillow' packages first.") from exc

    counts: dict[str, int] = {}
    target_dir = output_root / name
    for split in ("train", "validation", "test"):
        (target_dir / f"{split}.jsonl").unlink(missing_ok=True)
    for source, config_name in SOURCES[name]:
        available = get_dataset_split_names(source, config_name)
        for source_split in available:
            target_split = normalized_split_name(name, source_split)
            split_limit = limit if limit is not None else (train_limit if target_split == "train" else eval_limit)
            dataset = load_dataset(source, config_name, split=source_split)
            image_dir = target_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            rows: list[str] = []
            for index, item in enumerate(dataset):
                if index >= split_limit:
                    break
                question = _first(item, "question", "prompt", "query")
                answers = _first(item, "answers", "answer", "reference_answer")
                image = _first(item, "image")
                if not question or answers is None or image is None:
                    continue
                sample_id = str(_first(item, "id", "question_id") or f"{source_split}-{index}")
                image_id = str(
                    _first(item, "image_filename", "image_id", "name")
                    or sample_id
                )
                safe_image_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in image_id)
                image_rel = Path("images") / f"{target_split}-{safe_image_id}.png"
                image_path = target_dir / image_rel
                if not image_path.exists():
                    image.convert("RGB").save(image_path)
                if not isinstance(answers, list):
                    answers = [answers]
                rows.append(json.dumps({
                    "dataset": name,
                    "id": sample_id,
                    "image": str(image_rel),
                    "question": str(question),
                    "answers": [str(answer) for answer in answers],
                }, ensure_ascii=False))
            output_file = target_dir / f"{target_split}.jsonl"
            mode = "a" if output_file.exists() else "w"
            with output_file.open(mode, encoding="utf-8") as handle:
                if rows:
                    handle.write("\n".join(rows) + "\n")
            counts[target_split] = counts.get(target_split, 0) + len(rows)

    # Some InfographicVQA mirrors expose only train and validation. Keep the
    # validation split intact, but also provide it as the answer-bearing test
    # split expected by the local training and evaluation workflow.
    if name == "infographicsvqa" and counts.get("test", 0) == 0 and counts.get("validation", 0):
        validation_file = target_dir / "validation.jsonl"
        test_file = target_dir / "test.jsonl"
        test_file.write_bytes(validation_file.read_bytes())
        counts["test"] = counts["validation"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize MM-Vet and InfographicsVQA.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["all", *SOURCES], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-source-split smoke-test limit.")
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--eval-limit", type=int, default=100)
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(
            f"{name}: {prepare_dataset(name, args.output_dir, args.limit, args.train_limit, args.eval_limit)}"
        )


if __name__ == "__main__":
    main()
