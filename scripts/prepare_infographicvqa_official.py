from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


BASE_URL = (
    "https://hf-mirror.com/datasets/lmms-lab/DocVQA/resolve/main/"
    "InfographicVQA"
)
SHARD_COUNTS = {"train": 24, "validation": 4, "test": 4}


def _download(url: str, destination: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def _normalized_record(row: dict[str, Any], image_rel: Path) -> dict[str, Any]:
    record = {
        "dataset": "infographicsvqa",
        "id": str(row["questionId"]),
        "question_id": str(row["questionId"]),
        "image": image_rel.as_posix(),
        "question": row["question"],
        "answers": list(row.get("answers") or []),
        "data_split": row.get("data_split"),
        "answer_type": row.get("answer_type"),
        "image_url": row.get("image_url"),
        "operation_reasoning": row.get("operation/reasoning"),
        "ocr": row.get("ocr"),
    }
    return {key: value for key, value in record.items() if value is not None}


def prepare(output_dir: Path) -> dict[str, dict[str, int]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    with tempfile.TemporaryDirectory(prefix="infographicvqa-") as temp_name:
        temp_dir = Path(temp_name)
        for split, shard_count in SHARD_COUNTS.items():
            output_file = output_dir / f"{split}.jsonl"
            rows_written = 0
            answered_questions = 0
            images_written: set[str] = set()
            with output_file.open("w", encoding="utf-8") as handle:
                for shard_index in range(shard_count):
                    shard_name = (
                        f"{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
                    )
                    shard_path = temp_dir / shard_name
                    print(f"Downloading {shard_name}", flush=True)
                    _download(f"{BASE_URL}/{shard_name}", shard_path)
                    parquet = pq.ParquetFile(shard_path)
                    for row_group_index in range(parquet.num_row_groups):
                        for row in parquet.read_row_group(row_group_index).to_pylist():
                            image = row.get("image") or {}
                            source_name = Path(image.get("path") or f"{row['questionId']}.png").name
                            image_name = f"{split}-{source_name}"
                            image_rel = Path("images") / image_name
                            if image_name not in images_written:
                                image_bytes = image.get("bytes")
                                if not image_bytes:
                                    raise ValueError(
                                        f"missing image bytes for question {row['questionId']}"
                                    )
                                (output_dir / image_rel).write_bytes(image_bytes)
                                images_written.add(image_name)
                            handle.write(
                                json.dumps(
                                    _normalized_record(row, image_rel), ensure_ascii=False
                                )
                                + "\n"
                            )
                            rows_written += 1
                            answered_questions += bool(row.get("answers"))
                    shard_path.unlink()
            stats[split] = {
                "questions": rows_written,
                "images": len(images_written),
                "answered_questions": answered_questions,
            }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the complete official InfographicVQA splits."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
