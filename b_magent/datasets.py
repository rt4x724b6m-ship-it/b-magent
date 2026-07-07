from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GSM8KSample:
    question: str
    answer: str
    final_answer: str

    def to_training_text(self) -> str:
        return (
            "GSM8K sample | "
            f"question: {self.question} | "
            f"reasoning_answer: {self.answer} | "
            f"final_answer: {self.final_answer}"
        )


class GSM8KDataset:
    """Local GSM8K reader.

    Expected files:
    - data/gsm8k/train.jsonl
    - data/gsm8k/test.jsonl

    Each JSONL row should contain at least:
    - question
    - answer

    The canonical GSM8K answer format often contains a final answer marker
    like "#### 42"; this reader extracts that marker when present.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, split: str = "train") -> bool:
        return self._split_path(split).exists()

    def load(self, split: str = "train", limit: int | None = None) -> list[GSM8KSample]:
        path = self._split_path(split)
        if not path.exists():
            return []

        samples: list[GSM8KSample] = []
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            if not question or not answer:
                continue
            samples.append(
                GSM8KSample(
                    question=question,
                    answer=answer,
                    final_answer=self.extract_final_answer(answer),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def split_raw_jsonl(
        self,
        source_file: Path,
        test_ratio: float = 0.2,
        seed: int = 13,
    ) -> dict[str, int]:
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if not source_file.exists():
            raise FileNotFoundError(source_file)

        rows = [
            line.strip()
            for line in source_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        valid_rows = [line for line in rows if self._is_valid_row(line)]
        if len(valid_rows) < 2:
            raise ValueError("at least two valid GSM8K rows are required to create train/test splits")

        rng = random.Random(seed)
        rng.shuffle(valid_rows)
        test_count = round(len(valid_rows) * test_ratio)
        test_count = max(1, min(test_count, len(valid_rows) - 1))

        test_rows = valid_rows[:test_count]
        train_rows = valid_rows[test_count:]
        self.root.mkdir(parents=True, exist_ok=True)
        self._split_path("train").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
        self._split_path("test").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
        return {"train": len(train_rows), "test": len(test_rows), "skipped": len(rows) - len(valid_rows)}

    @staticmethod
    def extract_final_answer(answer: str) -> str:
        marker = "####"
        if marker not in answer:
            return ""
        return answer.rsplit(marker, 1)[1].strip()

    def _split_path(self, split: str) -> Path:
        return self.root / f"{split}.jsonl"

    @staticmethod
    def _is_valid_row(line: str) -> bool:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return False
        return bool(str(payload.get("question", "")).strip() and str(payload.get("answer", "")).strip())
