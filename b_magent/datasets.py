from __future__ import annotations

import json
import random
import csv
from dataclasses import dataclass
from pathlib import Path

from .retrieval_training import build_retrieval_training_context


@dataclass
class GSM8KSample:
    question: str
    answer: str
    final_answer: str = ""
    task_type: str = "general"
    reference_information: str = ""
    retrieval_targets: list[str] | None = None

    def __post_init__(self) -> None:
        self.retrieval_targets = list(self.retrieval_targets or [])

    def to_training_text(self) -> str:
        if self.task_type == "general" and self.final_answer:
            return (
                "GSM8K sample | "
                f"question: {self.question} | "
                f"reasoning_answer: {self.answer} | "
                f"final_answer: {self.final_answer}"
            )
        parts = [
            f"task sample | type: {self.task_type}",
            f"task: {self.question}",
            f"reference_response: {self.answer}",
        ]
        if self.final_answer:
            parts.append(f"final_answer: {self.final_answer}")
        if self.reference_information:
            parts.append(f"candidate_reference_information: {self.reference_information}")
        if self.retrieval_targets:
            parts.append(f"relevant_sources: {'; '.join(self.retrieval_targets)}")
        return " | ".join(parts)


class GSM8KDataset:
    """Local JSONL task-dataset reader.

    Expected files:
    - data/gsm8k/train.jsonl
    - data/gsm8k/test.jsonl

    Each JSONL row should contain at least:
    - question
    - answer

    Optional ``task_type`` identifies the benchmark/domain. The canonical
    GSM8K ``#### 42`` final-answer marker remains supported for compatibility.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, split: str = "train") -> bool:
        return self._split_path(split).exists() or self._csv_split_path(split).exists()

    def load(self, split: str = "train", limit: int | None = None) -> list[GSM8KSample]:
        path = self._split_path(split)
        if not path.exists() and self._csv_split_path(split).exists():
            return self._load_csv(split, limit)
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
                    task_type=str(payload.get("task_type", "general")).strip() or "general",
                )
            )
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def _load_csv(self, split: str, limit: int | None) -> list[GSM8KSample]:
        samples: list[GSM8KSample] = []
        with self._csv_split_path(split).open(encoding="utf-8", newline="") as handle:
            for payload in csv.DictReader(handle):
                question = str(payload.get("query", "")).strip()
                answer = str(payload.get("annotated_plan", "")).strip()
                raw_reference_information = str(payload.get("reference_information", ""))
                if not question or (split == "train" and not answer):
                    continue
                reference_context, retrieval_targets = build_retrieval_training_context(
                    raw_reference_information,
                    answer,
                )
                samples.append(
                    GSM8KSample(
                        question=question,
                        answer=answer,
                        task_type=self.root.name,
                        reference_information=reference_context,
                        retrieval_targets=retrieval_targets,
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

    def _csv_split_path(self, split: str) -> Path:
        return self.root / f"{split}.csv"

    @staticmethod
    def _is_valid_row(line: str) -> bool:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return False
        return bool(str(payload.get("question", "")).strip() and str(payload.get("answer", "")).strip())
