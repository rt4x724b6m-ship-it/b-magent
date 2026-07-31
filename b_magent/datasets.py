from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class VisionQASample:
    """One normalized sample shared by MM-Vet and InfographicsVQA."""

    question: str
    answer: str
    final_answer: str
    image_path: str
    dataset: str
    sample_id: str = ""
    answers: tuple[str, ...] = ()
    image_elements: dict[str, Any] = field(default_factory=dict)

    def to_training_text(self) -> str:
        accepted = self.answers or (self.final_answer,)
        return json.dumps(
            {
                "dataset": self.dataset,
                "id": self.sample_id,
                "image": self.image_path,
                "question": self.question,
                "answers": list(accepted),
                "image_elements": self.image_elements,
            },
            ensure_ascii=False,
        )


class VisionQADataset:
    """Reader for normalized visual-QA JSONL files and local images.

    A row must contain ``question``, ``image`` (or ``image_path``), and at
    least one answer in ``answer`` or ``answers``. Relative image paths are
    resolved against the dataset directory.
    """

    def __init__(self, root: Path, name: str | None = None) -> None:
        self.root = root
        self.name = name or root.name

    def exists(self, split: str = "train") -> bool:
        return (self.root / f"{split}.jsonl").exists()

    def load(self, split: str = "train", limit: int | None = None) -> list[VisionQASample]:
        path = self.root / f"{split}.jsonl"
        if not path.exists():
            return []
        samples: list[VisionQASample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            sample = self._normalize(payload)
            if sample is not None:
                samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def _normalize(self, payload: dict[str, Any]) -> VisionQASample | None:
        question = str(payload.get("question") or payload.get("prompt") or "").strip()
        raw_answers = payload.get("answers", payload.get("answer", payload.get("reference_answer", [])))
        if isinstance(raw_answers, (str, int, float)):
            answers = (str(raw_answers).strip(),)
        else:
            answers = tuple(str(item).strip() for item in (raw_answers or []) if str(item).strip())
        image_value = payload.get("image_path", payload.get("image", payload.get("image_name", "")))
        image_path = Path(str(image_value))
        if image_path and not image_path.is_absolute():
            image_path = self.root / image_path
        if not question or not answers or not str(image_value).strip():
            return None
        answer = answers[0]
        return VisionQASample(
            question=question,
            answer=answer,
            final_answer=answer,
            answers=answers,
            image_path=str(image_path),
            dataset=str(payload.get("dataset") or self.name),
            sample_id=str(payload.get("id", payload.get("question_id", ""))),
            image_elements=dict(payload.get("image_elements") or {}),
        )


class MultimodalBenchmarkDataset:
    """Official benchmark splits with MM-Vet excluded from training."""

    DATASETS = ("mm-vet", "infographicsvqa")
    TRAIN_DATASETS = ("infographicsvqa",)

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, split: str = "train") -> bool:
        names = self.TRAIN_DATASETS if split == "train" else self.DATASETS
        return any(VisionQADataset(self.root / name, name).exists(split) for name in names)

    def load(self, split: str = "train", limit: int | None = None) -> list[VisionQASample]:
        samples: list[VisionQASample] = []
        names = self.TRAIN_DATASETS if split == "train" else self.DATASETS
        for name in names:
            remaining = None if limit is None else max(0, limit - len(samples))
            if remaining == 0:
                break
            samples.extend(VisionQADataset(self.root / name, name).load(split, remaining))
        return samples


def load_project_dataset(root: Path) -> GSM8KDataset | VisionQADataset | MultimodalBenchmarkDataset:
    """Load the new combined benchmark, while accepting legacy GSM8K paths."""
    normalized_name = root.name.lower()
    if normalized_name in MultimodalBenchmarkDataset.DATASETS:
        return VisionQADataset(root, normalized_name)
    if normalized_name == "gsm8k" or (root / "train.jsonl").exists():
        return GSM8KDataset(root)
    return MultimodalBenchmarkDataset(root)
