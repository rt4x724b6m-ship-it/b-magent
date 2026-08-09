from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class TravelPlannerSample:
    """One TravelPlanner planning query with its annotated plan."""

    question: str
    answer: str
    final_answer: str
    level: str = ""
    org: str = ""
    dest: str = ""
    days: int = 0
    # Extended fields present in Agent-STAR-TravelDataset
    people_number: int = 1
    budget: int = 0
    local_constraint: dict[str, Any] = field(default_factory=dict)
    dates: list[str] = field(default_factory=list)
    visiting_city_number: int = 1
    sample_id: str = ""

    def to_training_text(self) -> str:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "level": self.level,
            "org": self.org,
            "dest": self.dest,
            "days": self.days,
        }
        if self.people_number != 1:
            payload["people_number"] = self.people_number
        if self.budget:
            payload["budget"] = self.budget
        if self.local_constraint:
            payload["local_constraint"] = self.local_constraint
        if self.dates:
            payload["dates"] = self.dates
        if self.visiting_city_number != 1:
            payload["visiting_city_number"] = self.visiting_city_number
        if self.sample_id:
            payload["sample_id"] = self.sample_id
        return json.dumps(payload, ensure_ascii=False)


class TravelPlannerDataset:
    """Reader for TravelPlanner JSON splits.

    Expected files:
    - data/TravelPlanner/train_train.json           (45 samples)
    - data/TravelPlanner/validation_validation.json (180 samples)
    - data/TravelPlanner/test_test.json             (1000 samples)

    Each row contains at least: query, annotated_plan, level, org, dest, days.
    """

    SPLIT_FILES: dict[str, str] = {
        "train": "train_train.json",
        "validation": "validation_validation.json",
        "test": "test_test.json",
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, split: str = "train") -> bool:
        return self._split_path(split).exists()

    def load(self, split: str = "train", limit: int | None = None) -> list[TravelPlannerSample]:
        path = self._split_path(split)
        if not path.exists():
            return []
        items = json.loads(path.read_text(encoding="utf-8"))
        samples: list[TravelPlannerSample] = []
        for item in items:
            question = str(item.get("query", "")).strip()
            # train split has annotated_plan; validation/test use reference_information
            raw_answer = item.get("annotated_plan") or item.get("reference_information", "")
            answer = (
                json.dumps(raw_answer, ensure_ascii=False)
                if not isinstance(raw_answer, str)
                else str(raw_answer).strip()
            )
            if not question:
                continue
            # test split has no reference answer; still include the question for inference
            if not answer:
                answer = "(no reference plan)"
            samples.append(
                TravelPlannerSample(
                    question=question,
                    answer=answer,
                    final_answer=answer,
                    level=str(item.get("level", "")),
                    org=str(item.get("org", "")),
                    dest=str(item.get("dest", "")),
                    days=int(item.get("days", 0) or 0),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def _split_path(self, split: str) -> Path:
        filename = self.SPLIT_FILES.get(split, f"{split}_{split}.json")
        return self.root / filename


class AgentStarTravelDataset:
    """Reader for Agent-STAR-TravelDataset JSONL splits.

    Expected directory layout::

        <root>/TravelTotal_17K.jsonl          — full synthetic training set
        <root>/Travel_Easy_1K.jsonl           — easy subset
        <root>/Travel_Medium_1K.jsonl         — medium subset
        <root>/Travel_Hard_1K.jsonl           — hard subset
        <root>/Travel_Mixed_1K_RL.jsonl       — mixed RL subset
        <root>/TravelPlanner_Val180.jsonl     — validation split (180 samples)

    Each row contains at minimum: org, dest, days, query, level.
    Optional fields: people_number, budget, local_constraint, date,
    visiting_city_number, id, data_source.
    """

    # Map logical split names to candidate file names (first found wins)
    SPLIT_CANDIDATES: dict[str, list[str]] = {
        "train": [
            "TravelTotal_17K.jsonl",
            "Travel_Mixed_1K_RL.jsonl",
            "Travel_Easy_1K.jsonl",
        ],
        "validation": ["TravelPlanner_Val180.jsonl"],
        "test": ["TravelPlanner_Val180.jsonl"],
        "easy": ["Travel_Easy_1K.jsonl"],
        "medium": ["Travel_Medium_1K.jsonl"],
        "hard": ["Travel_Hard_1K.jsonl"],
        "rl": ["Travel_Mixed_1K_RL.jsonl"],
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    def exists(self, split: str = "train") -> bool:
        return self._split_path(split) is not None

    def load(self, split: str = "train", limit: int | None = None) -> list[TravelPlannerSample]:
        path = self._split_path(split)
        if path is None:
            return []
        samples: list[TravelPlannerSample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            sample = self._normalize(item)
            if sample is not None:
                samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
        return samples

    @staticmethod
    def _normalize(item: dict[str, Any]) -> TravelPlannerSample | None:
        question = str(item.get("query", "")).strip()
        if not question:
            return None
        # Synthetic dataset has no gold annotated_plan; use empty placeholder
        answer = str(item.get("annotated_plan", item.get("reference_information", ""))).strip()
        if not answer:
            answer = "(no reference plan)"

        # local_constraint may have None values; keep as-is for the prompt
        raw_constraint = item.get("local_constraint") or {}
        local_constraint: dict[str, Any] = (
            dict(raw_constraint) if isinstance(raw_constraint, dict) else {}
        )

        dates: list[str] = []
        raw_dates = item.get("date", [])
        if isinstance(raw_dates, list):
            dates = [str(d) for d in raw_dates if d]

        return TravelPlannerSample(
            question=question,
            answer=answer,
            final_answer=answer,
            level=str(item.get("level", "")),
            org=str(item.get("org", "")),
            dest=str(item.get("dest", "")),
            days=int(item.get("days", 0) or 0),
            people_number=int(item.get("people_number", 1) or 1),
            budget=int(item.get("budget", 0) or 0),
            local_constraint=local_constraint,
            dates=dates,
            visiting_city_number=int(item.get("visiting_city_number", 1) or 1),
            sample_id=str(item.get("id", "")),
        )

    def _split_path(self, split: str) -> Path | None:
        for candidate in self.SPLIT_CANDIDATES.get(split, [f"{split}.jsonl"]):
            path = self.root / candidate
            if path.exists():
                return path
        return None


def load_project_dataset(
    root: Path,
) -> "GSM8KDataset | VisionQADataset | MultimodalBenchmarkDataset | TravelPlannerDataset | AgentStarTravelDataset":
    """Load the appropriate dataset for the given directory."""
    normalized_name = root.name.lower()
    # Agent-STAR synthetic travel dataset (JSONL format)
    if normalized_name == "agent-star-traveldataset" or (root / "TravelTotal_17K.jsonl").exists():
        return AgentStarTravelDataset(root)
    if normalized_name == "travelplanner":
        return TravelPlannerDataset(root)
    if normalized_name in MultimodalBenchmarkDataset.DATASETS:
        return VisionQADataset(root, normalized_name)
    if normalized_name == "gsm8k" or (root / "train.jsonl").exists():
        return GSM8KDataset(root)
    return MultimodalBenchmarkDataset(root)
