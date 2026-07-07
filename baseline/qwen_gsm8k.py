from __future__ import annotations

import argparse
import json
import sys
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b_magent.datasets import GSM8KDataset, GSM8KSample
from b_magent.local_qwen import LocalQwenAgentModel, LocalQwenEngine


DEFAULT_LOCAL_QWEN_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"


class QwenModel(Protocol):
    def generate(self, question: str) -> str:
        """Return a model answer for one GSM8K question."""


class EchoQwenModel:
    """Offline placeholder for the Qwen baseline interface.

    Use --echo-placeholder only for quick plumbing checks without model loading.
    """

    def generate(self, question: str) -> str:
        return f"I need to solve this GSM8K problem: {question}\n#### 0"


def build_local_qwen_baseline_model(
    model_name_or_path: str | Path = DEFAULT_LOCAL_QWEN_MODEL,
    device_map: str = "auto",
    torch_dtype: str = "float16",
) -> LocalQwenAgentModel:
    engine = LocalQwenEngine(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return LocalQwenAgentModel(agent_name="qwen_baseline", engine=engine)


@dataclass
class BaselinePrediction:
    index: int
    question: str
    gold_answer: str
    raw_prediction: str
    predicted_answer: str
    correct: bool


@dataclass
class BaselineReport:
    model_name: str
    split: str
    total: int
    correct: int
    accuracy: float
    predictions: list[BaselinePrediction]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_qwen_gsm8k_baseline(
    dataset_dir: Path,
    model: QwenModel | None = None,
    split: str = "test",
    limit: int | None = None,
    model_name: str = "qwen",
) -> BaselineReport:
    dataset = GSM8KDataset(dataset_dir)
    samples = dataset.load(split=split, limit=limit)
    if not samples:
        raise ValueError(f"no GSM8K samples found at {dataset_dir / f'{split}.jsonl'}")

    active_model = model or EchoQwenModel()
    predictions: list[BaselinePrediction] = []
    for index, sample in enumerate(samples):
        raw_prediction = active_model.generate(sample.question)
        predicted_answer = extract_numeric_answer(raw_prediction)
        gold_answer = normalize_answer(sample.final_answer)
        predictions.append(
            BaselinePrediction(
                index=index,
                question=sample.question,
                gold_answer=gold_answer,
                raw_prediction=raw_prediction,
                predicted_answer=predicted_answer,
                correct=predicted_answer == gold_answer,
            )
        )

    correct = sum(1 for prediction in predictions if prediction.correct)
    total = len(predictions)
    return BaselineReport(
        model_name=model_name,
        split=split,
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        predictions=predictions,
    )


def export_report(report: BaselineReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_numeric_answer(text: str) -> str:
    marker = "####"
    if marker in text:
        return normalize_answer(text.rsplit(marker, 1)[1])
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return ""
    return normalize_answer(matches[-1])


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    text = text.replace(",", "")
    text = text.rstrip(".")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Qwen2.5-1.5B GSM8K baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("baseline/qwen_gsm8k_report.json"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_LOCAL_QWEN_MODEL,
        help="Local path for Qwen2.5-1.5B.",
    )
    parser.add_argument(
        "--echo-placeholder",
        action="store_true",
        help="Use the offline echo placeholder instead of loading Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_model = None if args.echo_placeholder else build_local_qwen_baseline_model(args.model_path)
    report = run_qwen_gsm8k_baseline(
        dataset_dir=args.dataset_dir,
        model=active_model,
        split=args.split,
        limit=args.limit,
        model_name="echo-qwen-placeholder" if args.echo_placeholder else str(args.model_path),
    )
    export_report(report, args.output)
    print(f"split: {report.split}")
    print(f"total: {report.total}")
    print(f"correct: {report.correct}")
    print(f"accuracy: {report.accuracy:.4f}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
