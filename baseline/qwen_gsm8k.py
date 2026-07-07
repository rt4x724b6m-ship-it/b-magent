from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import sys
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b_magent.datasets import GSM8KDataset
from b_magent.local_qwen import LocalQwenEngine


DEFAULT_LOCAL_QWEN_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"
STANDARD_TEST_LIMIT = 100


class QwenModel(Protocol):
    def generate(self, question: str) -> str:
        """Return a model answer for one GSM8K question."""


def build_local_qwen_baseline_model(
    model_name_or_path: str | Path = DEFAULT_LOCAL_QWEN_MODEL,
    device_map: str = "auto",
    torch_dtype: str = "float16",
) -> LocalQwenEngine:
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"local Qwen model path does not exist: {model_path}. "
            "This baseline is offline-only and will not download models."
        )
    return LocalQwenEngine(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        local_files_only=True,
        system_prompt=None,
    )


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
    limit: int | None = STANDARD_TEST_LIMIT,
    model_name: str = "qwen",
    on_prediction: Callable[["BaselinePrediction", int], None] | None = None,
) -> BaselineReport:
    dataset = GSM8KDataset(dataset_dir)
    samples = dataset.load(split=split, limit=limit)
    if not samples:
        raise ValueError(f"no GSM8K samples found at {dataset_dir / f'{split}.jsonl'}")

    if model is None:
        raise ValueError("run_qwen_gsm8k_baseline requires a local Qwen model instance")
    active_model = model
    predictions: list[BaselinePrediction] = []
    for index, sample in enumerate(samples):
        raw_prediction = active_model.generate(sample.question)
        predicted_answer = extract_numeric_answer(raw_prediction)
        gold_answer = normalize_answer(sample.final_answer)
        prediction = BaselinePrediction(
            index=index,
            question=sample.question,
            gold_answer=gold_answer,
            raw_prediction=raw_prediction,
            predicted_answer=predicted_answer,
            correct=predicted_answer == gold_answer,
        )
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(prediction, len(samples))

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


def format_prediction_detail(prediction: BaselinePrediction, total: int) -> str:
    status = "正确" if prediction.correct else "错误"
    return (
        f"[{prediction.index + 1}/{total}] {status} "
        f"predicted={prediction.predicted_answer or '<empty>'} "
        f"gold={prediction.gold_answer}"
    )


def print_prediction_detail(prediction: BaselinePrediction, total: int) -> None:
    print(format_prediction_detail(prediction, total), flush=True)


def extract_numeric_answer(text: str) -> str:
    marker = "####"
    if marker in text:
        return extract_last_number(text.rsplit(marker, 1)[1])
    return extract_last_number(text)


def extract_last_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return ""
    return normalize_answer(matches[-1])


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    text = text.replace(",", "")
    text = text.rstrip(".")
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return text
    if numeric == numeric.to_integral_value():
        return str(numeric.quantize(Decimal(1)))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Qwen2.5-1.5B GSM8K baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument(
        "--limit",
        type=int,
        default=STANDARD_TEST_LIMIT,
        help="Number of official GSM8K test samples to evaluate.",
    )
    parser.add_argument("--output", type=Path, default=Path("baseline/qwen_gsm8k_report.json"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_LOCAL_QWEN_MODEL,
        help="Local path for Qwen2.5-1.5B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_model = build_local_qwen_baseline_model(args.model_path)
    report = run_qwen_gsm8k_baseline(
        dataset_dir=args.dataset_dir,
        model=active_model,
        split=args.split,
        limit=args.limit,
        model_name=str(args.model_path),
        on_prediction=print_prediction_detail,
    )
    print(f"split: {report.split}")
    print(f"total: {report.total}")
    print(f"correct: {report.correct}")
    print(f"accuracy: {report.accuracy:.4f}")
    export_report(report, args.output)
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
