from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b_magent.local_qwen import LocalQwenEngine, QwenGenerationConfig
from train.six_agent_training import infographic_anls, normalize_vision_answer


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "infographicsvqa" / "test.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "local_vlm_report.json"
DEFAULT_LIMIT = 100

SYSTEM_PROMPT = (
    "You are evaluating visual question answering. Inspect the supplied image carefully, "
    "including small text, charts, layout, objects, and spatial relationships. Answer the "
    "question using only information visible in the image. Return only the short final answer "
    "with no explanation, label, or surrounding sentence."
)


class VisionModel(Protocol):
    def generate_multimodal(self, prompt: str, image_paths: list[str | Path]) -> str: ...


@dataclass
class Prediction:
    sample_id: str
    image: str
    question: str
    gold_answers: list[str]
    raw_prediction: str
    normalized_prediction: str
    exact_match: bool
    normalized_exact_match: bool
    anls: float
    latency_seconds: float
    error: str | None = None


@dataclass
class EvaluationReport:
    model_path: str
    dataset_path: str
    total: int
    correct: int
    accuracy: float
    successful: int
    errors: int
    exact_correct: int
    normalized_correct: int
    exact_accuracy: float
    normalized_accuracy: float
    anls: float
    average_latency_seconds: float
    elapsed_seconds: float
    predictions: list[Prediction]


def clean_prediction(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:text)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:final answer|answer)\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip().strip('"\'')


def normalize_answer(value: str) -> str:
    """Use the same answer normalization as main.py's voting evaluator."""
    return normalize_vision_answer(value)


def score_anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Use the same normalized Levenshtein score as main.py's evaluator."""
    if threshold != 0.5:
        raise ValueError("main.py's ANLS evaluator uses a fixed threshold of 0.5")
    return infographic_anls(prediction, tuple(answers))


def resolve_image_path(dataset_path: Path, image_value: str) -> Path:
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = dataset_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    return image_path


def load_samples(dataset_path: Path, limit: int | None, shuffle_seed: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("question") or not row.get("image"):
                raise ValueError(f"{dataset_path}:{line_number} requires question and image")
            answers = row.get("answers", row.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                raise ValueError(f"{dataset_path}:{line_number} has no reference answer")
            row["answers"] = [str(answer) for answer in answers]
            samples.append(row)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return samples


def evaluate(
    dataset_path: Path,
    model: VisionModel,
    model_path: str,
    limit: int | None = DEFAULT_LIMIT,
    shuffle_seed: int | None = None,
) -> EvaluationReport:
    samples = load_samples(dataset_path, limit, shuffle_seed=shuffle_seed)
    predictions: list[Prediction] = []
    started = time.perf_counter()

    for index, sample in enumerate(samples, start=1):
        image_path = resolve_image_path(dataset_path, str(sample["image"]))
        prompt = f"Question: {sample['question']}"
        sample_started = time.perf_counter()
        raw_prediction = ""
        error: str | None = None
        try:
            raw_prediction = model.generate_multimodal(prompt, [image_path])
        except Exception as exc:  # Keep long evaluations resumable and auditable.
            error = f"{type(exc).__name__}: {exc}"
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        latency = time.perf_counter() - sample_started
        prediction = clean_prediction(raw_prediction)
        answers = sample["answers"]
        normalized_prediction = normalize_answer(prediction)
        normalized_answers = [normalize_answer(answer) for answer in answers]
        result = Prediction(
            sample_id=str(sample.get("id", index - 1)),
            image=str(image_path),
            question=str(sample["question"]),
            gold_answers=answers,
            raw_prediction=raw_prediction,
            normalized_prediction=normalized_prediction,
            exact_match=prediction in answers,
            normalized_exact_match=normalized_prediction in normalized_answers,
            anls=score_anls(prediction, answers),
            latency_seconds=latency,
            error=error,
        )
        predictions.append(result)
        status = "ERROR" if error else ("OK" if result.normalized_exact_match else "WRONG")
        print(
            f"[{index}/{len(samples)}] {status} id={result.sample_id} "
            f"anls={result.anls:.3f} time={latency:.2f}s pred={prediction!r}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    successful = sum(prediction.error is None for prediction in predictions)
    exact_correct = sum(prediction.exact_match for prediction in predictions)
    normalized_correct = sum(prediction.normalized_exact_match for prediction in predictions)
    total = len(predictions)
    return EvaluationReport(
        model_path=model_path,
        dataset_path=str(dataset_path.resolve()),
        total=total,
        correct=normalized_correct,
        accuracy=normalized_correct / total if total else 0.0,
        successful=successful,
        errors=total - successful,
        exact_correct=exact_correct,
        normalized_correct=normalized_correct,
        exact_accuracy=exact_correct / total if total else 0.0,
        normalized_accuracy=normalized_correct / total if total else 0.0,
        anls=sum(prediction.anls for prediction in predictions) / total if total else 0.0,
        average_latency_seconds=(
            sum(prediction.latency_seconds for prediction in predictions) / total if total else 0.0
        ),
        elapsed_seconds=elapsed,
        predictions=predictions,
    )


def export_report(report: EvaluationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_model(
    model_path: Path,
    max_new_tokens: int,
    dtype: str,
    max_pixels: int | None,
) -> LocalQwenEngine:
    model = LocalQwenEngine(
        model_name_or_path=model_path,
        torch_dtype=dtype,
        local_files_only=True,
        system_prompt=SYSTEM_PROMPT,
        generation_config=QwenGenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        ),
    )
    if max_pixels is not None:
        model.tokenizer.image_processor.max_pixels = max_pixels
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local multimodal model on image VQA JSONL.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Use 0 for the full split.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_048_576,
        help="Maximum image pixels after Qwen preprocessing; use 0 for the model default.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "auto"), default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"dataset not found: {args.dataset}")
    if not args.model_path.is_dir():
        raise SystemExit(f"local model not found: {args.model_path}")
    model = build_model(
        args.model_path,
        args.max_new_tokens,
        args.dtype,
        max_pixels=None if args.max_pixels == 0 else args.max_pixels,
    )
    try:
        report = evaluate(
            args.dataset,
            model=model,
            model_path=str(args.model_path.resolve()),
            limit=None if args.limit == 0 else args.limit,
        )
        export_report(report, args.output)
    finally:
        model.unload()
    print(
        f"\nAccuracy (main.py): {report.accuracy:.2%} "
        f"({report.correct}/{report.total})\n"
        f"Raw exact accuracy:  {report.exact_accuracy:.2%}\n"
        f"ANLS:                {report.anls:.4f}\n"
        f"Errors:              {report.errors}\n"
        f"Average latency:     {report.average_latency_seconds:.2f}s\n"
        f"Report:              {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
