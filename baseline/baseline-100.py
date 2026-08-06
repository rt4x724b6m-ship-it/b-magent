from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


# 支持直接执行 `python baseline/baseline-100.py`。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.local_vlm_eval import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    build_model,
)
from b_magent.datasets import VisionQADataset  # noqa: E402
from train.six_agent_training import (  # noqa: E402
    extract_prediction_answer,
    format_inference_question,
    infographic_anls,
    normalize_vision_answer,
)


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "baseline-100-report.json"


@dataclass
class BaselinePrediction:
    index: int
    sample_id: str
    dataset: str
    image: str
    question: str
    gold_answers: list[str]
    normalized_gold_answers: list[str]
    raw_prediction: str
    predicted_answer: str
    correct: bool
    anls: float | None
    latency_seconds: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original, untrained multimodal model on the image VQA test set."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=100, help="Number of test questions; 0 means all.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "auto"),
        default="float16",
    )
    return parser.parse_args()


def print_result(total: int, result: BaselinePrediction, correct_count: int) -> None:
    status = "推理错误" if result.error else ("正确" if result.correct else "错误")
    print("\n" + "=" * 88, flush=True)
    print(
        f"第 {result.index}/{total} 题 | {status} | "
        f"当前准确率: {correct_count / result.index:.2%}",
        flush=True,
    )
    print(f"样本 ID: {result.sample_id}", flush=True)
    print(f"图片: {result.image}", flush=True)
    print(f"问题: {result.question}", flush=True)
    print(f"原始标准答案: {' | '.join(result.gold_answers)}", flush=True)
    print(f"判分依据（归一化候选答案）: {' | '.join(result.normalized_gold_answers)}", flush=True)
    print(f"模型原始输出: {result.raw_prediction or '<空>'}", flush=True)
    print(f"提取后的模型答案: {result.predicted_answer or '<空>'}", flush=True)
    anls_text = "无" if result.anls is None else f"{result.anls:.3f}"
    print(f"ANLS: {anls_text} | 耗时: {result.latency_seconds:.2f} 秒", flush=True)
    if result.error:
        print(f"错误信息: {result.error}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"测试集不存在: {args.dataset.resolve()}")
    if not args.model_path.is_dir():
        raise SystemExit(f"原始模型不存在: {args.model_path.resolve()}")

    limit = None if args.limit == 0 else args.limit
    dataset = VisionQADataset(args.dataset.parent, args.dataset.parent.name)
    samples = dataset.load(args.dataset.stem, limit=limit)
    if not samples:
        raise SystemExit(f"测试集中没有有效样本: {args.dataset.resolve()}")

    print(f"原始模型: {args.model_path.resolve()}")
    print("训练适配器: 不加载（使用未训练的基础多模态模型）")
    print(f"测试集: {args.dataset.resolve()}")
    print(f"题目数: {len(samples)}")

    model = build_model(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        max_pixels=None if args.max_pixels == 0 else args.max_pixels,
    )
    predictions: list[BaselinePrediction] = []
    correct = 0
    started = time.perf_counter()

    try:
        for index, sample in enumerate(samples, start=1):
            sample_started = time.perf_counter()
            raw_prediction = ""
            error: str | None = None
            prompt = format_inference_question(sample)

            try:
                # 不传 adapter_path，确保只使用原始模型权重。
                raw_prediction = model.generate(prompt)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            predicted_answer = extract_prediction_answer(raw_prediction, sample)
            gold_answers = list(sample.answers or (sample.final_answer,))
            normalized_answers = [normalize_vision_answer(answer) for answer in gold_answers]
            result = BaselinePrediction(
                index=index,
                sample_id=sample.sample_id or str(index - 1),
                dataset=sample.dataset,
                image=sample.image_path,
                question=sample.question,
                gold_answers=gold_answers,
                normalized_gold_answers=normalized_answers,
                raw_prediction=raw_prediction,
                predicted_answer=predicted_answer,
                correct=normalize_vision_answer(predicted_answer) in normalized_answers,
                anls=(
                    infographic_anls(predicted_answer, tuple(normalized_answers))
                    if sample.dataset == "infographicsvqa"
                    else None
                ),
                latency_seconds=time.perf_counter() - sample_started,
                error=error,
            )
            predictions.append(result)
            correct += int(result.correct)
            print_result(len(samples), result, correct)

            # 每题写一次，长时间推理中断时也能保留已完成结果。
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "model_path": str(args.model_path.resolve()),
                        "adapter_loaded": False,
                        "dataset_path": str(args.dataset.resolve()),
                        "completed": len(predictions),
                        "total": len(samples),
                        "correct": correct,
                        "accuracy": correct / len(predictions),
                        "predictions": [asdict(item) for item in predictions],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        model.unload()

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 88)
    print(f"完成: {correct}/{len(predictions)} 正确，准确率 {correct / len(predictions):.2%}")
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"逐题报告: {args.output.resolve()}")


if __name__ == "__main__":
    main()
