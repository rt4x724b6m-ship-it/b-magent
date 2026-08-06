from __future__ import annotations

import argparse
import json
from pathlib import Path

from b_magent.local_qwen import DEFAULT_QWEN_MODEL, LocalQwenEngine
from b_magent.models import LibraryRecord
from train.six_agent_training import (
    AGENT_NAMES,
    VotingPrediction,
    build_six_local_qwen_agents,
    extract_visual_answer_text,
    normalize_infographicvqa_official_answer,
    run_six_agent_voting_on_test,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "infographicsvqa"
DEFAULT_OUTPUT = PROJECT_ROOT / "train" / "six_agent_lora_infographicsvqa_validation_report.json"
DEFAULT_LORA_OUTPUT_DIR = PROJECT_ROOT / "data" / "lora_adapters_qwen2_5_vl_7b"


class SchedulerModel:
    """Use one base VLM as both the agent router and three-result aggregator."""

    def __init__(self, engine: LocalQwenEngine) -> None:
        self.engine = engine

    def generate(self, prompt: str) -> str:
        return self.engine.generate(prompt)

    def generate_multimodal(self, prompt: str, image_paths: list[str]) -> str:
        return self.engine.generate_multimodal(prompt, image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select three of six visual agents, let each inspect the image, then let the "
            "scheduler aggregate their recognized content into the final answer."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--split",
        choices=["validation"],
        default="validation",
        help="Labeled local evaluation split. Official test labels are held by the benchmark server.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / DEFAULT_QWEN_MODEL)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--lora-output-dir", type=Path, default=DEFAULT_LORA_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--professional-memory-limit", type=int, default=3)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_048_576,
        help="Maximum image pixels after Qwen preprocessing; use 0 for the model default.",
    )
    return parser.parse_args()


def load_library_records(path: Path, *, allow_empty: bool = False) -> list[LibraryRecord]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Scheduler tag library not found: {path}. Train the agents before recognition."
        )
    records: list[LibraryRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(LibraryRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid library record at {path}:{line_number}") from exc
    if not records and not allow_empty:
        raise ValueError(f"Scheduler tag library is empty: {path}")
    return records


def print_prediction(prediction: VotingPrediction, total: int) -> None:
    selected = ", ".join(prediction.selected_agents)
    status = "correct" if prediction.correct else "wrong"
    gold_answers = prediction.gold_answers or [prediction.gold_answer]
    normalized_gold_answers = {
        normalize_infographicvqa_official_answer(answer)
        for answer in gold_answers
    }
    print(
        f"[{prediction.index + 1}/{total}] {status} | selected={selected} | "
        f"final_answer={prediction.final_answer or '<empty>'} | "
        f"evaluated_answer={prediction.evaluated_answer or prediction.final_answer or '<empty>'} | "
        f"gold={', '.join(gold_answers) or '<empty>'}",
        flush=True,
    )
    for vote in prediction.votes:
        agent_status = (
            "correct"
            if normalize_infographicvqa_official_answer(
                extract_visual_answer_text(vote.raw_prediction)
            )
            in normalized_gold_answers
            else "wrong"
        )
        print(
            f"  {vote.agent_name}: {agent_status} | "
            f"answer={vote.predicted_answer or '<empty>'}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    dataset_file = args.dataset_dir / f"{args.split}.jsonl"
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset split not found: {dataset_file}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Local multimodal model not found: {args.model_path}")

    tag_records = load_library_records(
        args.data_dir / "qwen_server_agent" / "agent_training_tags.jsonl"
    )
    global_library = args.data_dir / "qwen_server_agent" / "global_evaluation_library.jsonl"
    global_records = (
        load_library_records(global_library, allow_empty=True)
        if global_library.is_file()
        else []
    )
    lora_output_dir = args.lora_output_dir
    for agent_name in AGENT_NAMES:
        adapter_config = lora_output_dir / agent_name / "adapter" / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(
                f"Required LoRA adapter for {agent_name} not found: {adapter_config}. "
                "Finish LoRA training before running this test."
            )
        professional_library = args.data_dir / agent_name / "professional_library.jsonl"
        load_library_records(professional_library)

    models = build_six_local_qwen_agents(
        model_name_or_path=args.model_path,
        agent_names=AGENT_NAMES,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        lora_output_dir=lora_output_dir,
        data_dir=args.data_dir,
        professional_memory_limit=max(1, args.professional_memory_limit),
        require_lora=True,
    )
    shared_engine = models[AGENT_NAMES[0]].engine
    if args.max_pixels > 0:
        shared_engine.tokenizer.image_processor.max_pixels = args.max_pixels
    scheduler = SchedulerModel(shared_engine)

    print(
        "Starting scheduler-routed visual recognition\n"
        f"dataset: {dataset_file}\n"
        f"candidates: {', '.join(AGENT_NAMES)}\n"
        "recognizers per image: 3",
        flush=True,
    )
    report = run_six_agent_voting_on_test(
        dataset_dir=args.dataset_dir,
        models=models,
        limit=max(1, args.limit),
        split=args.split,
        on_prediction=print_prediction,
        server_model=scheduler,
        server_training_tag_records=tag_records,
        prior_global_evaluation_records=global_records,
        enable_server_cache=False,
        official_infographicvqa_metrics=True,
    )
    for prediction in report.predictions:
        if len(prediction.selected_agents) != 3:
            raise RuntimeError(
                f"Scheduler selected {len(prediction.selected_agents)} agents for item "
                f"{prediction.index}; expected exactly 3."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed: total={report.total}, exact_accuracy={report.accuracy:.4f}, "
        f"ANLS={report.anls if report.anls is not None else 'N/A'}\n"
        f"report: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
