from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT, extract_numeric_answer, normalize_answer
from b_magent.datasets import GSM8KDataset, GSM8KSample
from b_magent.distillation import (
    DEFAULT_DISTILLATION_THRESHOLD,
    DistillationConfig,
    DistillationManager,
    DistillationUpdate,
)
from b_magent.local_qwen import (
    DEFAULT_QWEN_MODEL,
    LocalQwenAgentModel,
    LocalQwenEngine,
    LocalQwenEvolutionBackend,
)
from b_magent.lora import DEFAULT_LORA_THRESHOLD, LoraEvolutionManager, LoraTrainingConfig, LoraUpdate
from b_magent.seed import seed_agent_libraries
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


AGENT_NAMES = ("qwen_agent_1", "qwen_agent_2", "qwen_agent_3", "qwen_agent_4")
STANDARD_PRIVATE_TRAIN_SIZE = 200


class TrainableQwenModel(Protocol):
    def train_batch(self, batch: list[GSM8KSample]) -> None:
        """Update one agent model with one private batch."""

    def generate(self, question: str) -> str:
        """Return one answer for evaluation."""


class MemoryQwenModel:
    """Offline trainable Qwen placeholder.

    This keeps the training/evaluation loop concrete without requiring model
    setup. Replace it with a real Qwen fine-tuning or adapter-training wrapper
    later; keep the two-method interface.
    """

    def __init__(self) -> None:
        self.memory: dict[str, str] = {}

    def train_batch(self, batch: list[GSM8KSample]) -> None:
        for sample in batch:
            self.memory[sample.question] = sample.final_answer

    def generate(self, question: str) -> str:
        answer = self.memory.get(question, "0")
        return f"#### {answer}"


@dataclass
class RoundAccuracy:
    round_index: int
    trained_batches: int
    test_total: int
    test_correct: int
    accuracy: float


@dataclass
class AgentTrainingReport:
    agent_name: str
    private_train_samples: int
    rounds: list[RoundAccuracy] = field(default_factory=list)

    @property
    def final_accuracy(self) -> float:
        if not self.rounds:
            return 0.0
        return self.rounds[-1].accuracy


@dataclass
class MultiAgentTrainingReport:
    rounds: int
    batches_per_round: int
    batch_size: int
    train_total: int
    test_total: int
    agents: list[AgentTrainingReport]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for agent_payload, agent_report in zip(payload["agents"], self.agents):
            agent_payload["final_accuracy"] = agent_report.final_accuracy
        return payload


@dataclass
class AgentVote:
    agent_name: str
    raw_prediction: str
    predicted_answer: str


@dataclass
class VotingPrediction:
    index: int
    question: str
    gold_answer: str
    votes: list[AgentVote]
    final_answer: str
    correct: bool


@dataclass
class VotingReport:
    total: int
    correct: int
    accuracy: float
    predictions: list[VotingPrediction]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BMagentTrainingRound:
    round_index: int
    task: str
    participants: list[str]
    evaluators: list[str]
    drafts: int
    peer_reviews: int
    self_improvements: int
    evaluation_evolutions: int
    lora_updates: list[LoraUpdate] = field(default_factory=list)
    distillation_updates: list[DistillationUpdate] = field(default_factory=list)


@dataclass
class BMagentTrainingReport:
    dataset_dir: str
    data_dir: str
    rounds: int
    train_total: int
    agents: list[str]
    private_dataset_counts: dict[str, int]
    training_rounds: list[BMagentTrainingRound]
    professional_records: dict[str, int]
    evaluation_records: dict[str, int]
    lora_enabled: bool = False
    lora_updates: dict[str, int] = field(default_factory=dict)
    distillation_enabled: bool = False
    distillation_updates: dict[str, int] = field(default_factory=dict)
    distilled_adapter_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_b_magent_training_entry(
    dataset_dir: Path,
    data_dir: Path,
    rounds: int | None = None,
    private_batch_size: int = 1,
    random_seed: int | None = None,
    backend: object | None = None,
    lora_manager: LoraEvolutionManager | None = None,
    distillation_manager: DistillationManager | None = None,
    on_round_start: Callable[[int, int, str], None] | None = None,
    on_round_end: Callable[[int, int, BMagentTrainingRound], None] | None = None,
) -> BMagentTrainingReport:
    if rounds is not None and rounds <= 0:
        raise ValueError("rounds must be positive when explicitly set")
    if private_batch_size <= 0:
        raise ValueError("private_batch_size must be positive")

    dataset = GSM8KDataset(dataset_dir)
    train_samples = dataset.load("train")
    if not train_samples:
        raise ValueError(f"no training samples found at {dataset_dir / 'train.jsonl'}")

    private_dataset_counts = write_even_agent_private_datasets(train_samples, data_dir, AGENT_NAMES)
    base_participant_schedule = build_participant_schedule(private_dataset_counts, private_batch_size)
    effective_rounds = rounds or len(base_participant_schedule)
    participant_schedule = expand_participant_schedule(base_participant_schedule, effective_rounds)
    agents = build_default_agents(data_dir.parent, backend=backend)
    seed_agent_libraries(agents)
    professional_before = {
        agent.name: len(agent.professional_library.all_records())
        for agent in agents
    }
    evaluation_before = {
        agent.name: len(agent.evaluation_library.all_records())
        for agent in agents
    }
    workflow = MultiAgentWorkflow(agents, random_seed=random_seed, private_batch_size=private_batch_size)
    training_rounds: list[BMagentTrainingRound] = []

    for index in range(effective_rounds):
        sample = train_samples[index % len(train_samples)]
        task = format_gsm8k_training_task(sample)
        if on_round_start is not None:
            on_round_start(index + 1, effective_rounds, sample.question)
        report = workflow.run(task, participant_names=participant_schedule[index])
        lora_updates = []
        if lora_manager is not None:
            lora_updates = lora_manager.update_from_round(
                task=report.task,
                drafts=report.drafts,
                peer_reviews=report.peer_reviews,
                self_improvements=report.self_improvements,
            )
        distillation_updates = []
        if distillation_manager is not None:
            distillation_updates = distillation_manager.update_from_lora_updates(lora_updates)
        round_report = BMagentTrainingRound(
            round_index=index + 1,
            task=task,
            participants=report.participants,
            evaluators=report.evaluators,
            drafts=len(report.drafts),
            peer_reviews=len(report.peer_reviews),
            self_improvements=len(report.self_improvements),
            evaluation_evolutions=len(report.evaluation_evolutions),
            lora_updates=lora_updates,
            distillation_updates=distillation_updates,
        )
        training_rounds.append(round_report)
        if on_round_end is not None:
            on_round_end(index + 1, effective_rounds, round_report)

    return BMagentTrainingReport(
        dataset_dir=str(dataset_dir),
        data_dir=str(data_dir),
        rounds=effective_rounds,
        train_total=len(train_samples),
        agents=[agent.name for agent in agents],
        private_dataset_counts=private_dataset_counts,
        training_rounds=training_rounds,
        professional_records={
            agent.name: len(agent.professional_library.all_records()) - professional_before[agent.name]
            for agent in agents
        },
        evaluation_records={
            agent.name: len(agent.evaluation_library.all_records()) - evaluation_before[agent.name]
            for agent in agents
        },
        lora_enabled=lora_manager is not None,
        lora_updates={
            agent.name: sum(
                1
                for round_report in training_rounds
                for update in round_report.lora_updates
                if update.agent_name == agent.name and update.trained
            )
            for agent in agents
        },
        distillation_enabled=distillation_manager is not None,
        distillation_updates={
            agent.name: sum(
                1
                for round_report in training_rounds
                for update in round_report.distillation_updates
                if update.agent_name == agent.name and update.trained
            )
            for agent in agents
        },
        distilled_adapter_paths={
            agent.name: str(distillation_manager.adapter_path(agent.name))
            for agent in agents
        } if distillation_manager is not None else {},
    )


def run_four_agent_private_training(
    dataset_dir: Path,
    rounds: int = 3,
    batches_per_round: int = 32,
    batch_size: int = 1,
    private_train_size: int = STANDARD_PRIVATE_TRAIN_SIZE,
    test_limit: int | None = STANDARD_TEST_LIMIT,
    agent_names: tuple[str, ...] = AGENT_NAMES,
    model_factory: type[TrainableQwenModel] = MemoryQwenModel,
) -> MultiAgentTrainingReport:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if batches_per_round <= 0:
        raise ValueError("batches_per_round must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if private_train_size <= 0:
        raise ValueError("private_train_size must be positive")

    dataset = GSM8KDataset(dataset_dir)
    train_samples = dataset.load("train")
    test_samples = dataset.load("test", limit=test_limit)
    if not train_samples:
        raise ValueError(f"no training samples found at {dataset_dir / 'train.jsonl'}")
    if not test_samples:
        raise ValueError(f"no test samples found at {dataset_dir / 'test.jsonl'}")

    private_splits = split_private_data(train_samples, len(agent_names), private_train_size)
    agent_reports: list[AgentTrainingReport] = []

    for agent_name, private_samples in zip(agent_names, private_splits):
        model = model_factory()
        agent_report = AgentTrainingReport(
            agent_name=agent_name,
            private_train_samples=len(private_samples),
        )
        trained_batches = 0
        for round_index in range(1, rounds + 1):
            for batch_index in range(batches_per_round):
                batch = make_cyclic_batch(private_samples, batch_index, batch_size)
                model.train_batch(batch)
                trained_batches += 1
            test_correct = evaluate_model(model, test_samples)
            test_total = len(test_samples)
            agent_report.rounds.append(
                RoundAccuracy(
                    round_index=round_index,
                    trained_batches=trained_batches,
                    test_total=test_total,
                    test_correct=test_correct,
                    accuracy=test_correct / test_total,
                )
            )
        agent_reports.append(agent_report)

    return MultiAgentTrainingReport(
        rounds=rounds,
        batches_per_round=batches_per_round,
        batch_size=batch_size,
        train_total=len(train_samples),
        test_total=len(test_samples),
        agents=agent_reports,
    )


def run_four_agent_voting_on_test(
    dataset_dir: Path,
    models: dict[str, TrainableQwenModel],
    agent_names: tuple[str, ...] = AGENT_NAMES,
    limit: int | None = STANDARD_TEST_LIMIT,
    on_prediction: Callable[["VotingPrediction", int], None] | None = None,
) -> VotingReport:
    dataset = GSM8KDataset(dataset_dir)
    test_samples = dataset.load("test", limit=limit)
    if not test_samples:
        raise ValueError(f"no test samples found at {dataset_dir / 'test.jsonl'}")
    missing_agents = [agent_name for agent_name in agent_names if agent_name not in models]
    if missing_agents:
        raise ValueError(f"missing models for agents: {', '.join(missing_agents)}")

    predictions: list[VotingPrediction] = []
    for index, sample in enumerate(test_samples):
        votes: list[AgentVote] = []
        for agent_name in agent_names:
            raw_prediction = models[agent_name].generate(sample.question)
            votes.append(
                AgentVote(
                    agent_name=agent_name,
                    raw_prediction=raw_prediction,
                    predicted_answer=extract_numeric_answer(raw_prediction),
                )
            )
        final_answer = majority_vote(votes)
        gold_answer = normalize_answer(sample.final_answer)
        prediction = VotingPrediction(
            index=index,
            question=sample.question,
            gold_answer=gold_answer,
            votes=votes,
            final_answer=final_answer,
            correct=final_answer == gold_answer,
        )
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(prediction, len(test_samples))

    correct = sum(1 for prediction in predictions if prediction.correct)
    total = len(predictions)
    return VotingReport(
        total=total,
        correct=correct,
        accuracy=correct / total,
        predictions=predictions,
    )


def build_four_local_qwen_agents(
    model_name_or_path: str | Path = DEFAULT_QWEN_MODEL,
    agent_names: tuple[str, ...] = AGENT_NAMES,
    device_map: str = "auto",
    torch_dtype: str = "float16",
    lora_output_dir: str | Path | None = None,
    prefer_distilled_adapter: bool = True,
) -> dict[str, LocalQwenAgentModel]:
    engine = LocalQwenEngine(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return {
        agent_name: LocalQwenAgentModel(
            agent_name=agent_name,
            engine=engine,
            lora_output_dir=lora_output_dir,
            prefer_distilled_adapter=prefer_distilled_adapter,
        )
        for agent_name in agent_names
    }


def majority_vote(votes: list[AgentVote]) -> str:
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote.predicted_answer] = counts.get(vote.predicted_answer, 0) + 1
    best_answer = ""
    best_count = -1
    for vote in votes:
        count = counts[vote.predicted_answer]
        if count > best_count:
            best_answer = vote.predicted_answer
            best_count = count
    return best_answer


def split_private_data(
    samples: list[GSM8KSample],
    agent_count: int,
    samples_per_agent: int = STANDARD_PRIVATE_TRAIN_SIZE,
) -> list[list[GSM8KSample]]:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if samples_per_agent <= 0:
        raise ValueError("samples_per_agent must be positive")
    required_samples = agent_count * samples_per_agent
    if len(samples) < required_samples:
        raise ValueError(
            f"need at least {required_samples} training samples for "
            f"{agent_count} agents with {samples_per_agent} private samples each; "
            f"found {len(samples)}"
        )
    return [
        samples[start : start + samples_per_agent]
        for start in range(0, required_samples, samples_per_agent)
    ]


def split_samples_evenly(samples: list[GSM8KSample], agent_count: int) -> list[list[GSM8KSample]]:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    splits: list[list[GSM8KSample]] = []
    base_size, remainder = divmod(len(samples), agent_count)
    cursor = 0
    for index in range(agent_count):
        size = base_size + (1 if index < remainder else 0)
        splits.append(samples[cursor : cursor + size])
        cursor += size
    return splits


def build_participant_schedule(
    private_dataset_counts: dict[str, int],
    private_batch_size: int = 1,
) -> list[list[str]]:
    if private_batch_size <= 0:
        raise ValueError("private_batch_size must be positive")
    remaining = {
        agent_name: (private_dataset_counts.get(agent_name, 0) + private_batch_size - 1) // private_batch_size
        for agent_name in AGENT_NAMES
    }
    schedule: list[list[str]] = []
    while sum(remaining.values()) > 0:
        active = sorted(
            (item for item in remaining.items() if item[1] > 0),
            key=lambda item: (-item[1], item[0]),
        )
        first = active[0][0]
        second = active[1][0] if len(active) > 1 else next(agent_name for agent_name in AGENT_NAMES if agent_name != first)
        schedule.append([first, second])
        remaining[first] = max(0, remaining[first] - 1)
        remaining[second] = max(0, remaining[second] - 1)
    return schedule


def expand_participant_schedule(schedule: list[list[str]], rounds: int) -> list[list[str]]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not schedule:
        raise ValueError("participant schedule is empty")
    return [schedule[index % len(schedule)] for index in range(rounds)]


def write_even_agent_private_datasets(
    samples: list[GSM8KSample],
    data_dir: Path,
    agent_names: tuple[str, ...] = AGENT_NAMES,
) -> dict[str, int]:
    splits = split_samples_evenly(samples, len(agent_names))
    counts: dict[str, int] = {}
    for agent_name, private_samples in zip(agent_names, splits):
        agent_dir = data_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        private_file = agent_dir / "private_data.jsonl"
        private_file.write_text(
            "\n".join(sample.to_training_text() for sample in private_samples) + ("\n" if private_samples else ""),
            encoding="utf-8",
        )
        counts[agent_name] = len(private_samples)
    return counts


def make_cyclic_batch(samples: list[GSM8KSample], batch_index: int, batch_size: int) -> list[GSM8KSample]:
    if not samples:
        return []
    start = batch_index * batch_size
    return [samples[(start + offset) % len(samples)] for offset in range(batch_size)]


def evaluate_model(model: TrainableQwenModel, test_samples: list[GSM8KSample]) -> int:
    correct = 0
    for sample in test_samples:
        predicted = extract_numeric_answer(model.generate(sample.question))
        gold = normalize_answer(sample.final_answer)
        if predicted == gold:
            correct += 1
    return correct


def export_report(report: MultiAgentTrainingReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_json_report(report: object, output_file: Path) -> None:
    to_dict = getattr(report, "to_dict", None)
    payload = to_dict() if callable(to_dict) else asdict(report)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_gsm8k_training_task(sample: GSM8KSample) -> str:
    return (
        "Solve this GSM8K training problem and preserve reusable solving lessons.\n"
        f"Question: {sample.question}\n"
        f"Gold reasoning: {sample.answer}\n"
        f"Gold final answer: {sample.final_answer}"
    )


def format_voting_prediction_detail(prediction: VotingPrediction, total: int) -> str:
    status = "正确" if prediction.correct else "错误"
    vote_summary = ", ".join(
        f"{vote.agent_name}={vote.predicted_answer or '<empty>'}"
        for vote in prediction.votes
    )
    return (
        f"[{prediction.index + 1}/{total}] result={status} "
        f"final={prediction.final_answer or '<empty>'} "
        f"gold={prediction.gold_answer} votes=({vote_summary})"
    )


def print_voting_prediction_detail(prediction: VotingPrediction, total: int) -> None:
    print(format_voting_prediction_detail(prediction, total), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entry for b_magent agents and GSM8K runners.")
    parser.add_argument(
        "--mode",
        choices=["b-magent", "placeholder", "local-qwen-vote"],
        default="b-magent",
        help=(
            "b-magent trains the b_magent agent libraries; placeholder keeps the old "
            "offline memory baseline; local-qwen-vote runs four local Qwen voters."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["local-qwen", "demo"],
        default="local-qwen",
        help="Backend for --mode b-magent. local-qwen calls the configured local model; demo is deterministic smoke test logic.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Training rounds. Use 0 to auto-cover the evenly split private training data.",
    )
    parser.add_argument(
        "--private-batch-size",
        type=int,
        default=1,
        help="Number of private examples each participating agent loads per round.",
    )
    parser.add_argument("--batches-per-round", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--private-train-size", type=int, default=STANDARD_PRIVATE_TRAIN_SIZE)
    parser.add_argument("--test-limit", type=int, default=STANDARD_TEST_LIMIT)
    parser.add_argument(
        "--local-qwen",
        action="store_true",
        help="Deprecated alias for --mode local-qwen-vote.",
    )
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL, help="Local Qwen model directory.")
    parser.add_argument("--output", type=Path, default=Path("train/b_magent_training_report.json"))
    parser.add_argument("--seed", type=int, default=None, help="Reserved seed for reproducible b_magent runs.")
    parser.add_argument(
        "--enable-lora",
        dest="enable_lora",
        action="store_true",
        default=True,
        help="Build per-agent SFT datasets from self-reflection and train LoRA adapters when the threshold is reached. Enabled by default.",
    )
    parser.add_argument(
        "--disable-lora",
        dest="enable_lora",
        action="store_false",
        help="Disable LoRA dataset construction and adapter training.",
    )
    parser.add_argument(
        "--lora-output-dir",
        type=Path,
        default=Path("data/lora_adapters"),
        help="Directory for per-agent LoRA SFT datasets and adapters.",
    )
    parser.add_argument("--lora-threshold", type=int, default=DEFAULT_LORA_THRESHOLD)
    parser.add_argument("--lora-max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-epochs", type=float, default=1.0)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-min-evaluation-score", type=float, default=0.6)
    parser.add_argument(
        "--allow-uncorrect-lora-labels",
        action="store_true",
        help="Allow LoRA SFT examples even when a gold final answer is present and the reflected answer does not match it.",
    )
    parser.add_argument(
        "--enable-distillation",
        dest="enable_distillation",
        action="store_true",
        default=False,
        help="After per-agent LoRA updates, periodically distill trained agent adapters into private distilled adapters. Disabled by default.",
    )
    parser.add_argument(
        "--disable-distillation",
        dest="enable_distillation",
        action="store_false",
        help="Disable many-to-many LoRA distillation.",
    )
    parser.add_argument("--distillation-threshold", type=int, default=DEFAULT_DISTILLATION_THRESHOLD)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-kd-weight", type=float, default=0.5)
    parser.add_argument("--distillation-sft-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not args.enable_lora:
        args.enable_distillation = False
    return args


def build_b_magent_backend(args: argparse.Namespace) -> object | None:
    if args.backend == "demo":
        return None
    engine = LocalQwenEngine(model_name_or_path=args.model_path)
    return LocalQwenEvolutionBackend(
        engine,
        lora_output_dir=args.lora_output_dir if args.enable_lora else None,
        prefer_distilled_adapter=args.enable_distillation,
    )


def build_lora_manager(args: argparse.Namespace) -> LoraEvolutionManager | None:
    if not args.enable_lora:
        return None
    config = LoraTrainingConfig(
        base_model_path=str(args.model_path),
        output_dir=args.lora_output_dir,
        threshold=args.lora_threshold,
        require_correct_answer=not args.allow_uncorrect_lora_labels,
        min_evaluation_score=args.lora_min_evaluation_score,
        max_seq_length=args.lora_max_seq_length,
        num_train_epochs=args.lora_epochs,
        learning_rate=args.lora_learning_rate,
    )
    return LoraEvolutionManager(config)


def build_distillation_manager(args: argparse.Namespace) -> DistillationManager | None:
    if not args.enable_distillation:
        return None
    if not args.enable_lora:
        raise ValueError("--enable-distillation requires --enable-lora")
    lora_config = LoraTrainingConfig(
        base_model_path=str(args.model_path),
        output_dir=args.lora_output_dir,
        threshold=args.lora_threshold,
        require_correct_answer=not args.allow_uncorrect_lora_labels,
        min_evaluation_score=args.lora_min_evaluation_score,
        max_seq_length=args.lora_max_seq_length,
        num_train_epochs=args.lora_epochs,
        learning_rate=args.lora_learning_rate,
    )
    config = DistillationConfig.from_lora_config(
        lora_config,
        threshold=args.distillation_threshold,
        temperature=args.distillation_temperature,
        kd_weight=args.distillation_kd_weight,
        sft_weight=args.distillation_sft_weight,
    )
    return DistillationManager(config)


def print_training_round_start(round_index: int, rounds: int, question: str) -> None:
    preview = " ".join(question.split())[:120]
    print(f"[{round_index}/{rounds}] 开始四智能体训练: {preview}", flush=True)


def print_training_round_end(round_index: int, rounds: int, report: BMagentTrainingRound) -> None:
    print(
        f"[{round_index}/{rounds}] 完成: drafts={report.drafts} "
        f"evaluations={report.peer_reviews} professional_evolutions={report.self_improvements} "
        f"evaluation_evolutions={report.evaluation_evolutions}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    mode = "local-qwen-vote" if args.local_qwen else args.mode
    if mode == "b-magent":
        print(f"backend: {args.backend}", flush=True)
        print(f"model: {args.model_path}", flush=True)
        print("开始训练", flush=True)
        report = run_b_magent_training_entry(
            dataset_dir=args.dataset_dir,
            data_dir=PROJECT_ROOT / "data",
            rounds=args.rounds if args.rounds > 0 else None,
            private_batch_size=args.private_batch_size,
            random_seed=args.seed,
            backend=build_b_magent_backend(args),
            lora_manager=build_lora_manager(args),
            distillation_manager=build_distillation_manager(args),
            on_round_start=print_training_round_start,
            on_round_end=print_training_round_end,
        )
        export_json_report(report, args.output)
        print(f"b_magent agents: {', '.join(report.agents)}")
        print(f"rounds: {report.rounds}")
        for agent_name in report.agents:
            professional_count = report.professional_records[agent_name]
            evaluation_count = report.evaluation_records[agent_name]
            print(
                f"{agent_name}: professional_records={professional_count} "
                f"evaluation_records={evaluation_count} "
                f"lora_updates={report.lora_updates.get(agent_name, 0)}"
            )
        if report.distillation_enabled:
            for agent_name, adapter_path in report.distilled_adapter_paths.items():
                print(
                    f"{agent_name}: distillation_updates={report.distillation_updates.get(agent_name, 0)} "
                    f"distilled_adapter={adapter_path}"
                )
    elif mode == "local-qwen-vote":
        models = build_four_local_qwen_agents(
            args.model_path,
            lora_output_dir=args.lora_output_dir,
            prefer_distilled_adapter=args.enable_distillation,
        )
        voting_report = run_four_agent_voting_on_test(
            args.dataset_dir,
            models,
            limit=args.test_limit,
            on_prediction=print_voting_prediction_detail,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(voting_report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        accuracy_percent = voting_report.accuracy * 100
        print(f"total: {voting_report.total}")
        print(f"correct: {voting_report.correct}")
        print(
            f"accuracy={voting_report.correct}/{voting_report.total}="
            f"{voting_report.accuracy:.4f} ({accuracy_percent:.2f}%)"
        )
    elif mode == "placeholder":
        report = run_four_agent_private_training(
            dataset_dir=args.dataset_dir,
            rounds=args.rounds if args.rounds > 0 else 3,
            batches_per_round=args.batches_per_round,
            batch_size=args.batch_size,
            private_train_size=args.private_train_size,
            test_limit=args.test_limit,
        )
        export_report(report, args.output)
        for agent in report.agents:
            print(f"{agent.agent_name}: accuracy={agent.final_accuracy:.4f}")
    else:
        raise ValueError(f"unknown mode: {mode}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
