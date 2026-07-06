from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from baseline.qwen_gsm8k import extract_numeric_answer, normalize_answer
from b_magent.datasets import GSM8KDataset, GSM8KSample


AGENT_NAMES = ("qwen_planner", "qwen_executor", "qwen_reviewer", "qwen_verifier")


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


def run_four_agent_private_training(
    dataset_dir: Path,
    rounds: int = 3,
    batches_per_round: int = 32,
    batch_size: int = 1,
    agent_names: tuple[str, ...] = AGENT_NAMES,
    model_factory: type[TrainableQwenModel] = MemoryQwenModel,
) -> MultiAgentTrainingReport:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if batches_per_round <= 0:
        raise ValueError("batches_per_round must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = GSM8KDataset(dataset_dir)
    train_samples = dataset.load("train")
    test_samples = dataset.load("test")
    if not train_samples:
        raise ValueError(f"no training samples found at {dataset_dir / 'train.jsonl'}")
    if not test_samples:
        raise ValueError(f"no test samples found at {dataset_dir / 'test.jsonl'}")

    private_splits = split_private_data(train_samples, len(agent_names))
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
    limit: int | None = None,
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
        predictions.append(
            VotingPrediction(
                index=index,
                question=sample.question,
                gold_answer=gold_answer,
                votes=votes,
                final_answer=final_answer,
                correct=final_answer == gold_answer,
            )
        )

    correct = sum(1 for prediction in predictions if prediction.correct)
    total = len(predictions)
    return VotingReport(
        total=total,
        correct=correct,
        accuracy=correct / total,
        predictions=predictions,
    )


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


def split_private_data(samples: list[GSM8KSample], agent_count: int) -> list[list[GSM8KSample]]:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    splits: list[list[GSM8KSample]] = [[] for _ in range(agent_count)]
    for index, sample in enumerate(samples):
        splits[index % agent_count].append(sample)
    return splits


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train four Qwen agents on separate private GSM8K splits.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batches-per-round", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("train/four_agent_training_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_four_agent_private_training(
        dataset_dir=args.dataset_dir,
        rounds=args.rounds,
        batches_per_round=args.batches_per_round,
        batch_size=args.batch_size,
    )
    export_report(report, args.output)
    for agent in report.agents:
        print(f"{agent.agent_name}: accuracy={agent.final_accuracy:.4f}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
