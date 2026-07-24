
# LoRA 使用每个智能体自己的精选 SFT 数据集训练。
# 专业经验库独立维护经他人评价筛选的成功经验，以及错误解法的反思经验。
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT, extract_numeric_answer, normalize_answer
from b_magent.datasets import GSM8KDataset, GSM8KSample
from b_magent.local_qwen import (
    DEFAULT_QWEN_MODEL,
    LocalQwenAgentModel,
    LocalQwenEngine,
    LocalQwenEvolutionBackend,
)
from b_magent.lora import DEFAULT_LORA_THRESHOLD, LoraEvolutionManager, LoraTrainingConfig, LoraUpdate
from b_magent.models import LibraryRecord
from b_magent.seed import seed_agent_libraries
from b_magent.tagging import ROUTING_TAGS, extract_math_task_tags, routing_tag_importance
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


AGENT_NAMES = ("qwen_agent_1", "qwen_agent_2", "qwen_agent_3", "qwen_agent_4")
STANDARD_PRIVATE_TRAIN_SIZE = 200


class TrainableQwenModel(Protocol):
    def train_batch(self, batch: list[GSM8KSample]) -> None:
        """Update one agent model with one private batch."""

    def generate(self, question: str) -> str:
        """Return one answer for evaluation."""


class ServerGuidedQwenModel(Protocol):
    def generate_with_server_guidance(self, question: str, server_guidance: str) -> str:
        """Answer a question using observable server-side evaluation advice."""


class ServerRoutingModel(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a server-side diagnostic for routing test questions."""


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
    tag_match_score: float = 0.0


@dataclass
class ServerRoutingAssessment:
    difficulty: str = "medium"
    key_steps: list[str] = field(default_factory=list)
    risk_steps: list[str] = field(default_factory=list)
    capability_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

    @property
    def routing_tags(self) -> set[str]:
        return set(self.capability_tags) | set(self.risk_tags)


@dataclass
class VotingPrediction:
    index: int
    question: str
    gold_answer: str
    votes: list[AgentVote]
    final_answer: str
    correct: bool
    server_diagnostic: str = ""
    difficulty: str = ""
    key_steps: list[str] = field(default_factory=list)
    risk_steps: list[str] = field(default_factory=list)
    selected_agents: list[str] = field(default_factory=list)
    routing_tags: list[str] = field(default_factory=list)
    routing_scores: dict[str, float] = field(default_factory=dict)
    matched_tags: dict[str, list[str]] = field(default_factory=dict)


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
    global_downlinks: int = 0
    global_uploads: int = 0
    lora_updates: list[LoraUpdate] = field(default_factory=list)


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
    curated_success_records: dict[str, int]
    error_reflection_records: dict[str, int]
    evaluation_records: dict[str, int]
    lora_enabled: bool = False
    lora_updates: dict[str, int] = field(default_factory=dict)

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
    on_round_start: Callable[[int, int, str], None] | None = None,
    on_round_end: Callable[[int, int, BMagentTrainingRound], None] | None = None,
    start_round: int = 0,
    preserve_private_datasets: bool = False,
) -> BMagentTrainingReport:
    if rounds is not None and rounds <= 0:
        raise ValueError("rounds must be positive when explicitly set")
    if private_batch_size <= 0:
        raise ValueError("private_batch_size must be positive")

    dataset = GSM8KDataset(dataset_dir)
    train_samples = dataset.load("train")
    if not train_samples:
        raise ValueError(f"no training samples found at {dataset_dir / 'train.jsonl'}")

    if start_round < 0:
        raise ValueError("start_round must not be negative")
    if rounds is not None and start_round >= rounds:
        raise ValueError("start_round must be smaller than total rounds")
    if preserve_private_datasets:
        private_dataset_counts = {
            agent_name: count_jsonl_lines(data_dir / agent_name / "private_data.jsonl")
            for agent_name in AGENT_NAMES
        }
        if not all(private_dataset_counts.values()):
            raise ValueError("resume requested but one or more private datasets are missing")
    else:
        private_dataset_counts = write_even_agent_private_datasets(train_samples, data_dir, AGENT_NAMES)
    base_participant_schedule = build_participant_schedule(private_dataset_counts, private_batch_size)
    effective_rounds = rounds or len(base_participant_schedule)
    participant_schedule = expand_participant_schedule(base_participant_schedule, effective_rounds)
    agents = build_default_agents(data_dir.parent, backend=backend)
    seed_agent_libraries(agents)
    if start_round:
        for agent in agents:
            agent.restore_private_cursor(private_batch_size)
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

    for index in range(start_round, effective_rounds):
        sample = train_samples[index % len(train_samples)]
        task = format_gsm8k_training_task(sample)
        if on_round_start is not None:
            on_round_start(index + 1, effective_rounds, sample.question)
        global_downlinks = downlink_global_evaluation_experience(
            task,
            agents,
            workflow.server_agent,
        )
        report = workflow.run(task, participant_names=participant_schedule[index])
        global_uploads = len(report.global_experience.global_updates) if report.global_experience else 0
        lora_updates = []
        if lora_manager is not None:
            lora_updates = lora_manager.update_from_round(
                task=report.task,
                drafts=report.drafts,
                peer_reviews=report.peer_reviews,
                self_improvements=report.self_improvements,
            )
            flush_pending = getattr(lora_manager, "flush_pending", None)
            if index == effective_rounds - 1 and callable(flush_pending):
                lora_updates.extend(flush_pending(AGENT_NAMES))
        round_report = BMagentTrainingRound(
            round_index=index + 1,
            task=task,
            participants=report.participants,
            evaluators=report.evaluators,
            drafts=len(report.drafts),
            peer_reviews=len(report.peer_reviews),
            self_improvements=len(report.self_improvements),
            evaluation_evolutions=len(report.evaluation_evolutions),
            global_downlinks=global_downlinks,
            global_uploads=global_uploads,
            lora_updates=lora_updates,
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
        curated_success_records={
            agent.name: count_new_records_with_tag(
                agent.professional_library.all_records(),
                professional_before[agent.name],
                "curated-success-experience",
            )
            for agent in agents
        },
        error_reflection_records={
            agent.name: count_new_records_with_tag(
                agent.professional_library.all_records(),
                professional_before[agent.name],
                "error-reflection-experience",
            )
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
    )


def count_new_records_with_tag(records: list[LibraryRecord], start_index: int, tag: str) -> int:
    return sum(1 for record in records[start_index:] if tag in record.tags)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def infer_completed_training_rounds(data_dir: Path) -> int:
    completed_improvements = 0
    for agent_name in AGENT_NAMES:
        path = data_dir / agent_name / "professional_library.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if "self-evolution" in payload.get("tags", []):
                completed_improvements += 1
    return completed_improvements // 2


def downlink_global_evaluation_experience(
    task: str,
    agents: list[object],
    server_agent: object,
    limit: int = 3,
) -> int:
    """Distribute server-side global review lessons to client evaluation libraries."""
    global_library = getattr(server_agent, "global_library", None)
    search = getattr(global_library, "search", None)
    if not callable(search):
        return 0
    global_records = search(task, limit=limit)
    if not global_records:
        return 0

    downlinks = 0
    for agent in agents:
        evaluation_library = getattr(agent, "evaluation_library", None)
        add_record = getattr(evaluation_library, "add_record", None)
        all_records = getattr(evaluation_library, "all_records", None)
        if not callable(add_record) or not callable(all_records):
            continue
        existing_source_ids = {
            _extract_global_source_id(record.detail)
            for record in all_records()
            if "source_global_experience_id=" in record.detail
        }
        for global_record in global_records:
            source_id = _global_record_source_id(global_record)
            if source_id in existing_source_ids:
                continue
            add_record(
                LibraryRecord(
                    agent_name=agent.name,
                    library_type="evaluation",
                    source_task=task,
                    summary=f"全局评价经验下发: {global_record.summary}",
                    detail=(
                        "Server-downlinked global evaluation experience for this round. "
                        f"source_global_experience_id={source_id} | "
                        f"source_server={global_record.agent_name} | "
                        f"source_detail={global_record.detail}"
                    ),
                    tags=[
                        "global-downlink",
                        "server-agent",
                        "evaluation",
                        *global_record.tags,
                    ],
                )
            )
            existing_source_ids.add(source_id)
            downlinks += 1
    return downlinks


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
    server_model: ServerRoutingModel | None = None,
    server_training_tag_records: list[LibraryRecord] | None = None,
    prior_global_evaluation_records: list[LibraryRecord] | None = None,
) -> VotingReport:
    dataset = GSM8KDataset(dataset_dir)
    test_samples = dataset.load("test", limit=limit)
    if not test_samples:
        raise ValueError(f"no test samples found at {dataset_dir / 'test.jsonl'}")
    missing_agents = [agent_name for agent_name in agent_names if agent_name not in models]
    if missing_agents:
        raise ValueError(f"missing models for agents: {', '.join(missing_agents)}")

    tag_index = _build_agent_tag_index(server_training_tag_records or [])
    predictions: list[VotingPrediction] = []
    with ThreadPoolExecutor(max_workers=len(agent_names), thread_name_prefix="voting-agent") as executor:
        for index, sample in enumerate(test_samples):
            server_diagnostic = ""
            selected_agents = list(agent_names)
            routing_tags: set[str] = set()
            routing_scores: dict[str, float] = {}
            matched_tags: dict[str, list[str]] = {}
            assessment = ServerRoutingAssessment()
            if server_model is not None and tag_index:
                server_diagnostic = _server_diagnose_question(
                    server_model,
                    sample.question,
                    prior_global_evaluation_records or [],
                )
                assessment = _parse_server_routing_assessment(server_diagnostic, sample.question)
                selected_agents, matched_tags = select_agents_by_server_tags(
                    server_diagnostic,
                    tag_index,
                    agent_names,
                    selected_count=3,
                    question=sample.question,
                    assessment=assessment,
                )
                routing_tags = assessment.routing_tags
                routing_scores = {
                    agent_name: _agent_routing_score(
                        tag_index.get(agent_name, {}),
                        routing_tags,
                        assessment.difficulty,
                    )
                    for agent_name in agent_names
                }
            raw_predictions = executor.map(
                lambda agent_name: _generate_with_server_guidance(
                    models[agent_name],
                    sample.question,
                    "",
                ),
                selected_agents,
            )
            votes = [
                AgentVote(
                    agent_name=agent_name,
                    raw_prediction=raw_prediction,
                    predicted_answer=extract_numeric_answer(raw_prediction),
                    tag_match_score=routing_scores.get(agent_name, 0.0),
                )
                for agent_name, raw_prediction in zip(selected_agents, raw_predictions)
            ]
            final_answer = routed_vote(votes) if server_model is not None and tag_index else majority_vote(votes)
            gold_answer = normalize_answer(sample.final_answer)
            prediction = VotingPrediction(
                index=index,
                question=sample.question,
                gold_answer=gold_answer,
                votes=votes,
                final_answer=final_answer,
                correct=final_answer == gold_answer,
                server_diagnostic=server_diagnostic,
                difficulty=assessment.difficulty if server_diagnostic else "",
                key_steps=assessment.key_steps,
                risk_steps=assessment.risk_steps,
                selected_agents=selected_agents,
                routing_tags=sorted(routing_tags),
                routing_scores=routing_scores,
                matched_tags=matched_tags,
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


def routed_vote(votes: list[AgentVote]) -> str:
    """Use the most tag-matched agent unless the 2nd and 3rd agree."""
    if not votes:
        return ""
    ranked_votes = sorted(
        enumerate(votes),
        key=lambda indexed_vote: (indexed_vote[1].tag_match_score, -indexed_vote[0]),
        reverse=True,
    )
    ranked = [vote for _, vote in ranked_votes]
    if len(ranked) >= 3 and ranked[1].predicted_answer == ranked[2].predicted_answer:
        return ranked[1].predicted_answer
    return ranked[0].predicted_answer


def select_agents_by_server_tags(
    server_diagnostic: str,
    agent_tag_index: dict[str, set[str] | dict[str, float]],
    agent_names: tuple[str, ...] = AGENT_NAMES,
    selected_count: int = 3,
    question: str = "",
    assessment: ServerRoutingAssessment | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    assessment = assessment or _parse_server_routing_assessment(server_diagnostic, question)
    diagnostic_tags = assessment.routing_tags
    matched_tags = {
        agent_name: sorted(set(agent_tag_index.get(agent_name, {})) & diagnostic_tags)
        for agent_name in agent_names
    }
    ranked = sorted(
        agent_names,
        key=lambda agent_name: (
            _agent_routing_score(
                agent_tag_index.get(agent_name, {}),
                diagnostic_tags,
                assessment.difficulty,
            ),
            -agent_names.index(agent_name),
        ),
        reverse=True,
    )
    selected_names = set(ranked[:selected_count])
    selected = [agent_name for agent_name in agent_names if agent_name in selected_names]
    return selected, {agent_name: matched_tags[agent_name] for agent_name in selected}


def _server_diagnose_question(
    server_model: ServerRoutingModel,
    question: str,
    prior_global_evaluation_records: list[LibraryRecord],
) -> str:
    relevant_global_records = _select_relevant_global_records(question, prior_global_evaluation_records)
    global_memory = "\n".join(
        f"- summary={record.summary}; detail={' '.join(record.detail.split())[:360]}; "
        f"tags={', '.join(record.tags)}"
        for record in relevant_global_records
    ) or "(none)"
    prompt = (
        "Server-side routing diagnosis.\n"
        "First solve the test question privately on the server. Use chain-of-thought internally, "
        "then report only observable errors, likely failure points, and routing tags.\n\n"
        "Prior aggregated evaluation experience:\n"
        f"{global_memory}\n\n"
        "Test question:\n"
        f"{question}\n\n"
        "Use the relevant global evaluation experience to identify the likely failure points. "
        "Return JSON only with this schema: "
        '{"difficulty":"easy|medium|hard","key_steps":["..."],"risk_steps":["..."],'
        '"capability_tags":["..."],"risk_tags":["..."]}. '
        "Select capability_tags and risk_tags only from: "
        "addition, subtraction, multiplication, division, fraction, percentage, ratio, rate, "
        "unit-conversion, money, time, geometry, counting, multi-step, arithmetic, final-answer, "
        "verification, boundary, structure. key_steps and risk_steps must be short observable step "
        "descriptions, not hidden chain-of-thought. Assess the whole problem rather than choosing one tag."
    )
    return server_model.generate(prompt)


def _build_agent_tag_index(records: list[LibraryRecord]) -> dict[str, dict[str, float]]:
    evidence: dict[str, dict[str, dict[str, float]]] = {}
    for record in records:
        if not record.agent_name:
            continue
        source_library_type = _source_library_type(record)
        if source_library_type == "evaluation":
            continue
        semantic_tags = {
            str(tag).strip().lower().replace("_", "-")
            for tag in record.tags
            if str(tag).strip().lower().replace("_", "-") in ROUTING_TAGS
        }
        semantic_tags.update(extract_math_task_tags(record.source_task))
        semantic_tags.add("overall-reliability")
        task_key = record.source_task.strip() or record.created_at
        value = _training_evidence_value(record)
        agent_evidence = evidence.setdefault(record.agent_name, {})
        for tag in semantic_tags:
            task_evidence = agent_evidence.setdefault(tag, {})
            task_evidence[task_key] = max(task_evidence.get(task_key, 0.0), value)

    index: dict[str, dict[str, float]] = {}
    for agent_name, tag_evidence in evidence.items():
        index[agent_name] = {}
        for tag, task_values in tag_evidence.items():
            values = list(task_values.values())
            quality = (1.0 + sum(values)) / (2.0 + len(values))
            evidence_bonus = 1.0 + min(math.log1p(len(values)) / 10.0, 0.35)
            index[agent_name][tag] = round(quality * evidence_bonus, 4)
    return index


def _extract_routing_tags(text: str) -> set[str]:
    return extract_math_task_tags(text)


def _parse_server_routing_assessment(text: str, question: str = "") -> ServerRoutingAssessment:
    payload = _parse_json_payload(text)
    capability_tags = _normalized_routing_tags(payload.get("capability_tags", []))
    risk_tags = _normalized_routing_tags(payload.get("risk_tags", []))
    fallback_tags = _extract_routing_tags(f"{text}\n{question}")
    if not capability_tags:
        capability_tags = sorted(_extract_routing_tags(question))
    combined_tags = set(capability_tags) | set(risk_tags) | fallback_tags
    difficulty = str(payload.get("difficulty", "")).strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        specific_tags = combined_tags - {"arithmetic", "final-answer", "verification", "structure"}
        if "multi-step" in combined_tags and len(specific_tags) >= 3:
            difficulty = "hard"
        elif "multi-step" in combined_tags or len(specific_tags) >= 2:
            difficulty = "medium"
        else:
            difficulty = "easy"
    return ServerRoutingAssessment(
        difficulty=difficulty,
        key_steps=_string_list(payload.get("key_steps", [])),
        risk_steps=_string_list(payload.get("risk_steps", [])),
        capability_tags=sorted(set(capability_tags) | fallback_tags),
        risk_tags=sorted(risk_tags),
    )


def _parse_json_payload(text: str) -> dict[str, object]:
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _normalized_routing_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = set(ROUTING_TAGS)
    return [
        tag
        for item in value
        if (tag := str(item).strip().lower().replace("_", "-")) in allowed
    ]


def _string_list(value: object, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split())[:240] for item in value if str(item).strip()][:limit]


def _format_server_guidance(assessment: ServerRoutingAssessment) -> str:
    key_steps = "; ".join(assessment.key_steps) or "Follow the required mathematical steps."
    risk_steps = "; ".join(assessment.risk_steps) or "Verify intermediate and final calculations."
    return (
        f"Difficulty: {assessment.difficulty}\n"
        f"Key steps to cover: {key_steps}\n"
        f"Likely failure points to avoid: {risk_steps}\n"
        f"Required capabilities: {', '.join(assessment.capability_tags) or '(none)'}\n"
        f"Risk checks: {', '.join(assessment.risk_tags) or '(none)'}"
    )


def _generate_with_server_guidance(
    model: TrainableQwenModel,
    question: str,
    server_guidance: str,
) -> str:
    guided_generate = getattr(model, "generate_with_server_guidance", None)
    if server_guidance and callable(guided_generate):
        return guided_generate(question, server_guidance)
    return model.generate(question)


def _tag_match_score(profile: set[str] | dict[str, float], routing_tags: set[str]) -> float:
    if isinstance(profile, set):
        return sum(routing_tag_importance(tag) for tag in profile & routing_tags)
    return round(
        sum(profile.get(tag, 0.0) * routing_tag_importance(tag) for tag in routing_tags),
        4,
    )


def _agent_routing_score(
    profile: set[str] | dict[str, float],
    routing_tags: set[str],
    difficulty: str,
) -> float:
    tag_score = _tag_match_score(profile, routing_tags)
    reliability = profile.get("overall-reliability", 0.0) if isinstance(profile, dict) else 0.0
    reliability_weight = {"easy": 0.05, "medium": 0.20, "hard": 0.40}.get(difficulty, 0.20)
    return round(tag_score + reliability * reliability_weight, 4)


def _select_relevant_global_records(
    question: str,
    records: list[LibraryRecord],
    limit: int = 5,
) -> list[LibraryRecord]:
    question_tags = _extract_routing_tags(question)
    question_words = set(_routing_words(question))
    ranked = sorted(
        enumerate(records),
        key=lambda indexed_record: (
            len(
                question_tags
                & (
                    _extract_routing_tags(indexed_record[1].source_task)
                    | _extract_routing_tags(indexed_record[1].summary)
                    | _extract_routing_tags(" ".join(indexed_record[1].tags))
                )
            ),
            len(question_words & set(_routing_words(indexed_record[1].source_task))),
            indexed_record[0],
        ),
        reverse=True,
    )
    return [record for _, record in ranked[:limit]]


def _routing_words(text: str) -> list[str]:
    return [
        token
        for token in "".join(character if character.isalnum() else " " for character in text.lower()).split()
        if len(token) >= 4
    ]


def _source_library_type(record: LibraryRecord) -> str:
    marker = "source_library_type="
    if marker in record.detail:
        return record.detail.split(marker, 1)[1].split(" | ", 1)[0].strip()
    if len(record.tags) >= 3 and record.tags[1] == "agent-training-tags":
        return str(record.tags[2]).strip()
    return record.library_type


def _training_evidence_value(record: LibraryRecord) -> float:
    tags = set(record.tags)
    if "curated-success-experience" in tags:
        return 1.0
    if "error-reflection-experience" in tags:
        return 0.0
    if "private-training" in tags:
        return 0.60
    if "evaluated-experience" in tags:
        return 0.40
    return 0.50


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


def reset_b_magent_training_state(
    data_dir: Path,
    lora_output_dir: Path | None = Path("data/lora_adapters"),
    agent_names: tuple[str, ...] = AGENT_NAMES,
    reset_evaluation_libraries: bool = True,
    report_files: tuple[Path, ...] = (),
) -> None:
    """Remove all generated experience and results before a fresh b_magent run."""
    for agent_name in agent_names:
        agent_dir = data_dir / agent_name
        for file_name in ("professional_library.jsonl", "private_data.jsonl"):
            (agent_dir / file_name).unlink(missing_ok=True)
        if reset_evaluation_libraries:
            (agent_dir / "evaluation_library.jsonl").unlink(missing_ok=True)

    server_dir = data_dir / "qwen_server_agent"
    for file_name in ("global_evaluation_library.jsonl", "agent_training_tags.jsonl"):
        (server_dir / file_name).unlink(missing_ok=True)
    shutil.rmtree(server_dir / "agent_training_tags", ignore_errors=True)

    if lora_output_dir is not None:
        shutil.rmtree(lora_output_dir, ignore_errors=True)

    for report_file in report_files:
        report_file.unlink(missing_ok=True)


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


def _global_record_source_id(record: LibraryRecord) -> str:
    return f"{record.agent_name}:{record.created_at}"


def _extract_global_source_id(detail: str) -> str:
    marker = "source_global_experience_id="
    if marker not in detail:
        return ""
    source_id = detail.split(marker, 1)[1]
    return source_id.split(" | ", 1)[0].strip()


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
    routing_summary = ""
    if prediction.routing_tags:
        scores = ", ".join(
            f"{agent_name}={prediction.routing_scores.get(agent_name, 0.0):.3f}"
            for agent_name in AGENT_NAMES
        )
        routing_summary = (
            f" difficulty={prediction.difficulty} "
            f"tags=({', '.join(prediction.routing_tags)}) scores=({scores})"
        )
    return (
        f"[{prediction.index + 1}/{total}] result={status} "
        f"final={prediction.final_answer or '<empty>'} "
        f"gold={prediction.gold_answer} votes=({vote_summary}){routing_summary}"
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
        default=200,
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
        "--resume",
        action="store_true",
        help="Keep existing libraries/LoRA state and continue until --rounds total rounds.",
    )
    parser.add_argument(
        "--enable-lora",
        dest="enable_lora",
        action="store_true",
        default=True,
        help="Build per-agent curated SFT datasets from self-reflection and train LoRA adapters whenever an example is accepted. Enabled by default.",
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
    parser.add_argument(
        "--lora-threshold",
        type=int,
        default=DEFAULT_LORA_THRESHOLD,
        help="Number of newly accepted samples accumulated per agent before one LoRA refresh.",
    )
    parser.add_argument("--lora-max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-train-batch-size", type=int, default=4)
    parser.add_argument("--lora-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-epochs", type=float, default=1.0)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-min-evaluation-score", type=float, default=0.6)
    parser.add_argument(
        "--allow-uncorrect-lora-labels",
        action="store_true",
        help="Allow LoRA SFT examples even when a gold final answer is present and the reflected answer does not match it.",
    )
    args = parser.parse_args()
    args.dataset_dir = resolve_project_path(args.dataset_dir)
    args.output = resolve_project_path(args.output)
    args.lora_output_dir = resolve_project_path(args.lora_output_dir)
    args.model_path = str(resolve_project_path(Path(args.model_path)))
    return args


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_b_magent_backend(args: argparse.Namespace) -> object | None:
    if args.backend == "demo":
        return None
    engine = LocalQwenEngine(model_name_or_path=args.model_path)
    return LocalQwenEvolutionBackend(
        engine,
        lora_output_dir=args.lora_output_dir if args.enable_lora else None,
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
        per_device_train_batch_size=args.lora_train_batch_size,
        gradient_accumulation_steps=args.lora_gradient_accumulation_steps,
        num_train_epochs=args.lora_epochs,
        learning_rate=args.lora_learning_rate,
    )
    return LoraEvolutionManager(config)


def print_training_round_start(round_index: int, rounds: int, question: str) -> None:
    preview = " ".join(question.split())[:120]
    print(f"[{round_index}/{rounds}] 开始四智能体训练: {preview}", flush=True)


def print_training_round_end(round_index: int, rounds: int, report: BMagentTrainingRound) -> None:
    print(
        f"[{round_index}/{rounds}] 完成: drafts={report.drafts} "
        f"evaluations={report.peer_reviews} professional_evolutions={report.self_improvements} "
        f"evaluation_evolutions={report.evaluation_evolutions} "
        f"global_downlinks={report.global_downlinks} global_uploads={report.global_uploads}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    mode = "local-qwen-vote" if args.local_qwen else args.mode
    if mode == "b-magent":
        print(f"backend: {args.backend}", flush=True)
        print(f"model: {args.model_path}", flush=True)
        start_round = 0
        if args.resume:
            start_round = infer_completed_training_rounds(PROJECT_ROOT / "data")
            print(f"保留已有训练成果，从第 {start_round + 1} 轮继续", flush=True)
        else:
            reset_b_magent_training_state(
                PROJECT_ROOT / "data",
                lora_output_dir=args.lora_output_dir,
                reset_evaluation_libraries=True,
                report_files=(
                    args.output,
                    PROJECT_ROOT / "data" / "latest_report.json",
                    PROJECT_ROOT / "outputs" / "latest_report.json",
                    PROJECT_ROOT / "outputs" / "demo_report.json",
                    PROJECT_ROOT / "train" / "four_agent_lora_voting_100_report.json",
                ),
            )
            print("已清空之前的训练存储", flush=True)
        print("开始训练", flush=True)
        report = run_b_magent_training_entry(
            dataset_dir=args.dataset_dir,
            data_dir=PROJECT_ROOT / "data",
            rounds=args.rounds if args.rounds > 0 else None,
            private_batch_size=args.private_batch_size,
            random_seed=args.seed,
            backend=build_b_magent_backend(args),
            lora_manager=build_lora_manager(args),
            on_round_start=print_training_round_start,
            on_round_end=print_training_round_end,
            start_round=start_round,
            preserve_private_datasets=args.resume,
        )
        export_json_report(report, args.output)
        print(f"b_magent agents: {', '.join(report.agents)}")
        print(f"rounds: {report.rounds}")
        for agent_name in report.agents:
            professional_count = report.professional_records[agent_name]
            evaluation_count = report.evaluation_records[agent_name]
            print(
                f"{agent_name}: professional_records={professional_count} "
                f"curated_success_records={report.curated_success_records.get(agent_name, 0)} "
                f"error_reflection_records={report.error_reflection_records.get(agent_name, 0)} "
                f"evaluation_records={evaluation_count} "
                f"lora_updates={report.lora_updates.get(agent_name, 0)}"
            )
    elif mode == "local-qwen-vote":
        models = build_four_local_qwen_agents(
            args.model_path,
            lora_output_dir=args.lora_output_dir,
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
