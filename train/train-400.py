
# LoRA 使用每个智能体自己的精选 SFT 数据集训练。
# 专业经验库独立维护经他人评价筛选的成功经验，以及错误解法的反思经验。
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import string
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT, extract_numeric_answer, normalize_answer
from b_magent.datasets import (
    GSM8KDataset,
    GSM8KSample,
    VisionQADataset,
    VisionQASample,
    load_project_dataset,
)
from b_magent.local_qwen import (
    DEFAULT_QWEN_MODEL,
    GENERAL_TASK_INSTRUCTION,
    LocalQwenAgentModel,
    LocalQwenEngine,
    LocalQwenEvolutionBackend,
)
from b_magent.lora import LoraEvolutionManager, LoraTrainingConfig, LoraUpdate
from b_magent.models import LibraryRecord
from b_magent.seed import seed_agent_libraries
from b_magent.tagging import ROUTING_TAGS, extract_math_task_tags, routing_tag_importance
from b_magent.workflow import MultiAgentWorkflow, build_default_agents
from b_magent.web import WebSearchResult, WebSearchService, build_web_search_service_from_env
from b_magent.answer_validation import AnswerValidator, build_answer_validator_from_env
from s_server import ServerKeyInformationStore


AGENT_NAMES = tuple(f"qwen_agent_{index}" for index in range(1, 7))
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "infographicsvqa"
STANDARD_PRIVATE_TRAIN_SIZE = 400
DEFAULT_PRIVATE_BATCH_SIZE = 4
DEFAULT_TRAINING_LORA_THRESHOLD = 50
UNRESOLVED_VISUAL_ANSWER = "unable to determine"


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
    target: str = ""
    entities: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    requires_web_search: bool = False
    web_cache_ttl_seconds: int = 86400
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
    dataset: str = "gsm8k"
    gold_answers: list[str] = field(default_factory=list)
    anls: float | None = None
    server_diagnostic: str = ""
    server_synthesis: str = ""
    server_cache_hit: bool = False
    cached_key_information: str = ""
    web_search_cache_hit: bool = False
    web_search_results: list[dict[str, object]] = field(default_factory=list)
    difficulty: str = ""
    key_steps: list[str] = field(default_factory=list)
    risk_steps: list[str] = field(default_factory=list)
    selected_agents: list[str] = field(default_factory=list)
    routing_tags: list[str] = field(default_factory=list)
    routing_scores: dict[str, float] = field(default_factory=dict)
    matched_tags: dict[str, list[str]] = field(default_factory=dict)
    answer_validation_rationale: str = ""
    requirements_met: list[str] = field(default_factory=list)
    requirements_missed: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class VotingReport:
    total: int
    correct: int
    accuracy: float
    predictions: list[VotingPrediction]
    anls: float | None = None

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
    training_tag_updates: int = 0
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
    global_experience_records: int = 0
    training_tag_records: int = 0
    lora_enabled: bool = False
    lora_updates: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_b_magent_training_entry(
    dataset_dir: Path,
    data_dir: Path,
    rounds: int | None = None,
    private_batch_size: int = DEFAULT_PRIVATE_BATCH_SIZE,
    private_train_size: int = STANDARD_PRIVATE_TRAIN_SIZE,
    random_seed: int | None = None,
    backend: object | None = None,
    lora_manager: LoraEvolutionManager | None = None,
    on_round_start: Callable[[int, int, str], None] | None = None,
    on_round_end: Callable[[int, int, BMagentTrainingRound], None] | None = None,
    start_round: int = 0,
    preserve_private_datasets: bool = False,
    answer_validator: AnswerValidator | None = None,
) -> BMagentTrainingReport:
    if rounds is not None and rounds <= 0:
        raise ValueError("rounds must be positive when explicitly set")
    if private_batch_size <= 0:
        raise ValueError("private_batch_size must be positive")
    if private_train_size <= 0:
        raise ValueError("private_train_size must be positive")

    dataset = load_project_dataset(dataset_dir)
    train_samples = dataset.load("train")
    if not train_samples:
        raise ValueError(f"no training samples found at {dataset_dir / 'train.jsonl'}")

    if start_round < 0:
        raise ValueError("start_round must not be negative")
    if rounds is not None and start_round > rounds:
        raise ValueError("start_round must not exceed total rounds")
    if preserve_private_datasets:
        private_dataset_counts = {
            agent_name: count_jsonl_lines(data_dir / agent_name / "private_data.jsonl")
            for agent_name in AGENT_NAMES
        }
        if not all(private_dataset_counts.values()):
            raise ValueError("resume requested but one or more private datasets are missing")
    else:
        sampled_train_samples, private_dataset_counts = write_random_agent_private_datasets(
            train_samples,
            data_dir,
            AGENT_NAMES,
            private_train_size,
            random_seed,
        )
    if preserve_private_datasets:
        sampled_train_samples = train_samples
    base_participant_schedule = build_participant_schedule(
        private_dataset_counts,
        private_batch_size,
        random_seed=random_seed,
    )
    effective_rounds = rounds or len(base_participant_schedule)
    participant_schedule = expand_participant_schedule(base_participant_schedule, effective_rounds)
    agents = build_default_agents(
        data_dir.parent,
        backend=backend,
        answer_validator=answer_validator,
    )
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
    global_records_before = len(workflow.server_agent.global_library.all_records())
    training_tag_records_before = count_jsonl_lines(
        data_dir / "qwen_server_agent" / "agent_training_tags.jsonl"
    )
    training_rounds: list[BMagentTrainingRound] = []

    for index in range(start_round, effective_rounds):
        sample = sampled_train_samples[index % len(sampled_train_samples)]
        task = format_training_task(sample)
        participant_names = participant_schedule[index]
        participant_name_set = set(participant_names)
        evaluator_agents = [
            agent for agent in agents if agent.name not in participant_name_set
        ]
        if on_round_start is not None:
            on_round_start(index + 1, effective_rounds, sample.question)
        global_downlinks = downlink_global_evaluation_experience(
            task,
            evaluator_agents,
            workflow.server_agent,
        )
        report = workflow.run(task, participant_names=participant_names)
        global_uploads = len(report.global_experience.global_updates) if report.global_experience else 0
        lora_updates = []
        if lora_manager is not None:
            lora_updates = lora_manager.update_from_round(
                task=report.task,
                drafts=report.drafts,
                peer_reviews=report.peer_reviews,
                self_improvements=report.self_improvements,
            )
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
            training_tag_updates=len(report.server_training_tag_updates),
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
        global_experience_records=(
            len(workflow.server_agent.global_library.all_records()) - global_records_before
        ),
        training_tag_records=(
            count_jsonl_lines(data_dir / "qwen_server_agent" / "agent_training_tags.jsonl")
            - training_tag_records_before
        ),
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


def load_training_progress(data_dir: Path) -> int:
    """Compatibility name for reading completed rounds without mutating state."""
    return infer_completed_training_rounds(data_dir)


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


def run_six_agent_private_training(
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


def run_six_agent_voting_on_test(
    dataset_dir: Path,
    models: dict[str, TrainableQwenModel],
    agent_names: tuple[str, ...] = AGENT_NAMES,
    limit: int | None = STANDARD_TEST_LIMIT,
    on_prediction: Callable[["VotingPrediction", int], None] | None = None,
    server_model: ServerRoutingModel | None = None,
    server_training_tag_records: list[LibraryRecord] | None = None,
    prior_global_evaluation_records: list[LibraryRecord] | None = None,
    server_key_information_store: ServerKeyInformationStore | None = None,
    web_search_service: WebSearchService | None = None,
    answer_validator: AnswerValidator | None = None,
    split: str = "test",
    enable_server_cache: bool = True,
) -> VotingReport:
    dataset = load_project_dataset(dataset_dir)
    test_samples = dataset.load(split, limit=limit)
    if not test_samples:
        raise ValueError(f"no test samples found at {dataset_dir / f'{split}.jsonl'}")
    missing_agents = [agent_name for agent_name in agent_names if agent_name not in models]
    if missing_agents:
        raise ValueError(f"missing models for agents: {', '.join(missing_agents)}")
    if server_model is not None and enable_server_cache and server_key_information_store is None:
        server_key_information_store = ServerKeyInformationStore(
            dataset_dir.parent / "qwen_server_agent" / "key_information_store.jsonl"
        )
    if server_model is not None and web_search_service is None:
        web_search_service = build_web_search_service_from_env(dataset_dir.parent)

    tag_index = _build_agent_tag_index(server_training_tag_records or [])
    predictions: list[VotingPrediction] = []
    with ThreadPoolExecutor(max_workers=len(agent_names), thread_name_prefix="voting-agent") as executor:
        for index, sample in enumerate(test_samples):
            inference_question = format_inference_question(sample)
            cache_question = inference_question if isinstance(sample, VisionQASample) else sample.question
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
                routing_tags = assessment.routing_tags
                selected_agents, matched_tags = select_agents_by_server_tags(
                    server_diagnostic,
                    tag_index,
                    agent_names,
                    selected_count=min(3, len(agent_names)),
                    question=sample.question,
                    assessment=assessment,
                )
                routing_scores = {
                    agent_name: _agent_routing_score(
                        tag_index.get(agent_name, {}),
                        routing_tags,
                        assessment.difficulty,
                    )
                    for agent_name in agent_names
                }
            cached_record = None
            if enable_server_cache and server_key_information_store is not None:
                cached_record = server_key_information_store.lookup(
                    cache_question,
                    key_facts=[
                        assessment.target,
                        *assessment.facts,
                        *assessment.constraints,
                        *assessment.entities,
                    ],
                    query_tags=assessment.routing_tags,
                )
            server_cache_hit = cached_record is not None
            cached_key_information = cached_record.detail if cached_record is not None else ""
            if server_cache_hit:
                selected_agents = []
                votes: list[AgentVote] = []
                server_synthesis = cached_record.summary
            else:
                raw_predictions = executor.map(
                    lambda agent_name: _generate_with_server_guidance(
                        models[agent_name],
                        inference_question,
                        "",
                    ),
                    selected_agents,
                )
                votes = [
                    AgentVote(
                        agent_name=agent_name,
                        raw_prediction=raw_prediction,
                        predicted_answer=extract_prediction_answer(raw_prediction, sample),
                        tag_match_score=routing_scores.get(agent_name, 0.0),
                    )
                    for agent_name, raw_prediction in zip(selected_agents, raw_predictions)
                ]
                if isinstance(sample, VisionQASample):
                    votes = _retry_empty_visual_votes(
                        votes,
                        models,
                        inference_question,
                        executor,
                        sample,
                    )
                server_synthesis = ""
            web_results: list[WebSearchResult] = []
            web_search_cache_hit = False
            if (
                not server_cache_hit
                and assessment.requires_web_search
                and web_search_service is not None
            ):
                web_queries = assessment.search_queries or [assessment.target, sample.question]
                web_results, web_search_cache_hit = web_search_service.search_and_fetch(
                    cache_question,
                    web_queries,
                    ttl_seconds=assessment.web_cache_ttl_seconds,
                )
            is_visual = isinstance(sample, VisionQASample)
            vote_normalizer = normalize_vision_answer if is_visual else normalize_answer
            unanimous_answer = unanimous_vote(votes, vote_normalizer) if is_visual else ""
            if server_model is not None and not server_cache_hit and not unanimous_answer:
                relevant_global_records = _select_relevant_global_records(
                    sample.question,
                    prior_global_evaluation_records or [],
                    assessment=assessment,
                )
                server_synthesis = _server_synthesize_agent_answers(
                    server_model,
                    inference_question,
                    votes,
                    assessment,
                    relevant_global_records,
                    web_results,
                )
            synthesized_answer = extract_prediction_answer(server_synthesis, sample)
            if (
                is_visual
                and server_model is not None
                and not unanimous_answer
                and not synthesized_answer
            ):
                server_synthesis = _retry_server_visual_answer(
                    server_model,
                    inference_question,
                    votes,
                )
                if not server_synthesis.strip():
                    server_synthesis = UNRESOLVED_VISUAL_ANSWER
                synthesized_answer = extract_prediction_answer(server_synthesis, sample)
            fallback_answer = (
                routed_vote(votes)
                if server_model is not None and tag_index
                else majority_vote(votes)
            )
            final_answer = unanimous_answer or synthesized_answer or fallback_answer
            if is_visual and not final_answer:
                final_answer = UNRESOLVED_VISUAL_ANSWER
            gold_answers = list(sample.answers) if is_visual else [sample.final_answer]
            normalize = normalize_vision_answer if is_visual else normalize_answer
            gold_answer = normalize(gold_answers[0])
            sample_anls = infographic_anls(final_answer, tuple(gold_answers)) if is_visual else None
            validation_rationale = ""
            requirements_met: list[str] = []
            requirements_missed: list[str] = []
            unsupported_claims: list[str] = []
            if answer_validator is not None and not is_visual:
                answer_for_validation = server_synthesis or next(
                    (
                        vote.raw_prediction
                        for vote in votes
                        if vote.predicted_answer == final_answer
                    ),
                    final_answer,
                )
                validation = answer_validator.validate(
                    format_answer_validation_task(sample),
                    answer_for_validation,
                )
                correct = validation.correct
                validation_rationale = validation.rationale
                requirements_met = validation.requirements_met
                requirements_missed = validation.requirements_missed
                unsupported_claims = validation.unsupported_claims
            else:
                correct = normalize(final_answer) in {normalize(answer) for answer in gold_answers}
            prediction = VotingPrediction(
                index=index,
                question=sample.question,
                gold_answer=gold_answer,
                votes=votes,
                final_answer=final_answer,
                correct=correct,
                dataset=sample.dataset if is_visual else "gsm8k",
                gold_answers=gold_answers,
                anls=sample_anls,
                server_diagnostic=server_diagnostic,
                server_synthesis=server_synthesis,
                server_cache_hit=server_cache_hit,
                cached_key_information=cached_key_information,
                web_search_cache_hit=web_search_cache_hit,
                web_search_results=[result.to_dict() for result in web_results],
                difficulty=assessment.difficulty if server_diagnostic else "",
                key_steps=assessment.key_steps,
                risk_steps=assessment.risk_steps,
                selected_agents=selected_agents,
                routing_tags=sorted(routing_tags),
                routing_scores=routing_scores,
                matched_tags=matched_tags,
                answer_validation_rationale=validation_rationale,
                requirements_met=requirements_met,
                requirements_missed=requirements_missed,
                unsupported_claims=unsupported_claims,
            )
            if (
                server_key_information_store is not None
                and enable_server_cache
                and not server_cache_hit
                and prediction.correct
                and server_synthesis
            ):
                server_key_information_store.store_verified(
                    cache_question,
                    server_synthesis,
                    final_answer,
                    _assessment_key_information(assessment),
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
        anls=(
            sum(prediction.anls or 0.0 for prediction in predictions) / total
            if predictions and any(prediction.anls is not None for prediction in predictions)
            else None
        ),
    )


def build_six_local_qwen_agents(
    model_name_or_path: str | Path = DEFAULT_QWEN_MODEL,
    agent_names: tuple[str, ...] = AGENT_NAMES,
    device_map: str = "auto",
    torch_dtype: str = "float16",
    lora_output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    professional_memory_limit: int = 3,
    require_lora: bool = False,
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
            professional_library_path=(
                Path(data_dir) / agent_name / "professional_library.jsonl"
                if data_dir is not None
                else None
            ),
            professional_memory_limit=professional_memory_limit,
            require_lora=require_lora,
        )
        for agent_name in agent_names
    }


def normalize_vision_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", value)
    punctuation = string.punctuation + "“”‘’–—"
    value = value.translate(str.maketrans({character: " " for character in punctuation}))
    return " ".join(value.split())


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def infographic_anls(prediction: str, answers: tuple[str, ...]) -> float:
    predicted = normalize_vision_answer(prediction)
    if not predicted or not answers:
        return 0.0
    best = 0.0
    for answer in answers:
        reference = normalize_vision_answer(answer)
        denominator = max(len(predicted), len(reference))
        similarity = 1.0 - (_edit_distance(predicted, reference) / denominator) if denominator else 1.0
        best = max(best, similarity)
    return best if best >= 0.5 else 0.0


def format_inference_question(sample: GSM8KSample | VisionQASample) -> str:
    if not isinstance(sample, VisionQASample):
        return sample.question
    return (
        f"Image: {sample.image_path}\n"
        f"Question: {sample.question}\n"
        "Inspect the complete image and return only the short direct answer from visible evidence."
    )


def format_training_task(sample: GSM8KSample | VisionQASample) -> str:
    if not isinstance(sample, VisionQASample):
        return format_gsm8k_training_task(sample)
    answers = " | ".join(sample.answers or (sample.final_answer,))
    return (
        f"Image: {sample.image_path}\n"
        f"Question: {sample.question}\n"
        f"Gold image elements: {json.dumps(sample.image_elements, ensure_ascii=False)}\n"
        "Gold reasoning: identify the visible evidence that directly supports the answer.\n"
        f"Gold final answer: {answers}"
    )


def extract_prediction_answer(
    raw_prediction: str,
    sample: GSM8KSample | VisionQASample,
) -> str:
    if not isinstance(sample, VisionQASample):
        return extract_numeric_answer(raw_prediction)
    text = str(raw_prediction).strip()
    value = ""
    parsed_json = False
    try:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            parsed_json = True
            value = str(payload.get("final_answer", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = re.search(r'["\']?final_answer["\']?\s*:\s*["\']([^"\'\n}]*)', text, re.I)
        if match:
            value = match.group(1)
    if not value and not parsed_json:
        match = re.search(r"(?:final\s+answer|answer)\s*:\s*(.+)", text, re.I)
        value = match.group(1) if match else text
    value = value.strip().strip("`\"'").rstrip(".,")
    normalized = normalize_vision_answer(value)
    # Long prose or malformed JSON is not a usable short VQA vote.
    if len(normalized) > 200 or len(normalized.split()) > 30 or normalized.startswith("{"):
        return ""
    return normalized


def _retry_empty_visual_votes(
    votes: list[AgentVote],
    models: dict[str, TrainableQwenModel],
    question: str,
    executor: ThreadPoolExecutor,
    sample: VisionQASample,
) -> list[AgentVote]:
    empty_votes = [vote for vote in votes if not vote.predicted_answer]
    if not empty_votes:
        return votes

    def retry(vote: AgentVote) -> str:
        model = models[vote.agent_name]
        generate_concise = getattr(model, "generate_concise_visual_answer", None)
        if callable(generate_concise):
            return str(generate_concise(question))
        return _generate_with_server_guidance(
            model,
            question,
            "Return only a non-empty short final answer. Do not return JSON or explanation.",
        )

    retry_outputs = executor.map(retry, empty_votes)
    for vote, retry_output in zip(empty_votes, retry_outputs):
        retry_answer = extract_prediction_answer(retry_output, sample)
        resolved_retry = retry_output.strip() or UNRESOLVED_VISUAL_ANSWER
        vote.raw_prediction = resolved_retry
        vote.predicted_answer = retry_answer or UNRESOLVED_VISUAL_ANSWER
    return votes


def _is_usable_vote_answer(answer: str) -> bool:
    compact = " ".join(str(answer).split())
    return bool(compact) and len(compact) <= 200 and len(compact.split()) <= 30


def majority_vote(votes: list[AgentVote]) -> str:
    votes = [vote for vote in votes if _is_usable_vote_answer(vote.predicted_answer)]
    if not votes:
        return ""
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


def unanimous_vote(votes: list[AgentVote], normalize: Callable[[str], str]) -> str:
    usable_votes = [vote for vote in votes if _is_usable_vote_answer(vote.predicted_answer)]
    if len(usable_votes) < 2 or len(usable_votes) != len(votes):
        return ""
    normalized_answers = {normalize(vote.predicted_answer) for vote in usable_votes}
    if len(normalized_answers) != 1:
        return ""
    return usable_votes[0].predicted_answer


def routed_vote(votes: list[AgentVote]) -> str:
    """Use the most tag-matched agent unless the 2nd and 3rd agree."""
    votes = [vote for vote in votes if _is_usable_vote_answer(vote.predicted_answer)]
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
        '"target":"...","entities":["..."],"facts":["..."],"constraints":["..."],'
        '"numbers":["..."],"units":["..."],"keywords":["..."],'
        '"search_queries":["..."],"requires_web_search":true|false,'
        '"web_cache_ttl_seconds":86400,"capability_tags":["..."],"risk_tags":["..."]}. '
        "Extract only information stated or directly requested by the question. search_queries "
        "must be short retrieval phrases describing the goal and its important constraints. "
        "Set requires_web_search=true only when the answer needs current or external facts that "
        "are not supplied in the question or prior memory. Use a shorter cache TTL for volatile "
        "facts such as prices, schedules, weather, or news. "
        "Select capability_tags and risk_tags only from: "
        "addition, subtraction, multiplication, division, fraction, percentage, ratio, rate, "
        "unit-conversion, money, time, geometry, counting, multi-step, arithmetic, final-answer, "
        "verification, boundary, structure, summarization, information-synthesis, "
        "constraint-preservation. "
        "For planning, research, and evidence-based answers, include the relevant summarization "
        "capabilities because client specialization is trained for synthesis rather than search. "
        "key_steps and risk_steps must be short observable step "
        "descriptions, not hidden chain-of-thought. Assess the whole problem rather than choosing one tag."
    )
    return server_model.generate(prompt)


def _server_synthesize_agent_answers(
    server_model: ServerRoutingModel,
    question: str,
    votes: list[AgentVote],
    assessment: ServerRoutingAssessment,
    relevant_global_records: list[LibraryRecord],
    web_results: list[WebSearchResult],
) -> str:
    agent_outputs = "\n\n".join(
        f"[{vote.agent_name}]\n{vote.raw_prediction}"
        for vote in votes
    ) or "(none)"
    global_memory = "\n".join(
        f"- summary={record.summary}; detail={' '.join(record.detail.split())[:360]}; "
        f"tags={', '.join(record.tags)}"
        for record in relevant_global_records
    ) or "(none)"
    web_evidence = "\n\n".join(
        f"[{index}] title={result.title}\nurl={result.url}\n"
        f"retrieved_at={result.retrieved_at}\n"
        f"images={', '.join(result.image_urls) or '(none)'}\n"
        f"evidence={(result.content or result.snippet)[:2000]}"
        for index, result in enumerate(web_results, start=1)
    ) or "(none)"
    prompt = (
        "Server-side selected-agent synthesis.\n"
        "Independently distill the useful facts, calculations, checks, and conclusions from every "
        "selected agent response. Form their union, remove duplicates and contradictions, and solve "
        "any remaining disagreement on the server. Read every selected response in full.\n\n"
        f"Question:\n{question}\n\n"
        f"Observable key steps:\n{json.dumps(assessment.key_steps, ensure_ascii=False)}\n\n"
        f"Observable risk steps:\n{json.dumps(assessment.risk_steps, ensure_ascii=False)}\n\n"
        f"Retrieved experience relevant to the target and constraints:\n{global_memory}\n\n"
        f"Server-retrieved web evidence:\n{web_evidence}\n\n"
        f"Selected agent responses:\n{agent_outputs}\n\n"
        "Use web evidence only for claims it directly supports. Preserve source URLs when external "
        "facts are used, and prefer official or more recently retrieved sources when they conflict. "
        + (
            "Inspect the image to resolve disagreements, then return only the shortest direct answer. "
            "Preserve exact visible spelling and numeric formatting; do not return JSON or reasoning."
            if _question_image_path(question) is not None
            else "Return a concise integrated solution followed by the final numeric answer in the exact "
            "form `#### number`."
        )
    )
    local_image_paths = list(
        dict.fromkeys(
            [str(_question_image_path(question))] if _question_image_path(question) is not None else []
            + [
                path
                for result in web_results
                for path in result.local_image_paths
                if Path(path).is_file()
            ]
        )
    )
    generate_multimodal = getattr(server_model, "generate_multimodal", None)
    if local_image_paths and callable(generate_multimodal):
        return generate_multimodal(prompt, local_image_paths)
    return server_model.generate(prompt)


def _retry_server_visual_answer(
    server_model: ServerRoutingModel,
    question: str,
    votes: list[AgentVote],
) -> str:
    agent_answers = "\n".join(
        f"- {vote.agent_name}: {vote.predicted_answer}"
        for vote in votes
    )
    prompt = (
        "Answer the visual question using the image and the three recognizer answers below. "
        "Resolve disagreements against the image. Return only one non-empty short final answer; "
        "do not return JSON, reasoning, labels, or explanation.\n\n"
        f"{question}\n\nRecognizer answers:\n{agent_answers}"
    )
    image_path = _question_image_path(question)
    generate_multimodal = getattr(server_model, "generate_multimodal", None)
    if image_path is not None and callable(generate_multimodal):
        return str(generate_multimodal(prompt, [str(image_path)]))
    return str(server_model.generate(prompt))


def _question_image_path(question: str) -> Path | None:
    match = re.search(r"(?m)^Image:\s*(.+?)\s*$", question)
    if not match:
        return None
    path = Path(match.group(1).strip())
    return path if path.is_file() else None


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
        completeness = _visual_completeness_score(record)
        agent_evidence = evidence.setdefault(record.agent_name, {})
        for tag in semantic_tags:
            task_evidence = agent_evidence.setdefault(tag, {})
            tag_value = completeness if tag == "overall-reliability" and completeness is not None else value
            task_evidence[task_key] = max(task_evidence.get(task_key, 0.0), tag_value)

    index: dict[str, dict[str, float]] = {}
    for agent_name, tag_evidence in evidence.items():
        index[agent_name] = {}
        for tag, task_values in tag_evidence.items():
            values = list(task_values.values())
            if tag != "overall-reliability" and len(values) < 3:
                continue
            quality = (1.0 + sum(values)) / (2.0 + len(values))
            evidence_bonus = 1.0 + min(math.log1p(len(values)) / 10.0, 0.35)
            index[agent_name][tag] = round(quality * evidence_bonus, 4)
    return index


def _visual_completeness_score(record: LibraryRecord) -> float | None:
    if "visual-content-completeness" not in record.tags:
        return None
    match = re.search(r"visual_completeness_score=([01](?:\.\d+)?)", record.detail)
    if not match:
        return None
    return min(max(float(match.group(1)), 0.0), 1.0)


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
        target=" ".join(str(payload.get("target", "")).split())[:360],
        entities=_string_list(payload.get("entities", []), limit=12),
        facts=_string_list(payload.get("facts", []), limit=12),
        constraints=_string_list(payload.get("constraints", []), limit=12),
        numbers=_string_list(payload.get("numbers", []), limit=12),
        units=_string_list(payload.get("units", []), limit=12),
        keywords=_string_list(payload.get("keywords", []), limit=12),
        search_queries=_string_list(payload.get("search_queries", []), limit=8),
        requires_web_search=payload.get("requires_web_search") is True,
        web_cache_ttl_seconds=_bounded_int(
            payload.get("web_cache_ttl_seconds"),
            default=86400,
            minimum=300,
            maximum=2_592_000,
        ),
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


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


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
    assessment: ServerRoutingAssessment | None = None,
) -> list[LibraryRecord]:
    assessment = assessment or ServerRoutingAssessment()
    queries = _unique_nonempty(
        [question, assessment.target, *assessment.search_queries, *assessment.keywords]
    )
    key_facts = _unique_nonempty(
        [*assessment.facts, *assessment.constraints, *assessment.entities, *assessment.numbers, *assessment.units]
    )
    query_tags = assessment.routing_tags or _extract_routing_tags(question)
    query_terms = set().union(*(_retrieval_terms(query) for query in queries)) if queries else set()
    fact_terms = [_retrieval_terms(fact) for fact in key_facts]
    ranked = sorted(
        enumerate(records),
        key=lambda indexed_record: (
            _global_record_relevance(indexed_record[1], query_terms, fact_terms, query_tags),
            indexed_record[0],
        ),
        reverse=True,
    )
    return [record for _, record in ranked[:limit]]


def _global_record_relevance(
    record: LibraryRecord,
    query_terms: set[str],
    fact_terms: list[set[str]],
    query_tags: set[str],
) -> float:
    record_text = f"{record.source_task} {record.summary} {record.detail} {' '.join(record.tags)}"
    record_terms = _retrieval_terms(record_text)
    record_tags = _extract_routing_tags(record_text)
    lexical = len(query_terms & record_terms) / len(query_terms) if query_terms else 0.0
    facts = max(
        (len(terms & record_terms) / len(terms) for terms in fact_terms if terms),
        default=0.0,
    )
    tags = len(query_tags & record_tags) / len(query_tags) if query_tags else 0.0
    quality = _training_evidence_value(record)
    return round(lexical * 0.35 + facts * 0.30 + tags * 0.25 + quality * 0.10, 6)


def _retrieval_terms(text: str) -> set[str]:
    lower = str(text).lower()
    terms = set(_routing_words(lower))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lower):
        if len(sequence) >= 2:
            terms.add(sequence)
            terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    terms.update(re.findall(r"\d+(?:\.\d+)?%?", lower))
    return terms


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if str(value).strip()))


def _assessment_key_information(assessment: ServerRoutingAssessment) -> dict[str, object]:
    return {
        "target": assessment.target,
        "entities": assessment.entities,
        "facts": assessment.facts,
        "constraints": assessment.constraints,
        "numbers": assessment.numbers,
        "units": assessment.units,
        "keywords": assessment.keywords,
        "search_queries": assessment.search_queries,
        "requires_web_search": assessment.requires_web_search,
        "web_cache_ttl_seconds": assessment.web_cache_ttl_seconds,
        "key_steps": assessment.key_steps,
        "risk_steps": assessment.risk_steps,
        "routing_tags": sorted(assessment.routing_tags),
    }


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
    private_batch_size: int = DEFAULT_PRIVATE_BATCH_SIZE,
    random_seed: int | None = None,
) -> list[list[str]]:
    if private_batch_size <= 0:
        raise ValueError("private_batch_size must be positive")
    remaining = {
        agent_name: (private_dataset_counts.get(agent_name, 0) + private_batch_size - 1) // private_batch_size
        for agent_name in AGENT_NAMES
    }
    rng = random.Random(random_seed)
    schedule: list[list[str]] = []
    while sum(remaining.values()) > 0:
        active = [agent_name for agent_name, count in remaining.items() if count > 0]
        selected = rng.sample(active, min(3, len(active)))
        if len(selected) < 3:
            inactive = [agent_name for agent_name in AGENT_NAMES if agent_name not in selected]
            selected.extend(rng.sample(inactive, 3 - len(selected)))
        schedule.append(selected)
        for agent_name in selected:
            remaining[agent_name] = max(0, remaining[agent_name] - 1)
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


def write_random_agent_private_datasets(
    samples: list[GSM8KSample],
    data_dir: Path,
    agent_names: tuple[str, ...] = AGENT_NAMES,
    samples_per_agent: int = STANDARD_PRIVATE_TRAIN_SIZE,
    random_seed: int | None = None,
) -> tuple[list[GSM8KSample], dict[str, int]]:
    required_samples = len(agent_names) * samples_per_agent
    if len(samples) < required_samples:
        raise ValueError(
            f"need at least {required_samples} training samples for {len(agent_names)} "
            f"agents with {samples_per_agent} private samples each; found {len(samples)}"
        )
    sampled = random.Random(random_seed).sample(samples, required_samples)
    counts: dict[str, int] = {}
    for agent_index, agent_name in enumerate(agent_names):
        start = agent_index * samples_per_agent
        private_samples = sampled[start : start + samples_per_agent]
        agent_dir = data_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        private_file = agent_dir / "private_data.jsonl"
        private_file.write_text(
            "\n".join(sample.to_training_text() for sample in private_samples) + "\n",
            encoding="utf-8",
        )
        counts[agent_name] = len(private_samples)
    return sampled, counts


def reset_b_magent_training_state(
    data_dir: Path,
    lora_output_dir: Path | None = Path("data/lora_adapters_qwen2_5_vl_7b"),
    agent_names: tuple[str, ...] = AGENT_NAMES,
    reset_evaluation_libraries: bool = True,
    reset_key_information_store: bool = False,
    report_files: tuple[Path, ...] = (),
) -> None:
    """Remove training state while preserving verified server key information by default."""
    for agent_name in agent_names:
        agent_dir = data_dir / agent_name
        for file_name in ("professional_library.jsonl", "private_data.jsonl"):
            (agent_dir / file_name).unlink(missing_ok=True)
        if reset_evaluation_libraries:
            (agent_dir / "evaluation_library.jsonl").unlink(missing_ok=True)

    server_dir = data_dir / "qwen_server_agent"
    for file_name in ("global_evaluation_library.jsonl", "agent_training_tags.jsonl"):
        (server_dir / file_name).unlink(missing_ok=True)
    if reset_key_information_store:
        (server_dir / "key_information_store.jsonl").unlink(missing_ok=True)
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
    task = (
        f"Complete this {sample.task_type} evidence summarization task.\n"
        "The second-layer server has already retrieved the candidate evidence. Synthesize the useful "
        "facts, remove duplication and conflicts, preserve every user constraint, and produce a concise "
        "evidence-grounded final answer. Do not perform another search.\n"
        f"Task: {sample.question}"
    )
    if sample.reference_information:
        task += f"\n\nCandidate reference information:\n{sample.reference_information}"
    if sample.retrieval_targets:
        task += f"\n\nGold retrieval targets:\n{'; '.join(sample.retrieval_targets)}"
    task += f"\n\nGold reference response:\n{sample.answer}"
    if sample.final_answer:
        task += f"\nGold final answer: {sample.final_answer}"
    return task


def format_answer_validation_task(sample: GSM8KSample | VisionQASample) -> str:
    task = f"Task: {sample.question}"
    if isinstance(sample, VisionQASample):
        return f"Visual question: {sample.question}"
    if sample.reference_information:
        task += f"\n\nCandidate reference information:\n{sample.reference_information}"
    return task


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


# Backward-compatible names for older launch commands.
run_four_agent_private_training = run_six_agent_private_training
run_four_agent_voting_on_test = run_six_agent_voting_on_test
build_four_local_qwen_agents = build_six_local_qwen_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entry for b_magent agents and benchmark runners.")
    parser.add_argument(
        "--mode",
        choices=["b-magent", "placeholder", "local-qwen-vote"],
        default="b-magent",
        help=(
            "b-magent trains the b_magent agent libraries; placeholder keeps the old "
            "offline memory baseline; local-qwen-vote runs six local Qwen candidates."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["local-qwen", "demo"],
        default="local-qwen",
        help="Backend for --mode b-magent. local-qwen calls the configured local model; demo is deterministic smoke test logic.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Training and test dataset directory (default: data/infographicsvqa).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Training rounds. Use 0 to auto-cover every sampled private training record.",
    )
    parser.add_argument(
        "--private-batch-size",
        type=int,
        default=DEFAULT_PRIVATE_BATCH_SIZE,
        help="Number of private examples each participating agent loads per round.",
    )
    parser.add_argument("--batches-per-round", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--private-train-size",
        type=int,
        default=STANDARD_PRIVATE_TRAIN_SIZE,
        help="Random, non-overlapping training samples assigned to each agent (default: 400).",
    )
    parser.add_argument("--test-limit", type=int, default=STANDARD_TEST_LIMIT)
    parser.add_argument(
        "--local-qwen",
        action="store_true",
        help="Deprecated alias for --mode local-qwen-vote.",
    )
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL, help="Local Qwen model directory.")
    parser.add_argument("--output", type=Path, default=Path("train/b_magent_training_report.json"))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible private-data sampling and agent scheduling.",
    )
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
        default=Path("data/lora_adapters_qwen2_5_vl_7b"),
        help="Directory for per-agent LoRA SFT datasets and adapters.",
    )
    parser.add_argument(
        "--lora-threshold",
        type=int,
        default=DEFAULT_TRAINING_LORA_THRESHOLD,
        help="Number of newly accepted samples accumulated per agent before one LoRA refresh.",
    )
    parser.add_argument("--lora-max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-train-batch-size", type=int, default=1)
    parser.add_argument("--lora-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-epochs", type=float, default=1.0)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-min-evaluation-score", type=float, default=0.6)
    parser.add_argument(
        "--lora-min-training-examples",
        type=int,
        default=16,
        help="Minimum curated examples required before creating or refreshing an adapter.",
    )
    parser.add_argument(
        "--llm-experience-tags",
        action="store_true",
        help=(
            "Use an additional model generation to classify each professional experience. "
            "By default, stable rule-based task tags are used to reduce training time."
        ),
    )
    parser.add_argument(
        "--allow-uncorrect-lora-labels",
        action="store_true",
        help="Allow LoRA SFT examples even when a gold final answer is present and the reflected answer does not match it.",
    )
    parser.add_argument(
        "--answer-validator",
        choices=["gpt-5.6-sol", "local"],
        default="local",
        help=(
            "Correctness judge for non-visual open-ended answers. InfographicVQA always uses "
            "its dataset answers locally; gpt-5.6-sol requires OPENAI_API_KEY."
        ),
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
    engine = LocalQwenEngine(
        model_name_or_path=args.model_path,
        system_prompt=GENERAL_TASK_INSTRUCTION,
    )
    return LocalQwenEvolutionBackend(
        engine,
        lora_output_dir=args.lora_output_dir if args.enable_lora else None,
        enable_llm_experience_tags=args.llm_experience_tags,
    )


def build_lora_manager(
    args: argparse.Namespace,
    backend: object | None = None,
) -> LoraEvolutionManager | None:
    if not args.enable_lora:
        return None
    config = LoraTrainingConfig(
        base_model_path=str(args.model_path),
        output_dir=args.lora_output_dir,
        threshold=args.lora_threshold,
        require_correct_answer=not args.allow_uncorrect_lora_labels,
        min_evaluation_score=args.lora_min_evaluation_score,
        min_training_examples=args.lora_min_training_examples,
        professional_library_dir=PROJECT_ROOT / "data",
        max_seq_length=args.lora_max_seq_length,
        per_device_train_batch_size=args.lora_train_batch_size,
        gradient_accumulation_steps=args.lora_gradient_accumulation_steps,
        num_train_epochs=args.lora_epochs,
        learning_rate=args.lora_learning_rate,
    )
    release_model_memory = getattr(backend, "release_model_memory", None)
    return LoraEvolutionManager(
        config,
        before_train=release_model_memory if callable(release_model_memory) else None,
    )


def print_training_round_start(round_index: int, rounds: int, question: str) -> None:
    preview = " ".join(question.split())[:120]
    print(f"[{round_index}/{rounds}] 开始六智能体训练: {preview}", flush=True)


def print_training_round_end(round_index: int, rounds: int, report: BMagentTrainingRound) -> None:
    print(
        f"[{round_index}/{rounds}] 完成: drafts={report.drafts} "
        f"evaluations={report.peer_reviews} professional_evolutions={report.self_improvements} "
        f"evaluation_evolutions={report.evaluation_evolutions} "
        f"global_downlinks={report.global_downlinks} global_uploads={report.global_uploads} "
        f"training_tag_updates={report.training_tag_updates}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    mode = "local-qwen-vote" if args.local_qwen else args.mode
    if mode == "b-magent":
        print(f"backend: {args.backend}", flush=True)
        print(f"model: {args.model_path}", flush=True)
        project_dataset = load_project_dataset(args.dataset_dir)
        if not project_dataset.load("train", limit=1):
            raise ValueError(
                f"no training samples found at {args.dataset_dir / 'train.jsonl'}; "
                "training state was not cleared"
            )
        start_round = 0
        if args.resume:
            start_round = load_training_progress(PROJECT_ROOT / "data")
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
                    PROJECT_ROOT / "train" / "six_agent_lora_voting_100_report.json",
                ),
            )
            print("已清空之前的训练存储", flush=True)
        print("开始训练", flush=True)
        backend = build_b_magent_backend(args)
        report = run_b_magent_training_entry(
            dataset_dir=args.dataset_dir,
            data_dir=PROJECT_ROOT / "data",
            rounds=args.rounds if args.rounds > 0 else None,
            private_batch_size=args.private_batch_size,
            private_train_size=args.private_train_size,
            random_seed=args.seed,
            backend=backend,
            lora_manager=build_lora_manager(args, backend),
            on_round_start=print_training_round_start,
            on_round_end=print_training_round_end,
            start_round=start_round,
            preserve_private_datasets=args.resume,
            answer_validator=(
                build_answer_validator_from_env()
                if args.answer_validator == "gpt-5.6-sol"
                and not isinstance(project_dataset, VisionQADataset)
                else None
            ),
        )
        export_json_report(report, args.output)
        print(f"b_magent agents: {', '.join(report.agents)}")
        print(f"rounds: {report.rounds}")
        print(
            f"global_experience_records={report.global_experience_records} "
            f"training_tag_records={report.training_tag_records}"
        )
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
        project_dataset = load_project_dataset(args.dataset_dir)
        models = build_six_local_qwen_agents(
            args.model_path,
            lora_output_dir=args.lora_output_dir,
            data_dir=PROJECT_ROOT / "data",
        )
        voting_report = run_six_agent_voting_on_test(
            args.dataset_dir,
            models,
            limit=args.test_limit,
            on_prediction=print_voting_prediction_detail,
            answer_validator=(
                build_answer_validator_from_env()
                if args.answer_validator == "gpt-5.6-sol"
                and not isinstance(project_dataset, VisionQADataset)
                else None
            ),
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
        report = run_six_agent_private_training(
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
