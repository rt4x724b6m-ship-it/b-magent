from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any
from .answer_validation import AnswerValidator

from .agent import QwenAgent
from .backend import DemoQwenBackend
from .models import Draft, EvolutionReport, LibraryRecord, PeerEvaluation
from .lora import is_improved_answer_correct
from s_server import ServerAgent
from s_server.server_agent import select_consensus_peer_reviews


def build_default_agents(
    base_dir: Path | None = None,
    backend: Any | None = None,
    answer_validator: AnswerValidator | None = None,
) -> list[QwenAgent]:
    root = base_dir or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    backend = backend or DemoQwenBackend()
    return [
        QwenAgent("qwen_agent_1", "通用智能体", data_dir, backend, answer_validator),
        QwenAgent("qwen_agent_2", "通用智能体", data_dir, backend, answer_validator),
        QwenAgent("qwen_agent_3", "通用智能体", data_dir, backend, answer_validator),
        QwenAgent("qwen_agent_4", "通用智能体", data_dir, backend, answer_validator),
    ]


def build_default_server_agent(base_dir: Path | None = None, backend: Any | None = None) -> ServerAgent:
    root = base_dir or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    return ServerAgent("qwen_server_agent", data_dir, backend or DemoQwenBackend())


class MultiAgentWorkflow:
    def __init__(
        self,
        agents: list[QwenAgent],
        server_agent: ServerAgent | None = None,
        random_seed: int | None = None,
        private_batch_size: int | None = None,
    ) -> None:
        if len(agents) != 4:
            raise ValueError("b_magent training requires exactly four agents")
        self.agents = agents
        backend = agents[0].backend if agents else DemoQwenBackend()
        data_dir = agents[0].data_dir if agents else Path(__file__).resolve().parent.parent / "data"
        self.server_agent = server_agent or ServerAgent("qwen_server_agent", data_dir, backend)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.private_batch_size = private_batch_size

    def run(self, task: str, participant_names: list[str] | None = None) -> EvolutionReport:
        participants = self._select_participants(participant_names)
        participant_names = {agent.name for agent in participants}
        evaluators = [agent for agent in self.agents if agent.name not in participant_names]

        drafts: list[Draft] = []
        for agent in participants:
            private_training = agent.train_private_data(task, batch_size=self.private_batch_size)
            drafts.append(agent.solve_task(task, private_training))

        peer_reviews: list[PeerEvaluation] = []
        for evaluator in evaluators:
            for draft in drafts:
                if evaluator.name == draft.agent_name:
                    continue
                peer_reviews.append(evaluator.evaluate_peer(task, draft))

        # During supervised training, gold labels are the authority for the
        # correctness component. Evaluators still own safety, efficiency, and
        # qualitative feedback, but a weak judge must not invert known labels.
        if "Image:" in task:
            drafts_by_agent = {draft.agent_name: draft for draft in drafts}
            for review in peer_reviews:
                target_draft = drafts_by_agent.get(review.target)
                if target_draft is None:
                    continue
                gold_correct = is_improved_answer_correct(task, target_draft.answer)
                if gold_correct is not None:
                    review.scores.correctness = 1.0 if gold_correct else 0.0

        visual_completeness_records = [
            record
            for draft in drafts
            if (record := _build_visual_completeness_record(task, draft)) is not None
        ]

        self_improvements = []
        for participant in participants:
            draft = next(item for item in drafts if item.agent_name == participant.name)
            reviews_for_agent = [review for review in peer_reviews if review.target == participant.name]
            self_improvements.append(participant.self_improve(task, draft, reviews_for_agent))

        consensus_reviews = select_consensus_peer_reviews(peer_reviews)
        evaluation_evolutions = [
            evaluator.evolve_evaluation_library(
                task,
                [review for review in consensus_reviews if review.evaluator == evaluator.name],
                consensus_reviews,
                self_improvements,
            )
            for evaluator in evaluators
        ]
        global_experience = self.server_agent.aggregate_evaluation_experience(
            task,
            peer_reviews,
            evaluation_evolutions,
            evaluators,
        )
        training_tag_records = [
            record
            for record in (
                [agent.last_private_training_record for agent in participants]
                + [
                    update
                    for improvement in self_improvements
                    for update in improvement.professional_updates
                ]
                + [
                    update
                    for evolution in evaluation_evolutions
                    for update in evolution.evaluation_updates
                ]
                + visual_completeness_records
            )
            if record is not None
        ]
        server_training_tag_updates = self.server_agent.store_agent_training_tags(task, training_tag_records)

        return EvolutionReport(
            task=task,
            participants=[agent.name for agent in participants],
            evaluators=[agent.name for agent in evaluators],
            drafts=drafts,
            peer_reviews=peer_reviews,
            self_improvements=self_improvements,
            evaluation_evolutions=evaluation_evolutions,
            global_experience=global_experience,
            server_training_tag_updates=server_training_tag_updates,
        )

    def _select_participants(self, participant_names: list[str] | None) -> list[QwenAgent]:
        if participant_names is None:
            return self._rng.sample(self.agents, 2)
        if len(participant_names) != 2:
            raise ValueError("each workflow round requires exactly two participant names")
        agents_by_name = {agent.name: agent for agent in self.agents}
        missing = [name for name in participant_names if name not in agents_by_name]
        if missing:
            raise ValueError(f"unknown participant agents: {', '.join(missing)}")
        if len(set(participant_names)) != 2:
            raise ValueError("participant names must be distinct")
        return [agents_by_name[name] for name in participant_names]

    def export_report(self, report: EvolutionReport, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _build_visual_completeness_record(task: str, draft: Draft) -> LibraryRecord | None:
    gold_elements = _extract_gold_image_elements(task)
    if not gold_elements:
        return None
    predicted = _parse_answer_payload(draft.answer)
    recognized = predicted.get("recognized_elements") or predicted.get("image_elements") or {}
    if not isinstance(recognized, dict):
        recognized = {}

    ocr_coverage = _token_recall(gold_elements.get("visible_text", []), recognized.get("visible_text", []))
    object_coverage = _token_recall(gold_elements.get("objects", []), recognized.get("objects", []))
    gold_layout = [gold_elements.get("summary", ""), gold_elements.get("layout", ""), *gold_elements.get("colors", [])]
    predicted_layout = [recognized.get("summary", ""), recognized.get("layout", ""), *recognized.get("colors", [])]
    layout_coverage = _token_recall(gold_layout, predicted_layout)
    completeness = 0.45 * ocr_coverage + 0.35 * object_coverage + 0.20 * layout_coverage
    detail = (
        f"visual_completeness_score={completeness:.4f} | "
        f"ocr_coverage_score={ocr_coverage:.4f} | "
        f"object_coverage_score={object_coverage:.4f} | "
        f"layout_coverage_score={layout_coverage:.4f}"
    )
    return LibraryRecord(
        agent_name=draft.agent_name,
        library_type="visual_completeness",
        source_task=task,
        summary=f"{draft.agent_name} full-image recognition coverage: {completeness:.4f}",
        detail=detail,
        tags=[
            "visual-content-completeness",
            "ocr-coverage",
            "object-coverage",
            "layout-coverage",
        ],
    )


def _extract_gold_image_elements(task: str) -> dict[str, Any]:
    marker = "Gold image elements:"
    if marker not in task:
        return {}
    candidate = task.split(marker, 1)[1].lstrip()
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate)
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_answer_payload(answer: str) -> dict[str, Any]:
    candidate = str(answer).strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _token_recall(expected: object, predicted: object) -> float:
    expected_tokens = _coverage_tokens(expected)
    if not expected_tokens:
        return 1.0
    predicted_tokens = _coverage_tokens(predicted)
    return len(expected_tokens & predicted_tokens) / len(expected_tokens)


def _coverage_tokens(value: object) -> set[str]:
    if isinstance(value, (list, tuple)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    ascii_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    cjk_tokens = {character for character in text if "\u4e00" <= character <= "\u9fff"}
    return ascii_tokens | cjk_tokens
