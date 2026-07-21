from __future__ import annotations

from pathlib import Path
from typing import Any

from b_magent.evaluation_format import format_structured_evaluation
from b_magent.library import EvolutionLibrary
from b_magent.models import EvaluationEvolution, GlobalExperience, LibraryRecord, PeerEvaluation


class ServerAgent:
    """Aggregate evaluator-side experience into a global evaluation library."""

    def __init__(
        self,
        name: str,
        data_dir: Path,
        backend: Any,
        specialty: str = "全局评价经验聚合服务器",
    ) -> None:
        self.name = name
        self.specialty = specialty
        self.backend = backend
        self.global_library = EvolutionLibrary(
            data_dir / name / "global_evaluation_library.jsonl",
            "global_evaluation",
        )

    def aggregate_evaluation_experience(
        self,
        task: str,
        peer_reviews: list[PeerEvaluation],
        evaluation_evolutions: list[EvaluationEvolution],
        evaluator_agents: list[Any] | None = None,
    ) -> GlobalExperience:
        consensus_reviews = select_consensus_peer_reviews(peer_reviews)
        consensus_targets = _unique(review.target for review in consensus_reviews)
        source_records = _select_consensus_evaluation_updates(
            evaluation_evolutions,
            consensus_evaluators=_unique(review.evaluator for review in consensus_reviews),
        )
        evaluator_names = _unique([review.evaluator for review in peer_reviews])
        if not consensus_targets or not source_records:
            return GlobalExperience(
                server_name=self.name,
                source_evaluators=evaluator_names,
                source_update_count=0,
                synthesized_experience=(
                    "No global upload: evaluator suggestions did not form a sufficiently similar "
                    "consensus for the same trained agent."
                ),
                global_updates=[],
            )
        prior_global_memory = [record.summary for record in self.global_library.search(task)]
        aggregate = getattr(self.backend, "aggregate_global_experience", None)
        if callable(aggregate):
            synthesized = aggregate(
                self.name,
                task,
                peer_reviews,
                evaluation_evolutions,
                source_records,
                prior_global_memory,
            )
        else:
            synthesized = _fallback_global_experience(
                task,
                peer_reviews,
                evaluation_evolutions,
                source_records,
                prior_global_memory,
            )

        detail = _build_global_detail(
            peer_reviews,
            evaluation_evolutions,
            source_records,
            consensus_targets,
            prior_global_memory,
            synthesized,
        )
        record = LibraryRecord(
            agent_name=self.name,
            library_type="global_evaluation",
            source_task=task,
            summary=synthesized,
            detail=detail,
            tags=[
                "server-agent",
                "global-evaluation",
                "trajectory-aggregation",
                *_keyword_tags(task, peer_reviews, source_records),
            ],
        )
        update = self.global_library.add_record(record)
        return GlobalExperience(
            server_name=self.name,
            source_evaluators=evaluator_names,
            source_update_count=len(source_records),
            synthesized_experience=synthesized,
            global_updates=[update],
        )


def _fallback_global_experience(
    task: str,
    peer_reviews: list[PeerEvaluation],
    evaluation_evolutions: list[EvaluationEvolution],
    source_records: list[LibraryRecord],
    prior_global_memory: list[str],
) -> str:
    suggestions = _unique(
        suggestion
        for review in peer_reviews
        for suggestion in review.suggestions
    )
    evaluator_lessons = _unique(record.summary for record in source_records)
    top_check = suggestions[0] if suggestions else "检查可观察答案结构、最终答案一致性和证据充分性"
    task_hint = " ".join(str(task).split())[:80] or "future tasks"
    return format_structured_evaluation(
        task=task_hint,
        observed_error=(
            "Evaluator trajectories show reusable review risks around answer structure, final answer consistency, "
            "and evidence sufficiency."
        ),
        evaluation_decision=(
            "Upload a global review lesson only after two evaluators give similar suggestions for the same trained agent."
        ),
        confidence=f"evaluator_updates={len(evaluator_lessons)}; prior_global_memory={len(prior_global_memory)}",
        improvement_pattern=(
            f"Aggregate evaluator trajectories before future reviews; prioritize {top_check}; compare evaluator "
            "rationales, target corrections, score patterns, and reusable review lessons."
        ),
    )


def _build_global_detail(
    peer_reviews: list[PeerEvaluation],
    evaluation_evolutions: list[EvaluationEvolution],
    source_records: list[LibraryRecord],
    consensus_targets: list[str],
    prior_global_memory: list[str],
    synthesized: str,
) -> str:
    review_summaries = [
        (
            f"evaluator={review.evaluator}, target={review.target}, "
            f"scores=({review.scores.correctness},{review.scores.safety},{review.scores.efficiency}), "
            f"suggestions={'; '.join(review.suggestions)}"
        )
        for review in peer_reviews
    ]
    evolution_summaries = [
        (
            f"evaluator={evolution.agent_name}, "
            f"updates={'; '.join(record.summary for record in evolution.evaluation_updates)}"
        )
        for evolution in evaluation_evolutions
    ]
    uploaded_record_summaries = [
        (
            f"agent={record.agent_name}, summary={record.summary}"
        )
        for record in source_records
    ]
    return format_structured_evaluation(
        task=f"consensus_targets={_summarize(consensus_targets)}",
        observed_error=(
            f"peer_review_trajectories={_summarize(review_summaries)} | "
            f"evaluator_evolved_experience={_summarize(evolution_summaries)}"
        ),
        evaluation_decision=(
            "Aggregate evaluator-side review trajectories only when two evaluators give similar suggestions "
            "for the same trained agent; otherwise keep evaluator experience private. "
            f"uploaded_consensus_evaluation_experience={_summarize(uploaded_record_summaries)}"
        ),
        confidence=f"prior_global_memory={_summarize(prior_global_memory)}",
        improvement_pattern=synthesized,
    )


def select_consensus_peer_reviews(peer_reviews: list[PeerEvaluation], threshold: float = 0.35) -> list[PeerEvaluation]:
    """Return only evaluator reviews that agree with another evaluator on the same target."""
    by_target: dict[str, list[PeerEvaluation]] = {}
    for review in peer_reviews:
        by_target.setdefault(review.target, []).append(review)
    consensus_reviews: list[PeerEvaluation] = []
    for reviews in by_target.values():
        distinct_reviews = _first_review_per_evaluator(reviews)
        if len(distinct_reviews) < 2:
            continue
        first, second = distinct_reviews[:2]
        if _suggestion_similarity(first.suggestions, second.suggestions) >= threshold:
            consensus_reviews.extend([first, second])
    return consensus_reviews


def _first_review_per_evaluator(reviews: list[PeerEvaluation]) -> list[PeerEvaluation]:
    seen: set[str] = set()
    distinct: list[PeerEvaluation] = []
    for review in reviews:
        if review.evaluator in seen:
            continue
        seen.add(review.evaluator)
        distinct.append(review)
    return distinct


def _suggestion_similarity(left: list[str], right: list[str]) -> float:
    left_tokens = _suggestion_tokens(left)
    right_tokens = _suggestion_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _suggestion_tokens(items: list[str]) -> set[str]:
    text = " ".join(str(item).lower() for item in items)
    ascii_tokens = {token for token in text.replace("，", " ").replace("。", " ").split() if len(token) > 2}
    cjk_chars = {char for char in text if "\u4e00" <= char <= "\u9fff"}
    return ascii_tokens | cjk_chars


def _select_consensus_evaluation_updates(
    evaluation_evolutions: list[EvaluationEvolution],
    consensus_evaluators: object,
) -> list[LibraryRecord]:
    consensus_evaluator_names = set(_unique(consensus_evaluators))
    records: list[LibraryRecord] = []
    for evolution in evaluation_evolutions:
        if evolution.agent_name not in consensus_evaluator_names:
            continue
        for record in evolution.evaluation_updates:
            if record.library_type != "evaluation":
                continue
            if "global-downlink" in record.tags:
                continue
            records.append(record)
    return records


def _unique(items: object) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _summarize(items: list[str], limit: int = 5) -> str:
    cleaned = _unique(items)
    if not cleaned:
        return "none"
    return " ; ".join(_shorten(item) for item in cleaned[:limit])


def _shorten(text: str, limit: int = 220) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _keyword_tags(
    task: str,
    peer_reviews: list[PeerEvaluation],
    source_records: list[LibraryRecord],
) -> list[str]:
    text = " ".join(
        [
            task,
            *(suggestion for review in peer_reviews for suggestion in review.suggestions),
            *(record.summary for record in source_records),
        ]
    ).lower()
    tags: list[str] = []
    candidates = {
        "final-answer": ("final", "answer", "####", "答案"),
        "verification": ("verify", "check", "验证", "检查", "自检"),
        "boundary": ("boundary", "edge", "condition", "边界", "条件"),
        "structure": ("step", "structure", "清单", "步骤", "编号"),
        "scoring": ("score", "correctness", "safety", "efficiency", "评分"),
    }
    for tag, needles in candidates.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags
