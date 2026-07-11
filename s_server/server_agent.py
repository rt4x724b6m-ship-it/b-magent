from __future__ import annotations

from pathlib import Path
from typing import Any

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
        consensus_targets = _consensus_targets(peer_reviews)
        source_records = _select_consensus_evaluation_updates(
            evaluation_evolutions,
            consensus_evaluators=_consensus_evaluators(peer_reviews, consensus_targets),
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
    return (
        f"Global review lesson for {task_hint}: aggregate evaluator trajectories before future reviews; "
        f"prioritize {top_check}; compare evaluator rationales, target corrections, score patterns, "
        f"and reusable review lessons from {len(evaluator_lessons)} evaluator updates. "
        f"Prior global memory count: {len(prior_global_memory)}."
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
    return (
        "Aggregate evaluator-side review trajectories only when two evaluators give similar suggestions "
        "for the same trained agent; otherwise keep evaluator experience private. "
        f"consensus_targets={_summarize(consensus_targets)} | "
        f"prior_global_memory={_summarize(prior_global_memory)} | "
        f"peer_review_trajectories={_summarize(review_summaries)} | "
        f"uploaded_consensus_evaluation_experience={_summarize(uploaded_record_summaries)} | "
        f"evaluator_evolved_experience={_summarize(evolution_summaries)} | "
        f"global_future_review_lesson={synthesized}"
    )


def _consensus_targets(peer_reviews: list[PeerEvaluation], threshold: float = 0.35) -> list[str]:
    by_target: dict[str, list[PeerEvaluation]] = {}
    for review in peer_reviews:
        by_target.setdefault(review.target, []).append(review)
    targets: list[str] = []
    for target, reviews in by_target.items():
        if len({review.evaluator for review in reviews}) < 2:
            continue
        first, second = reviews[:2]
        if _suggestion_similarity(first.suggestions, second.suggestions) >= threshold:
            targets.append(target)
    return targets


def _consensus_evaluators(peer_reviews: list[PeerEvaluation], consensus_targets: list[str]) -> set[str]:
    targets = set(consensus_targets)
    return {
        review.evaluator
        for review in peer_reviews
        if review.target in targets
    }


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
    consensus_evaluators: set[str],
) -> list[LibraryRecord]:
    records: list[LibraryRecord] = []
    for evolution in evaluation_evolutions:
        if evolution.agent_name not in consensus_evaluators:
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
