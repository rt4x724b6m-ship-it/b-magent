from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .library import EvolutionLibrary
from .models import LibraryRecord


@dataclass
class EvolutionInput:
    agent_name: str
    specialty: str
    task: str
    answer: str
    thought_trace: list[str] = field(default_factory=list)
    peer_suggestions: list[str] = field(default_factory=list)
    evaluator_suggestions: list[str] = field(default_factory=list)
    evaluator_rationales: list[str] = field(default_factory=list)
    evaluation_memory_used: list[str] = field(default_factory=list)
    evaluation_scores: list[str] = field(default_factory=list)


@dataclass
class EvolutionResult:
    agent_name: str
    professional_record: LibraryRecord | None
    evaluation_record: LibraryRecord | None


class SelfEvolutionLibrary:
    """Private dual-library evolution store for one agent.

    The professional library stores lessons that improve task solving.
    The evaluation library stores lessons that improve future reviewing.
    Both are JSONL-backed and intentionally separate.
    """

    def __init__(self, data_dir: Path, agent_name: str) -> None:
        self.data_dir = data_dir
        self.agent_name = agent_name
        self.professional = EvolutionLibrary(
            data_dir / agent_name / "professional_library.jsonl",
            "professional",
        )
        self.evaluation = EvolutionLibrary(
            data_dir / agent_name / "evaluation_library.jsonl",
            "evaluation",
        )

    def evolve_from_round(self, event: EvolutionInput) -> EvolutionResult:
        professional_record = self.evolve_professional(event)
        evaluation_record = self.evolve_evaluation(event)
        return EvolutionResult(
            agent_name=event.agent_name,
            professional_record=professional_record,
            evaluation_record=evaluation_record,
        )

    def evolve_professional(self, event: EvolutionInput) -> LibraryRecord:
        suggestions = _unique(event.peer_suggestions)
        thought_summary = _summarize_list(event.thought_trace, fallback="No thought trace provided")
        suggestion_summary = _summarize_list(suggestions, fallback="No peer suggestions provided")
        record = LibraryRecord(
            agent_name=event.agent_name,
            library_type="professional",
            source_task=event.task,
            summary=f"{event.specialty} professional update from self-evolution round",
            detail=(
                f"answer_snapshot={_shorten(event.answer)} | "
                f"thought_trace={thought_summary} | "
                f"peer_suggestions={suggestion_summary}"
            ),
            tags=[event.specialty, "self-evolution", "professional"],
        )
        return self.professional.add_record(record)

    def evolve_evaluation(self, event: EvolutionInput) -> LibraryRecord:
        suggestions = _unique(event.evaluator_suggestions or event.peer_suggestions)
        suggestion_summary = _summarize_list(suggestions, fallback="No evaluator suggestions provided")
        rationale_summary = _summarize_list(event.evaluator_rationales, fallback="No evaluator rationale provided")
        memory_summary = _summarize_list(event.evaluation_memory_used, fallback="No prior evaluation memory retrieved")
        score_summary = _summarize_list(event.evaluation_scores, fallback="No evaluation scores recorded")
        record = LibraryRecord(
            agent_name=event.agent_name,
            library_type="evaluation",
            source_task=event.task,
            summary=f"{event.specialty} evaluation self-reflection from own reviews",
            detail=(
                "Reflect on this agent's own peer reviews before updating future review behavior. "
                f"prior_evaluation_memory={memory_summary} | "
                f"own_review_suggestions={suggestion_summary} | "
                f"own_review_rationales={rationale_summary} | "
                f"own_review_scores={score_summary} | "
                "future_review_lesson=Compare the new draft against retrieved evaluation memories, "
                "then give concrete, checkable, and score-consistent feedback."
            ),
            tags=[event.specialty, "self-evolution", "evaluation"],
        )
        return self.evaluation.add_record(record)

    def search_professional(self, query: str, limit: int = 3) -> list[LibraryRecord]:
        return self.professional.search(query, limit=limit)

    def search_evaluation(self, query: str, limit: int = 3) -> list[LibraryRecord]:
        return self.evaluation.search(query, limit=limit)


def evolve_all_agents(
    data_dir: Path,
    events: list[EvolutionInput],
) -> list[EvolutionResult]:
    results: list[EvolutionResult] = []
    for event in events:
        library = SelfEvolutionLibrary(data_dir, event.agent_name)
        results.append(library.evolve_from_round(event))
    return results


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _summarize_list(items: list[str], fallback: str) -> str:
    cleaned = _unique(items)
    if not cleaned:
        return fallback
    return " ; ".join(_shorten(item, limit=180) for item in cleaned[:5])


def _shorten(text: str, limit: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
