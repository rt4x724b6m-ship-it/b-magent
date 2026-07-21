from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .evaluation_format import format_structured_evaluation
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
    peer_evaluation_rationales: list[str] = field(default_factory=list)
    is_correct: bool | None = None


@dataclass
class EvolutionResult:
    agent_name: str
    professional_record: LibraryRecord | None
    evaluation_record: LibraryRecord | None


class SelfEvolutionLibrary:
    """Private dual-library evolution store for one agent.

    The professional library stores evaluator-selected successful solving
    experience plus reflective lessons from evaluated wrong solutions.
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
        rationale_summary = _summarize_list(
            event.peer_evaluation_rationales,
            fallback="No peer evaluation rationale provided",
        )
        score_summary = _summarize_list(event.evaluation_scores, fallback="No peer evaluation scores recorded")
        professional_lesson = _build_professional_lesson(event, suggestions)
        experience_kind = _professional_experience_kind(event.is_correct)
        record = LibraryRecord(
            agent_name=event.agent_name,
            library_type="professional",
            source_task=event.task,
            summary=professional_lesson,
            detail=(
                f"experience_kind={experience_kind} | "
                f"answer_snapshot={_shorten(event.answer)} | "
                f"thought_trace={thought_summary} | "
                f"peer_suggestions={suggestion_summary} | "
                f"peer_evaluation_rationales={rationale_summary} | "
                f"peer_evaluation_scores={score_summary} | "
                f"future_solving_lesson={professional_lesson}"
            ),
            tags=[
                event.specialty,
                "self-evolution",
                "professional",
                experience_kind,
                *_keyword_tags(event.task, suggestions),
            ],
        )
        return self.professional.add_record(record)

    def evolve_evaluation(self, event: EvolutionInput) -> LibraryRecord:
        suggestions = _unique(event.evaluator_suggestions or event.peer_suggestions)
        suggestion_summary = _summarize_list(suggestions, fallback="No evaluator suggestions provided")
        rationale_summary = _summarize_list(event.evaluator_rationales, fallback="No evaluator rationale provided")
        memory_summary = _summarize_list(event.evaluation_memory_used, fallback="No prior evaluation memory retrieved")
        score_summary = _summarize_list(event.evaluation_scores, fallback="No evaluation scores recorded")
        evaluation_lesson = _build_evaluation_lesson(event, suggestions)
        structured_detail = format_structured_evaluation(
            task=event.task,
            observed_error=(
                f"own_review_suggestions={suggestion_summary} | "
                f"own_review_rationales={rationale_summary}"
            ),
            evaluation_decision=(
                "Reflect on this agent's own peer reviews after the reviewed agents receive feedback; "
                f"review_scores_peer_comparisons_and_target_results={score_summary}"
            ),
            confidence=f"prior_evaluation_memory={memory_summary}",
            improvement_pattern=f"future_review_lesson={evaluation_lesson}",
        )
        record = LibraryRecord(
            agent_name=event.agent_name,
            library_type="evaluation",
            source_task=event.task,
            summary=evaluation_lesson,
            detail=structured_detail,
            tags=[event.specialty, "self-evolution", "evaluation", *_keyword_tags(event.task, suggestions)],
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


def _build_professional_lesson(event: EvolutionInput, suggestions: list[str]) -> str:
    top_suggestion = _shorten(suggestions[0], limit=90) if suggestions else "make the answer concrete and checkable"
    task_hint = _task_hint(event.task)
    outcome = _outcome_label(event.is_correct)
    basis = _professional_basis(event.is_correct)
    return (
        f"{event.specialty} {outcome} {basis} solving lesson for {task_hint}: "
        f"before finalizing, {top_suggestion}; keep steps explicit and verify the final answer."
    )


def _build_evaluation_lesson(event: EvolutionInput, suggestions: list[str]) -> str:
    top_check = _shorten(suggestions[0], limit=90) if suggestions else "check correctness, safety, efficiency, and missing evidence"
    task_hint = _task_hint(event.task)
    outcome = _outcome_label(event.is_correct)
    return format_structured_evaluation(
        task=task_hint,
        observed_error=f"{event.specialty} evaluator outcome is {outcome}; review may miss {top_check}.",
        evaluation_decision="Evaluate observable answer structure, final-answer consistency, and evidence sufficiency.",
        confidence=_summarize_list(event.evaluation_scores, fallback="No evaluation scores recorded"),
        improvement_pattern=f"review lesson: give concrete fixes tied to scores; prioritize {top_check}.",
    )


def _outcome_label(is_correct: bool | None) -> str:
    if is_correct is True:
        return "success"
    if is_correct is False:
        return "error"
    return "uncertain"


def _professional_basis(is_correct: bool | None) -> str:
    if is_correct is True:
        return "curated-by-evaluation"
    if is_correct is False:
        return "reflection-from-error"
    return "evaluated"


def _professional_experience_kind(is_correct: bool | None) -> str:
    if is_correct is True:
        return "curated-success-experience"
    if is_correct is False:
        return "error-reflection-experience"
    return "evaluated-experience"


def _task_hint(task: str) -> str:
    cleaned = " ".join(str(task).split())
    if not cleaned:
        return "future similar tasks"
    return _shorten(cleaned, limit=80)


def _keyword_tags(task: str, suggestions: list[str]) -> list[str]:
    text = " ".join([task, *suggestions]).lower()
    candidates = {
        "arithmetic": ("arithmetic", "numeric", "calculation", "math", "算", "数字"),
        "final-answer": ("final", "answer", "####", "答案"),
        "verification": ("verify", "check", "验证", "检查", "自检"),
        "boundary": ("boundary", "edge", "condition", "边界", "条件"),
        "structure": ("step", "structure", "清单", "步骤", "编号"),
    }
    tags: list[str] = []
    for tag, needles in candidates.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags
