from __future__ import annotations

from .models import EvaluationScores


EVALUATION_SECTION_ORDER = (
    "Task",
    "Observed Error",
    "Evaluation Decision",
    "Confidence",
    "Improvement Pattern",
)


def format_structured_evaluation(
    *,
    task: str,
    observed_error: str,
    evaluation_decision: str,
    confidence: str | float,
    improvement_pattern: str,
) -> str:
    return "\n\n↓\n\n".join(
        [
            f"Task\n{_clean(task) or 'No task provided.'}",
            f"Observed Error\n{_clean(observed_error) or 'No observable error identified.'}",
            f"Evaluation Decision\n{_clean(evaluation_decision) or 'No evaluation decision recorded.'}",
            f"Confidence\n{_format_confidence(confidence)}",
            f"Improvement Pattern\n{_clean(improvement_pattern) or 'No improvement pattern recorded.'}",
        ]
    )


def format_confidence_from_scores(scores: EvaluationScores) -> str:
    average = (scores.correctness + scores.safety + scores.efficiency) / 3
    return f"{average:.2f}"


def _format_confidence(confidence: str | float) -> str:
    if isinstance(confidence, float):
        return f"{confidence:.2f}"
    return _clean(confidence) or "unknown"


def _clean(text: object) -> str:
    return " ".join(str(text).split())
