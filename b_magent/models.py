from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class LibraryRecord:
    agent_name: str
    library_type: str
    source_task: str
    summary: str
    detail: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LibraryRecord":
        return cls(
            agent_name=str(payload.get("agent_name", "")),
            library_type=str(payload.get("library_type", "")),
            source_task=str(payload.get("source_task", "")),
            summary=str(payload.get("summary", "")),
            detail=str(payload.get("detail", "")),
            tags=[str(tag) for tag in payload.get("tags", [])],
            created_at=str(payload.get("created_at") or utc_now()),
        )


@dataclass
class Draft:
    agent_name: str
    specialty: str
    answer: str
    thought_trace: list[str]
    private_training_used: list[str]
    professional_memory_used: list[str]
    evaluation_alerts_used: list[str]


@dataclass
class PeerReview:
    reviewer: str
    target: str
    suggestions: list[str]
    rationale: str
    evaluation_memory_used: list[str]


@dataclass
class SelfImprovement:
    agent_name: str
    applied_suggestions: list[str]
    revised_answer: str
    professional_updates: list[LibraryRecord]


@dataclass
class EvaluationEvolution:
    agent_name: str
    synthesized_suggestions: list[str]
    evaluation_updates: list[LibraryRecord]


@dataclass
class EvolutionReport:
    task: str
    participants: list[str]
    evaluators: list[str]
    drafts: list[Draft]
    peer_reviews: list[PeerReview]
    self_improvements: list[SelfImprovement]
    evaluation_evolutions: list[EvaluationEvolution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def reflections(self) -> list[EvaluationEvolution]:
        return self.evaluation_evolutions

