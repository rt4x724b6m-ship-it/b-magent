from __future__ import annotations

from pathlib import Path

from .backend import DemoQwenBackend
from .datasets import GSM8KDataset
from .library import EvolutionLibrary
from .models import Draft, EvaluationEvolution, LibraryRecord, PeerReview, SelfImprovement
from .self_evolution import EvolutionInput, SelfEvolutionLibrary


class QwenAgent:
    def __init__(
        self,
        name: str,
        specialty: str,
        data_dir: Path,
        backend: DemoQwenBackend | None = None,
    ) -> None:
        self.name = name
        self.specialty = specialty
        self.data_dir = data_dir
        self.backend = backend or DemoQwenBackend()
        self.professional_library = EvolutionLibrary(
            data_dir / name / "professional_library.jsonl",
            "professional",
        )
        self.evaluation_library = EvolutionLibrary(
            data_dir / name / "evaluation_library.jsonl",
            "evaluation",
        )
        self.self_evolution_library = SelfEvolutionLibrary(data_dir, name)

    def train_private_data(self, task: str) -> list[str]:
        private_items = self._load_private_data()
        record = LibraryRecord(
            agent_name=self.name,
            library_type="professional",
            source_task=task,
            summary=f"{self.specialty} private training summary",
            detail=" | ".join(private_items),
            tags=[self.specialty, "private-training"],
        )
        self.professional_library.add_record(record)
        return private_items

    def solve_task(self, task: str, private_training: list[str]) -> Draft:
        professional_records = self.professional_library.search(task)
        evaluation_records = self.evaluation_library.search(task)
        professional_memory = [record.summary for record in professional_records]
        evaluation_alerts = [record.summary for record in evaluation_records]
        answer, thought_trace = self.backend.solve(
            self.name,
            self.specialty,
            task,
            private_training,
            professional_memory,
            evaluation_alerts,
        )
        return Draft(
            agent_name=self.name,
            specialty=self.specialty,
            answer=answer,
            thought_trace=thought_trace,
            private_training_used=private_training,
            professional_memory_used=professional_memory,
            evaluation_alerts_used=evaluation_alerts,
        )

    def review_peer(self, task: str, draft: Draft) -> PeerReview:
        evaluation_records = self.evaluation_library.search(task)
        evaluation_memory = [record.summary for record in evaluation_records]
        return self.backend.suggest_improvements(self.name, draft, task, evaluation_memory)

    def self_improve(self, task: str, draft: Draft, reviews: list[PeerReview]) -> SelfImprovement:
        suggestions = _unique(item for review in reviews for item in review.suggestions)
        revised_answer = draft.answer
        if suggestions:
            revised_answer += "\nSelf-evolution revisions:\n" + "\n".join(f"- {item}" for item in suggestions)
        event = EvolutionInput(
            agent_name=self.name,
            specialty=self.specialty,
            task=task,
            answer=draft.answer,
            thought_trace=draft.thought_trace,
            peer_suggestions=suggestions,
        )
        update = self.self_evolution_library.evolve_professional(event)
        return SelfImprovement(
            agent_name=self.name,
            applied_suggestions=suggestions,
            revised_answer=revised_answer,
            professional_updates=[update],
        )

    def evolve_evaluation_library(self, task: str, all_reviews: list[PeerReview]) -> EvaluationEvolution:
        suggestions = _unique(item for review in all_reviews for item in review.suggestions)
        event = EvolutionInput(
            agent_name=self.name,
            specialty=self.specialty,
            task=task,
            answer="",
            evaluator_suggestions=suggestions,
        )
        update = self.self_evolution_library.evolve_evaluation(event)
        return EvaluationEvolution(
            agent_name=self.name,
            synthesized_suggestions=suggestions,
            evaluation_updates=[update],
        )

    def _load_private_data(self) -> list[str]:
        agent_jsonl = self.data_dir / self.name / "private_data.jsonl"
        agent_text = self.data_dir / self.name / "private_data.txt"
        if agent_jsonl.exists():
            return [line.strip() for line in agent_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if agent_text.exists():
            return [line.strip() for line in agent_text.read_text(encoding="utf-8").splitlines() if line.strip()]

        gsm8k_items = self._load_gsm8k_private_data()
        if gsm8k_items:
            return gsm8k_items

        return [
            f"{self.specialty} private sample: keep reusable solving strategy",
            f"{self.specialty} private sample: make the answer easy for evaluators to revise",
        ]

    def _load_gsm8k_private_data(self) -> list[str]:
        dataset = GSM8KDataset(self.data_dir / "gsm8k")
        samples = dataset.load(split="train", limit=3)
        return [sample.to_training_text() for sample in samples]


def _unique(items: object) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in result:
            result.append(text)
    return result
