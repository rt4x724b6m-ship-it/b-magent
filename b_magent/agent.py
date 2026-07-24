from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .backend import DemoQwenBackend
from .datasets import GSM8KDataset
from .library import EvolutionLibrary
from .models import Draft, EvaluationEvolution, LibraryRecord, PeerEvaluation, SelfImprovement
from .self_evolution import EvolutionInput, SelfEvolutionLibrary
from .tagging import extract_math_task_tags
from .trajectory import extract_answer_features, mask_draft_for_evaluation


class QwenAgent:
    def __init__(
        self,
        name: str,
        specialty: str,
        data_dir: Path,
        backend: Any | None = None,
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
        self.last_private_training_record: LibraryRecord | None = None
        self._private_cursor = 0

    def train_private_data(self, task: str, batch_size: int | None = None) -> list[str]:
        private_items = self._load_private_data()
        training_batch = self._next_private_batch(private_items, batch_size)
        reflection = _build_private_training_reflection(self.specialty, training_batch)
        learned_tags = sorted(extract_math_task_tags("\n".join([task, *training_batch])))
        record = LibraryRecord(
            agent_name=self.name,
            library_type="professional",
            source_task=task,
            summary=reflection,
            detail=(
                f"reflection={reflection} | "
                f"private_sample_count={len(training_batch)} | "
                f"sample_summaries={_summarize_texts(training_batch)}"
            ),
            tags=[self.specialty, "private-training", "professional", "reflection", *learned_tags],
        )
        self.last_private_training_record = self.professional_library.add_record(record)
        return training_batch

    def solve_task(self, task: str, private_training: list[str]) -> Draft:
        professional_records = self.professional_library.search(
            task,
            exclude_tags={"error-reflection-experience", "evaluated-experience"},
        )
        evaluation_records = self.evaluation_library.search(
            task,
            exclude_tags={"error-evaluation-experience"},
        )
        professional_memory = [record.summary for record in professional_records]
        evaluation_alerts = [record.summary for record in evaluation_records]
        visible_task = _strip_gold_annotations(task)
        answer, thought_trace = self.backend.solve(
            self.name,
            self.specialty,
            visible_task,
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
            tool_calls=[
                f"load_private_data(count={len(private_training)})",
                f"search_professional_library(count={len(professional_memory)})",
                f"search_evaluation_library(count={len(evaluation_alerts)})",
            ],
        )

    def evaluate_peer(self, task: str, draft: Draft) -> PeerEvaluation:
        evaluation_records = self.evaluation_library.search(
            task,
            exclude_tags={"error-evaluation-experience"},
        )
        evaluation_memory = [record.summary for record in evaluation_records]
        masked_draft = mask_draft_for_evaluation(draft)
        return self.backend.suggest_improvements(self.name, masked_draft, _strip_gold_annotations(task), evaluation_memory)

    def self_improve(self, task: str, draft: Draft, evaluations: list[PeerEvaluation]) -> SelfImprovement:
        suggestions = _unique(item for evaluation in evaluations for item in evaluation.suggestions)
        peer_rationales = _unique(evaluation.rationale for evaluation in evaluations)
        peer_scores = [
            (
                f"evaluator={evaluation.evaluator}: "
                f"correctness={evaluation.scores.correctness}, "
                f"safety={evaluation.scores.safety}, "
                f"efficiency={evaluation.scores.efficiency}"
            )
            for evaluation in evaluations
        ]
        professional_memory = [
            record.summary
            for record in self.professional_library.search(
                task,
                exclude_tags={"error-reflection-experience", "evaluated-experience"},
            )
        ]
        evaluation_alerts = [
            record.summary
            for record in self.evaluation_library.search(
                task,
                exclude_tags={"error-evaluation-experience"},
            )
        ]
        improve_answer = getattr(self.backend, "improve_answer", None)
        if callable(improve_answer):
            revised_answer, reflection = improve_answer(
                self.name,
                self.specialty,
                _strip_gold_annotations(task),
                draft,
                suggestions,
                professional_memory,
                evaluation_alerts,
            )
        else:
            revised_answer = draft.answer
            if suggestions:
                revised_answer += "\nSelf-evolution revisions:\n" + "\n".join(f"- {item}" for item in suggestions)
            reflection = (
                "Reflection: reviewed evaluator feedback and converted concrete suggestions "
                "into an improved answer."
            )
        gold_answer = _extract_gold_final_answer(task)
        is_correct = None
        if gold_answer is not None:
            is_correct = _extract_final_answer(revised_answer) == gold_answer
        experience_tags: list[str] = []
        generate_experience_tags = getattr(self.backend, "generate_experience_tags", None)
        if callable(generate_experience_tags):
            experience_tags = generate_experience_tags(
                self.name,
                self.specialty,
                _strip_gold_annotations(task),
                draft.answer,
                revised_answer,
                suggestions,
                reflection,
            )
        event = EvolutionInput(
            agent_name=self.name,
            specialty=self.specialty,
            task=task,
            answer=draft.answer,
            thought_trace=draft.thought_trace,
            peer_suggestions=suggestions,
            peer_evaluation_rationales=peer_rationales,
            evaluation_scores=peer_scores,
            reflection=reflection,
            is_correct=is_correct,
            experience_tags=experience_tags,
        )
        update = self.self_evolution_library.evolve_professional(event)
        return SelfImprovement(
            agent_name=self.name,
            applied_suggestions=suggestions,
            revised_answer=revised_answer,
            professional_updates=[update],
            reflection=reflection,
            is_correct=is_correct,
        )

    def evolve_evaluation_library(
        self,
        task: str,
        own_evaluations: list[PeerEvaluation],
        all_evaluations: list[PeerEvaluation] | None = None,
        self_improvements: list[SelfImprovement] | None = None,
    ) -> EvaluationEvolution:
        if not own_evaluations:
            return EvaluationEvolution(
                agent_name=self.name,
                synthesized_suggestions=[],
                evaluation_updates=[],
            )
        all_evaluations = all_evaluations or own_evaluations
        self_improvements = self_improvements or []
        target_improvements = {item.agent_name: item for item in self_improvements}
        suggestions = _unique(item for evaluation in own_evaluations for item in evaluation.suggestions)
        rationales = _unique(evaluation.rationale for evaluation in own_evaluations)
        evaluation_memory_used = _unique(
            item
            for evaluation in own_evaluations
            for item in evaluation.evaluation_memory_used
        )
        score_reflections = [
            (
                f"target={evaluation.target}: "
                f"correctness={evaluation.scores.correctness}, "
                f"safety={evaluation.scores.safety}, "
                f"efficiency={evaluation.scores.efficiency}"
            )
            for evaluation in own_evaluations
        ]
        peer_comparisons = [
            (
                f"target={evaluation.target}, evaluator={evaluation.evaluator}: "
                f"suggestions={'; '.join(evaluation.suggestions)}, rationale={evaluation.rationale}"
            )
            for own_review in own_evaluations
            for evaluation in all_evaluations
            if evaluation.target == own_review.target and evaluation.evaluator != self.name
        ]
        target_results = [
            (
                f"target={own_review.target}: "
                f"revised_answer_summary={_summarize_answer_for_evaluation(target_improvements[own_review.target].revised_answer)}, "
                f"is_correct={target_improvements[own_review.target].is_correct}, "
                f"applied_suggestions={'; '.join(target_improvements[own_review.target].applied_suggestions)}"
            )
            for own_review in own_evaluations
            if own_review.target in target_improvements
        ]
        event = EvolutionInput(
            agent_name=self.name,
            specialty=self.specialty,
            task=task,
            answer="",
            evaluator_suggestions=suggestions,
            evaluator_rationales=rationales,
            evaluation_memory_used=evaluation_memory_used,
            evaluation_scores=score_reflections + peer_comparisons + target_results,
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

    def _next_private_batch(self, private_items: list[str], batch_size: int | None) -> list[str]:
        if batch_size is None or batch_size <= 0 or batch_size >= len(private_items):
            return private_items
        start = self._private_cursor
        batch = [
            private_items[(start + offset) % len(private_items)]
            for offset in range(batch_size)
        ]
        self._private_cursor = (start + batch_size) % len(private_items)
        return batch

    def restore_private_cursor(self, batch_size: int) -> None:
        consumed_batches = sum(
            1
            for record in self.professional_library.all_records()
            if "private-training" in record.tags
        )
        self._private_cursor = consumed_batches * batch_size


def _unique(items: object) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in result:
            result.append(text)
    return result


def _extract_gold_final_answer(task: str) -> str | None:
    match = re.search(r"Gold final answer:\s*([^\n]+)", task)
    if not match:
        return None
    return _normalize_answer(match.group(1))


def _extract_final_answer(text: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return _normalize_answer(matches[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return _normalize_answer(numbers[-1]) if numbers else ""


def _normalize_answer(text: str) -> str:
    cleaned = str(text).strip().replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def _strip_gold_annotations(task: str) -> str:
    lines = []
    in_gold_reasoning = False
    for line in task.splitlines():
        if re.match(r"\s*Gold reasoning:", line):
            in_gold_reasoning = True
            continue
        if re.match(r"\s*Gold final answer:", line):
            in_gold_reasoning = False
            continue
        if in_gold_reasoning:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _build_private_training_reflection(specialty: str, training_batch: list[str]) -> str:
    if not training_batch:
        return f"{specialty} private-data reflection: no private sample was available; rely on explicit task constraints."
    return (
        f"{specialty} private-data reflection: extracted {len(training_batch)} private sample(s) into a reusable "
        "solving rule; identify the requested value, preserve the final-answer format, and verify calculations "
        "before submitting."
    )


def _summarize_texts(items: list[str], limit: int = 120) -> str:
    summaries = []
    for item in items[:3]:
        cleaned = " ".join(str(item).split())
        if len(cleaned) > limit:
            cleaned = cleaned[: limit - 3] + "..."
        summaries.append(cleaned)
    return " ; ".join(summaries) or "No private samples"


def _summarize_answer_for_evaluation(answer: str) -> str:
    features = extract_answer_features(answer)
    final_answer = _extract_final_answer(answer)
    final_answer_text = final_answer if final_answer else "missing"
    missing = features["missing_quality_signals"]
    missing_text = ", ".join(missing) if isinstance(missing, list) and missing else "none"
    return (
        f"final_answer={final_answer_text}; "
        f"numbered_steps={features['numbered_steps']}; "
        f"bullet_items={features['bullet_items']}; "
        f"calculation_signals={features['calculation_signals']}; "
        f"final_marker_present={features['has_final_marker']}; "
        f"missing_quality_signals={missing_text}"
    )
