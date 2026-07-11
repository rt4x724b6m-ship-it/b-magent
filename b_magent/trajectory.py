from __future__ import annotations

import re

from .models import Draft


def build_evaluation_trajectory(draft: Draft) -> list[str]:
    """Build a FoT-style reusable trace without exposing raw private samples."""
    raw_steps = [item for item in draft.thought_trace if str(item).strip()]
    answer_features = extract_answer_features(draft.answer)
    trace: list[str] = [
        f"insight_problem_framing: Identify the task goal, constraints, and required final output before reviewing {draft.agent_name}'s answer.",
        "insight_answer_audit: Check whether each answer step is explicit, verifiable, and connected to the task.",
        (
            "insight_solution_structure: Evaluate the submitted answer from observable structure only. "
            f"Numbered steps: {answer_features['numbered_steps']}; "
            f"bullet items: {answer_features['bullet_items']}; "
            f"calculation signals: {answer_features['calculation_signals']}."
        ),
        (
            "insight_final_answer_check: Verify that the answer ends with a clear final result marker "
            f"and that the final result is consistent with the preceding public answer. "
            f"Final marker present: {answer_features['has_final_marker']}."
        ),
    ]
    if raw_steps:
        trace.append(
            "insight_reasoning_trace: Review the disclosed reasoning procedure as reusable procedural knowledge, "
            f"not as private training evidence. Available public step count: {len(raw_steps)}."
        )
    if draft.professional_memory_used:
        trace.append(
            "insight_memory_abstraction: Treat retrieved professional memories as summarized strategy hints; "
            f"memory summary count: {len(draft.professional_memory_used)}."
        )
    if draft.evaluation_alerts_used:
        trace.append(
            "insight_evaluation_checks: Apply prior evaluation alerts as generic quality checks; "
            f"check count: {len(draft.evaluation_alerts_used)}."
        )
    if answer_features["missing_quality_signals"]:
        trace.append(
            "insight_missing_quality_signals: Prioritize feedback on absent public quality signals: "
            + ", ".join(answer_features["missing_quality_signals"])
            + "."
        )
    trace.append(
        "insight_feedback_format: Return concrete corrections, missing boundary cases, score rationale, "
        "and final-answer consistency checks without requesting raw private data."
    )
    return trace


def mask_draft_for_evaluation(draft: Draft) -> Draft:
    """Return a peer-review view that hides raw private data behind trajectory traces."""
    return Draft(
        agent_name=draft.agent_name,
        specialty=draft.specialty,
        answer=build_transmission_summary(draft),
        thought_trace=build_evaluation_trajectory(draft),
        private_training_used=[],
        professional_memory_used=[
            f"professional_memory_summary_count={len(draft.professional_memory_used)}"
        ]
        if draft.professional_memory_used
        else [],
        evaluation_alerts_used=[
            f"evaluation_alert_summary_count={len(draft.evaluation_alerts_used)}"
        ]
        if draft.evaluation_alerts_used
        else [],
        tool_calls=[
            "mask_private_training_for_peer_evaluation()",
            f"build_fot_style_trajectory(count={len(draft.thought_trace)})",
        ],
    )


def build_transmission_summary(draft: Draft) -> str:
    """Federated-style payload: share abstract quality signals, not raw answer text."""
    features = extract_answer_features(draft.answer)
    final_answer = extract_final_answer(draft.answer)
    missing = features["missing_quality_signals"]
    missing_text = ", ".join(missing) if isinstance(missing, list) and missing else "none"
    return (
        "federated_answer_summary:\n"
        f"- source_agent: {draft.agent_name}\n"
        f"- specialty: {draft.specialty}\n"
        f"- final_answer: {final_answer or 'missing'}\n"
        f"- final_marker_present: {features['has_final_marker']}\n"
        f"- numbered_steps: {features['numbered_steps']}\n"
        f"- bullet_items: {features['bullet_items']}\n"
        f"- calculation_signals: {features['calculation_signals']}\n"
        f"- missing_quality_signals: {missing_text}\n"
        f"- private_training_count: {len(draft.private_training_used)}\n"
        f"- local_trace_step_count: {len([item for item in draft.thought_trace if str(item).strip()])}"
    )


def extract_answer_features(answer: str) -> dict[str, object]:
    numbered_steps = len(re.findall(r"(?m)^\s*\d+[.)、]", answer))
    bullet_items = len(re.findall(r"(?m)^\s*[-*]\s+", answer))
    calculation_signals = len(re.findall(r"[=+\-*/]|####|-?\d+(?:\.\d+)?", answer))
    has_final_marker = "####" in answer
    lower_answer = answer.lower()
    missing_quality_signals: list[str] = []
    if not has_final_marker:
        missing_quality_signals.append("explicit final answer marker")
    if not any(word in lower_answer for word in ("check", "verify", "验证", "检查", "自检")):
        missing_quality_signals.append("verification step")
    if not any(word in lower_answer for word in ("edge", "boundary", "case", "边界", "条件")):
        missing_quality_signals.append("boundary-condition discussion")
    if numbered_steps + bullet_items == 0:
        missing_quality_signals.append("step-by-step structure")
    return {
        "numbered_steps": numbered_steps,
        "bullet_items": bullet_items,
        "calculation_signals": calculation_signals,
        "has_final_marker": has_final_marker,
        "missing_quality_signals": missing_quality_signals,
    }


def extract_final_answer(answer: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", answer)
    if matches:
        return _normalize_answer(matches[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return _normalize_answer(numbers[-1]) if numbers else ""


def _normalize_answer(text: str) -> str:
    cleaned = str(text).strip().replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned
