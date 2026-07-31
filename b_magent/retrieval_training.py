from __future__ import annotations

import ast
import re


def build_retrieval_training_context(
    raw_reference_information: str,
    reference_response: str,
    *,
    max_blocks: int = 8,
    max_chars: int = 8_000,
) -> tuple[str, list[str]]:
    blocks = _parse_reference_blocks(raw_reference_information)
    if not blocks:
        return str(raw_reference_information)[:max_chars], []
    response_terms = _terms(reference_response)
    ranked = sorted(
        enumerate(blocks),
        key=lambda item: (_overlap(response_terms, _terms(_block_text(item[1]))), -item[0]),
        reverse=True,
    )
    relevant = [item for item in ranked if _overlap(response_terms, _terms(_block_text(item[1]))) > 0]
    selected = relevant[: max(1, max_blocks - 2)]
    selected_indexes = {index for index, _ in selected}
    remaining = [item for item in ranked if item[0] not in selected_indexes]
    selected.extend(remaining[: max_blocks - len(selected)])
    selected.sort(key=lambda item: item[0])
    context_parts: list[str] = []
    for index, block in selected:
        description = str(block.get("Description", f"source-{index + 1}")).strip()
        content = " ".join(str(block.get("Content", "")).split())
        context_parts.append(f"[source-{index + 1}] {description}\n{content}")
    context = "\n\n".join(context_parts)[:max_chars]
    relevant_sources = [
        f"source-{index + 1}: {str(block.get('Description', '')).strip()}"
        for index, block in relevant[:4]
    ]
    return context, relevant_sources


def strip_hidden_retrieval_labels(task: str) -> str:
    markers = ("Gold retrieval targets:", "Gold reference response:")
    lines = task.splitlines()
    for index, line in enumerate(lines):
        if any(line.strip().startswith(marker) for marker in markers):
            return "\n".join(lines[:index]).strip()
    return task.strip()


def extract_gold_reference_response(task: str) -> str | None:
    marker = "Gold reference response:"
    if marker not in task:
        return None
    return task.split(marker, 1)[1].strip() or None


def extract_gold_retrieval_targets(task: str) -> list[str]:
    marker = "Gold retrieval targets:"
    end_marker = "Gold reference response:"
    if marker not in task:
        return []
    value = task.split(marker, 1)[1]
    if end_marker in value:
        value = value.split(end_marker, 1)[0]
    return [item.strip() for item in value.split(";") if item.strip()]


def build_verified_retrieval_output(task: str) -> str | None:
    response = extract_gold_reference_response(task)
    if response is None:
        return None
    targets = extract_gold_retrieval_targets(task)
    return (
        f"Relevant sources: {'; '.join(targets) or '(reference context)'}\n"
        f"Evidence-based summary and answer:\n{response}"
    )


def is_retrieval_summary_grounded(task: str, answer: str) -> bool | None:
    gold = extract_gold_reference_response(task)
    if gold is None:
        return None
    visible_task = strip_hidden_retrieval_labels(task)
    gold_terms = _terms(gold)
    answer_terms = _terms(answer)
    evidence_terms = _terms(visible_task)
    if not gold_terms or not answer_terms:
        return False
    gold_coverage = _overlap(gold_terms, answer_terms)
    grounded_precision = _overlap(answer_terms, evidence_terms | gold_terms)
    required_numbers = set(re.findall(r"\d+(?:\.\d+)?", gold))
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    number_coverage = len(required_numbers & answer_numbers) / len(required_numbers) if required_numbers else 1.0
    return gold_coverage >= 0.35 and grounded_precision >= 0.75 and number_coverage >= 0.60


def _parse_reference_blocks(raw: str) -> list[dict[str, object]]:
    try:
        payload = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _block_text(block: dict[str, object]) -> str:
    return f"{block.get('Description', '')} {block.get('Content', '')}"


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", str(text).lower())
        if len(token) >= 2
    }


def _overlap(required: set[str], available: set[str]) -> float:
    return len(required & available) / len(required) if required else 0.0
