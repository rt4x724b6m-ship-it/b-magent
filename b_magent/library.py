from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from .models import LibraryRecord


class EvolutionLibrary:
    def __init__(self, path: Path, library_type: str) -> None:
        self.path = path
        self.library_type = library_type
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add_record(self, record: LibraryRecord) -> LibraryRecord:
        if record.library_type != self.library_type:
            raise ValueError(f"expected {self.library_type} record, got {record.library_type}")
        # Idempotency matters when a round is retried after an interrupted run.
        record_id = _record_fingerprint(record)
        for existing in self.all_records():
            if _record_fingerprint(existing) == record_id:
                return existing
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def all_records(self) -> list[LibraryRecord]:
        records: list[LibraryRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(LibraryRecord.from_dict(json.loads(line)))
        return records

    def search(
        self,
        query: str,
        limit: int = 3,
        exclude_tags: set[str] | None = None,
        query_tags: set[str] | None = None,
        key_facts: list[str] | None = None,
    ) -> list[LibraryRecord]:
        excluded = {"quarantined", *(exclude_tags or set())}
        normalized_query_tags = {_normalize_tag(tag) for tag in (query_tags or set())}
        facts = [fact for fact in (key_facts or []) if str(fact).strip()]
        terms = _semantic_terms(" ".join([query, *facts]))
        query_numbers_and_units = _numbers_and_units(" ".join([query, *facts]))
        records = [record for record in self.all_records() if not excluded.intersection(record.tags)]
        if not records:
            return []

        def rank(record: LibraryRecord) -> tuple[float, str]:
            record_text = f"{record.source_task} {record.summary} {' '.join(record.tags)} {record.detail}"
            record_terms = _semantic_terms(record_text)
            overlap = terms & record_terms
            # Query coverage is more stable than Jaccard here: experience details
            # can be long, and unrelated detail terms should not dilute a precise
            # match on the question's entities, operations, and units.
            lexical_score = len(overlap) / len(terms) if terms else 0.0
            record_tags = {_normalize_tag(tag) for tag in record.tags}
            tag_score = (
                len(normalized_query_tags & record_tags) / len(normalized_query_tags)
                if normalized_query_tags
                else 0.0
            )
            fact_score = max(
                (_term_overlap(_semantic_terms(fact), record_terms) for fact in facts),
                default=0.0,
            )
            numeric_unit_score = _term_overlap(
                query_numbers_and_units,
                _numbers_and_units(record_text),
            )
            quality_score = _experience_quality(record_tags)
            relevance_score = (
                lexical_score * 0.30
                + tag_score * 0.25
                + fact_score * 0.25
                + numeric_unit_score * 0.10
            )
            score = relevance_score + quality_score * 0.10 if relevance_score > 0 else 0.0
            return score, record.created_at

        ranked = sorted(records, key=rank, reverse=True)
        matched = [record for record in ranked if rank(record)[0] > 0]
        if matched:
            return matched[:limit]
        return [record for record in ranked if "seed" in record.tags][:limit]


_STOP_TERMS = {
    "solve", "this", "gsm8k", "training", "problem", "preserve", "reusable", "solving",
    "lessons", "question", "gold", "reasoning", "final", "answer", "the", "and", "for",
    "with", "from", "that", "how", "many", "much", "does", "did", "was", "were", "has",
    "have", "his", "her", "their", "into", "after", "before", "each", "what", "when",
    "image", "visual", "inspect", "visible", "evidence", "complete", "direct", "return",
    "short", "elements", "relevant", "current", "supplied",
}


def _semantic_terms(text: str) -> set[str]:
    visible = re.split(r"\n\s*Gold reasoning:", str(text), maxsplit=1, flags=re.IGNORECASE)[0]
    lower = visible.lower()
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?", lower)
    terms = {token for token in tokens if len(token) >= 2 and token not in _STOP_TERMS}
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lower):
        if len(sequence) == 1:
            continue
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def _numbers_and_units(text: str) -> set[str]:
    return set(
        re.findall(
            r"\d+(?:\.\d+)?%?|[$¥€£]|percent|percentage|dollars?|hours?|minutes?|days?|"
            r"百分之|元|美元|小时|分钟|天|米|千米|公里|千克|公斤",
            str(text).lower(),
        )
    )


def _term_overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left) if left else 0.0


def _normalize_tag(tag: str) -> str:
    return str(tag).strip().lower().replace("_", "-").replace(" ", "-")


def _experience_quality(tags: set[str]) -> float:
    if "curated-success-experience" in tags:
        return 1.0
    if "error-reflection-experience" in tags or "error-evaluation-experience" in tags:
        return 0.0
    if "private-training" in tags:
        return 0.7
    if "evaluated-experience" in tags:
        return 0.6
    return 0.5


def _record_fingerprint(record: LibraryRecord) -> str:
    payload = "|".join((record.agent_name, record.library_type, record.source_task, record.summary, record.detail))
    return sha256(payload.encode("utf-8")).hexdigest()
