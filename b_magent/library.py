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
    ) -> list[LibraryRecord]:
        excluded = {"quarantined", *(exclude_tags or set())}
        terms = _semantic_terms(query)
        records = [record for record in self.all_records() if not excluded.intersection(record.tags)]
        if not records:
            return []

        def rank(record: LibraryRecord) -> tuple[float, str]:
            record_terms = _semantic_terms(
                f"{record.source_task} {record.summary} {' '.join(record.tags)} {record.detail}"
            )
            overlap = terms & record_terms
            union = terms | record_terms
            score = len(overlap) / len(union) if union else 0.0
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
}


def _semantic_terms(text: str) -> set[str]:
    visible = re.split(r"\n\s*Gold reasoning:", str(text), maxsplit=1, flags=re.IGNORECASE)[0]
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", visible.lower())
    return {token for token in tokens if len(token) >= 2 and token not in _STOP_TERMS}


def _record_fingerprint(record: LibraryRecord) -> str:
    payload = "|".join((record.agent_name, record.library_type, record.source_task, record.summary, record.detail))
    return sha256(payload.encode("utf-8")).hexdigest()
