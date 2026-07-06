from __future__ import annotations

import json
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

    def search(self, query: str, limit: int = 3) -> list[LibraryRecord]:
        terms = [term for term in query.replace("，", " ").replace("。", " ").split() if term]
        records = self.all_records()
        if not records:
            return []

        def rank(record: LibraryRecord) -> tuple[int, str]:
            haystack = " ".join([record.source_task, record.summary, record.detail, *record.tags])
            score = sum(1 for term in terms if term in haystack)
            if query and query in haystack:
                score += 2
            return score, record.created_at

        ranked = sorted(records, key=rank, reverse=True)
        matched = [record for record in ranked if rank(record)[0] > 0]
        return (matched or ranked)[:limit]

