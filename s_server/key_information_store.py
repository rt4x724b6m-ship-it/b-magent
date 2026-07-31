from __future__ import annotations

import json
import re
from pathlib import Path

from b_magent.library import EvolutionLibrary
from b_magent.models import LibraryRecord
from b_magent.tagging import extract_math_task_tags


class ServerKeyInformationStore:
    """Persistent server cache for verified question analysis and synthesis."""

    def __init__(self, path: Path) -> None:
        self.library = EvolutionLibrary(path, "server_key_information")

    def lookup(
        self,
        question: str,
        *,
        key_facts: list[str] | None = None,
        query_tags: set[str] | None = None,
        min_similarity: float = 0.45,
    ) -> LibraryRecord | None:
        normalized_question = _normalize_text(question)
        records = self.library.all_records()
        for record in reversed(records):
            if _normalize_text(record.source_task) == normalized_question:
                return record

        candidates = self.library.search(
            question,
            limit=5,
            query_tags=query_tags,
            key_facts=key_facts,
        )
        query_terms = _key_terms(" ".join([question, *(key_facts or [])]))
        for record in candidates:
            record_terms = _key_terms(f"{record.source_task} {record.detail}")
            similarity = len(query_terms & record_terms) / len(query_terms) if query_terms else 0.0
            if similarity >= min_similarity:
                return record
        return None

    def store_verified(
        self,
        question: str,
        synthesis: str,
        final_answer: str,
        key_information: dict[str, object],
    ) -> LibraryRecord:
        tags = sorted(
            {
                "server-key-information",
                "verified-answer",
                *extract_math_task_tags(question),
                *(str(tag) for tag in key_information.get("routing_tags", []) if str(tag).strip()),
            }
        )
        return self.library.add_record(
            LibraryRecord(
                agent_name="qwen_server_agent",
                library_type="server_key_information",
                source_task=question,
                summary=synthesis,
                detail=json.dumps(
                    {
                        "final_answer": final_answer,
                        "key_information": key_information,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                tags=tags,
            )
        )


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).lower()))


def _key_terms(text: str) -> set[str]:
    lower = str(text).lower()
    terms = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?%?", lower))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lower):
        terms.update(sequence[index : index + 2] for index in range(max(len(sequence) - 1, 0)))
    return {term for term in terms if len(term) >= 2 or term[0].isdigit()}
