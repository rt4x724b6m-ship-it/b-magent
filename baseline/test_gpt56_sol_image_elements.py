from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baseline.gpt56_sol_image_elements import enrich_jsonl, parse_elements, response_text


class GPT56SolImageElementsTest(unittest.TestCase):
    def test_enriches_only_with_image_elements_and_reuses_image_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "images" / "same.png").write_bytes(b"image")
            rows = [
                {"image": "images/same.png", "question": "q1", "answers": ["a1"]},
                {"image": "images/same.png", "question": "q2", "answers": ["a2"]},
            ]
            source = root / "train.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            calls: list[Path] = []

            def describe(path: Path) -> dict[str, object]:
                calls.append(path)
                return {"summary": "a diagram", "objects": ["chart"]}

            output = root / "out" / "train.jsonl"
            self.assertEqual(enrich_jsonl(source, output, describe), 2)
            enriched = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(calls), 1)
            for before, after in zip(rows, enriched):
                self.assertEqual({key: after[key] for key in before}, before)
                self.assertEqual(after["image_elements"]["summary"], "a diagram")

    def test_reads_responses_api_output_and_json_fence(self) -> None:
        payload = {"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "```json\n{\"summary\": \"x\"}\n```"}
        ]}]}
        self.assertEqual(parse_elements(response_text(payload)), {"summary": "x"})


if __name__ == "__main__":
    unittest.main()
