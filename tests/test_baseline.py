from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from baseline.qwen_gsm8k import export_report, run_qwen_gsm8k_baseline


class FixedQwenModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.questions_seen: list[str] = []

    def generate(self, question: str) -> str:
        self.questions_seen.append(question)
        return self.answers[len(self.questions_seen) - 1]


class QwenGSM8KBaselineTestCase(unittest.TestCase):
    def test_single_qwen_model_runs_on_test_split(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_baseline_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            rows = [
                {"question": "What is 20 + 22?", "answer": "20 + 22 = 42. #### 42"},
                {"question": "What is 10 - 3?", "answer": "10 - 3 = 7. #### 7"},
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            model = FixedQwenModel(["Reasoning... #### 42", "Reasoning... #### 8"])
            report = run_qwen_gsm8k_baseline(dataset_dir, model=model, split="test", model_name="qwen-test")

            self.assertEqual(model.questions_seen, ["What is 20 + 22?", "What is 10 - 3?"])
            self.assertEqual(report.total, 2)
            self.assertEqual(report.correct, 1)
            self.assertEqual(report.accuracy, 0.5)
            self.assertEqual(report.predictions[0].predicted_answer, "42")
            self.assertTrue(report.predictions[0].correct)
            self.assertFalse(report.predictions[1].correct)

            output_file = temp_dir / "baseline" / "report.json"
            export_report(report, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_name"], "qwen-test")
            self.assertEqual(payload["split"], "test")
            self.assertEqual(payload["accuracy"], 0.5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
