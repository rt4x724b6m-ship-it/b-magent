from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.qwen_gsm8k import (
    STANDARD_TEST_LIMIT,
    build_local_qwen_baseline_model,
    export_report,
    extract_numeric_answer,
    normalize_answer,
    run_qwen_gsm8k_baseline,
)
from b_magent.local_qwen import DEFAULT_QWEN_MODEL, LocalQwenAgentModel, NUMERIC_ANSWER_INSTRUCTION
from train.four_agent_private_train import AGENT_NAMES, build_four_local_qwen_agents


class FixedQwenModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.questions_seen: list[str] = []

    def generate(self, question: str) -> str:
        self.questions_seen.append(question)
        return self.answers[len(self.questions_seen) - 1]


class QwenGSM8KBaselineTestCase(unittest.TestCase):
    def test_extracts_numeric_answer_from_answer_prefixed_text(self) -> None:
        self.assertEqual(extract_numeric_answer("Answer: 65"), "65")
        self.assertEqual(extract_numeric_answer("There are 477 notebooks in all"), "477")
        self.assertEqual(
            extract_numeric_answer("#### (2 * 71) - 11 = 142 - 11 = 131\n\nThe answer is 131"),
            "131",
        )

    def test_normalizes_integer_decimal_answers(self) -> None:
        self.assertEqual(normalize_answer("16.00"), "16")
        self.assertEqual(normalize_answer("1,234.0"), "1234")
        self.assertEqual(normalize_answer("16.25"), "16.25")

    def test_local_qwen_agent_prompt_constrains_final_answer_format(self) -> None:
        class CapturingEngine:
            def __init__(self) -> None:
                self.prompt = ""

            def generate(self, prompt: str) -> str:
                self.prompt = prompt
                return "#### 42"

        engine = CapturingEngine()
        model = LocalQwenAgentModel(agent_name="qwen_test", engine=engine)

        self.assertEqual(model.generate("What is 20 + 22?"), "#### 42")
        self.assertIn(NUMERIC_ANSWER_INSTRUCTION, engine.prompt)

    def test_baseline_uses_raw_local_qwen_without_system_prompt(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_model_test_"))
        try:
            model = build_local_qwen_baseline_model(temp_dir)
            self.assertIsNone(model.system_prompt)
            self.assertTrue(model.local_files_only)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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

    def test_baseline_treats_integer_decimal_answers_as_correct(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_baseline_decimal_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            rows = [
                {"question": "What is 8 + 8?", "answer": "8 + 8 = 16. #### 16"},
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            model = FixedQwenModel(["Reasoning... #### 16.00"])
            report = run_qwen_gsm8k_baseline(dataset_dir, model=model, split="test", model_name="qwen-test")

            self.assertEqual(report.correct, 1)
            self.assertEqual(report.predictions[0].predicted_answer, "16")
            self.assertEqual(report.predictions[0].gold_answer, "16")
            self.assertTrue(report.predictions[0].correct)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_baseline_defaults_to_first_100_official_test_questions(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_baseline_limit_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            rows = [
                {"question": f"q{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_TEST_LIMIT + 1)
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            model = FixedQwenModel([f"#### {i}" for i in range(STANDARD_TEST_LIMIT)])
            report = run_qwen_gsm8k_baseline(dataset_dir, model=model, split="test")

            self.assertEqual(report.total, STANDARD_TEST_LIMIT)
            self.assertEqual(model.questions_seen, [f"q{i}" for i in range(STANDARD_TEST_LIMIT)])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_builds_four_qwen25_15b_local_agents(self) -> None:
        models = build_four_local_qwen_agents()

        self.assertEqual(list(models), list(AGENT_NAMES))
        self.assertTrue(all(isinstance(model, LocalQwenAgentModel) for model in models.values()))
        self.assertEqual(
            {model.engine.model_name_or_path for model in models.values()},
            {DEFAULT_QWEN_MODEL},
        )
        self.assertEqual(
            {id(model.engine) for model in models.values()},
            {id(next(iter(models.values())).engine)},
        )


if __name__ == "__main__":
    unittest.main()
