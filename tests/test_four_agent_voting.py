from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT, run_qwen_gsm8k_baseline
from b_magent.library import EvolutionLibrary
from b_magent.local_qwen import DEFAULT_QWEN_MODEL, LocalQwenEngine, NUMERIC_ANSWER_INSTRUCTION
from train.four_agent_private_train import (
    AGENT_NAMES,
    format_voting_prediction_detail,
    print_voting_prediction_detail,
    reset_b_magent_training_state,
    run_four_agent_voting_on_test,
)


class FixedVoteModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.index = 0

    def train_batch(self, batch: object) -> None:
        return None

    def generate(self, question: str) -> str:
        answer = self.answers[self.index]
        self.index += 1
        return f"reasoning for {question} #### {answer}"


class RecordingModel:
    def __init__(self) -> None:
        self.questions_seen: list[str] = []

    def train_batch(self, batch: object) -> None:
        return None

    def generate(self, question: str) -> str:
        self.questions_seen.append(question)
        return "#### 0"


class KnowledgeLibraryVoteModel:
    def __init__(
        self,
        agent_name: str,
        engine: LocalQwenEngine,
        data_dir: Path,
        lora_output_dir: Path,
        memory_limit: int = 3,
    ) -> None:
        self.agent_name = agent_name
        self.engine = engine
        self.lora_output_dir = lora_output_dir
        self.professional_library = EvolutionLibrary(
            data_dir / agent_name / "professional_library.jsonl",
            "professional",
        )
        self.evaluation_library = EvolutionLibrary(
            data_dir / agent_name / "evaluation_library.jsonl",
            "evaluation",
        )
        self.memory_limit = memory_limit

    def train_batch(self, batch: object) -> None:
        return None

    def generate(self, question: str) -> str:
        professional_records = self.professional_library.search(question, limit=self.memory_limit)
        evaluation_records = self.evaluation_library.search(question, limit=self.memory_limit)
        prompt = (
            f"Agent: {self.agent_name}\n"
            "Use this agent's self-evolution knowledge libraries as extra context.\n\n"
            "Professional library memories:\n"
            f"{_format_library_records(professional_records)}\n\n"
            "Evaluation library checks:\n"
            f"{_format_library_records(evaluation_records)}\n\n"
            "Question:\n"
            f"{question}\n\n"
            f"Output constraint:\n{NUMERIC_ANSWER_INSTRUCTION}"
        )
        return self.engine.generate(prompt, adapter_path=self.adapter_path)

    @property
    def adapter_path(self) -> Path:
        return self.lora_output_dir / self.agent_name / "adapter"


def _format_library_records(records: list[object]) -> str:
    if not records:
        return "(none)"
    lines = []
    for index, record in enumerate(records, start=1):
        summary = " ".join(str(getattr(record, "summary", "")).split())
        detail = " ".join(str(getattr(record, "detail", "")).split())
        if len(detail) > 360:
            detail = detail[:357] + "..."
        tags = ", ".join(str(tag) for tag in getattr(record, "tags", []))
        lines.append(f"{index}. summary={summary}; detail={detail}; tags={tags}")
    return "\n".join(lines)


class FourAgentVotingTestCase(unittest.TestCase):
    def test_b_magent_reset_preserves_evaluation_libraries_by_default(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_reset_test_"))
        try:
            data_dir = temp_dir / "data"
            for agent_name in AGENT_NAMES:
                agent_dir = data_dir / agent_name
                agent_dir.mkdir(parents=True)
                (agent_dir / "professional_library.jsonl").write_text("professional\n", encoding="utf-8")
                (agent_dir / "evaluation_library.jsonl").write_text("evaluation\n", encoding="utf-8")
                (agent_dir / "private_data.jsonl").write_text("private\n", encoding="utf-8")

            reset_b_magent_training_state(data_dir, lora_output_dir=None)

            for agent_name in AGENT_NAMES:
                agent_dir = data_dir / agent_name
                self.assertFalse((agent_dir / "professional_library.jsonl").exists())
                self.assertFalse((agent_dir / "private_data.jsonl").exists())
                self.assertEqual(
                    (agent_dir / "evaluation_library.jsonl").read_text(encoding="utf-8"),
                    "evaluation\n",
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_reset_can_clear_evaluation_libraries_explicitly(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_reset_eval_test_"))
        try:
            data_dir = temp_dir / "data"
            for agent_name in AGENT_NAMES:
                agent_dir = data_dir / agent_name
                agent_dir.mkdir(parents=True)
                (agent_dir / "evaluation_library.jsonl").write_text("evaluation\n", encoding="utf-8")

            reset_b_magent_training_state(
                data_dir,
                lora_output_dir=None,
                reset_evaluation_libraries=True,
            )

            for agent_name in AGENT_NAMES:
                self.assertFalse((data_dir / agent_name / "evaluation_library.jsonl").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_four_agents_vote_final_answer_on_test_dataset(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_vote_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            test_rows = [
                {"question": "What is 20 + 22?", "answer": "20 + 22 = 42. #### 42"},
                {"question": "What is 10 - 3?", "answer": "10 - 3 = 7. #### 7"},
                {"question": "What is 5 + 5?", "answer": "5 + 5 = 10. #### 10"},
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            models = {
                "qwen_agent_1": FixedVoteModel(["42", "7", "9"]),
                "qwen_agent_2": FixedVoteModel(["42", "8", "11"]),
                "qwen_agent_3": FixedVoteModel(["41", "7", "12"]),
                "qwen_agent_4": FixedVoteModel(["0", "8", "13"]),
            }
            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.total, 3)
            self.assertEqual(report.correct, 2)
            self.assertEqual(report.accuracy, 2 / 3)
            self.assertEqual([vote.agent_name for vote in report.predictions[0].votes], list(AGENT_NAMES))
            self.assertEqual(report.predictions[0].final_answer, "42")
            self.assertTrue(report.predictions[0].correct)

            # The second row is a 2-2 tie: agent_1/agent_3 vote 7, agent_2/agent_4 vote 8.
            # Ties are resolved by the first answer in AGENT_NAMES order.
            self.assertEqual(report.predictions[1].final_answer, "7")
            self.assertTrue(report.predictions[1].correct)

            self.assertEqual(report.predictions[2].final_answer, "9")
            self.assertFalse(report.predictions[2].correct)
            self.assertIn("result=正确", format_voting_prediction_detail(report.predictions[0], report.total))
            self.assertIn("result=错误", format_voting_prediction_detail(report.predictions[2], report.total))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_voting_treats_integer_decimal_answers_as_correct(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_vote_decimal_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            test_rows = [
                {"question": "What is 8 + 8?", "answer": "8 + 8 = 16. #### 16"},
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            models = {
                "qwen_agent_1": FixedVoteModel(["16.00"]),
                "qwen_agent_2": FixedVoteModel(["16.00"]),
                "qwen_agent_3": FixedVoteModel(["16"]),
                "qwen_agent_4": FixedVoteModel(["16"]),
            }
            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.correct, 1)
            self.assertEqual(report.predictions[0].final_answer, "16")
            self.assertTrue(report.predictions[0].correct)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_voting_evaluates_first_100_official_test_questions_by_default(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_vote_limit_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            test_rows = [
                {"question": f"q{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_TEST_LIMIT + 1)
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            answers = [str(i) for i in range(STANDARD_TEST_LIMIT)]
            models = {agent_name: FixedVoteModel(answers.copy()) for agent_name in AGENT_NAMES}
            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.total, STANDARD_TEST_LIMIT)
            self.assertEqual(report.correct, STANDARD_TEST_LIMIT)
            self.assertEqual(report.predictions[-1].question, f"q{STANDARD_TEST_LIMIT - 1}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_voting_uses_same_first_100_questions_as_qwen_baseline(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_vote_baseline_same_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            test_rows = [
                {"question": f"same-q{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_TEST_LIMIT + 3)
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            baseline_model = RecordingModel()
            baseline_report = run_qwen_gsm8k_baseline(dataset_dir, model=baseline_model, split="test")
            voting_models = {agent_name: RecordingModel() for agent_name in AGENT_NAMES}
            voting_report = run_four_agent_voting_on_test(dataset_dir, models=voting_models)

            baseline_questions = [prediction.question for prediction in baseline_report.predictions]
            voting_questions = [prediction.question for prediction in voting_report.predictions]
            self.assertEqual(voting_questions, baseline_questions)
            self.assertEqual(voting_questions, [f"same-q{i}" for i in range(STANDARD_TEST_LIMIT)])
            for model in voting_models.values():
                self.assertEqual(model.questions_seen, baseline_questions)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class KnowledgeLibraryFourAgentVotingIntegrationTestCase(unittest.TestCase):
    def test_four_agents_vote_with_lora_tuned_models_on_first_100_test_questions(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        dataset_dir = project_root / "data" / "gsm8k"
        model_path = project_root / DEFAULT_QWEN_MODEL
        data_dir = project_root / "data"
        lora_output_dir = data_dir / "lora_adapters"

        if not (dataset_dir / "test.jsonl").exists():
            self.skipTest(f"missing GSM8K test split: {dataset_dir / 'test.jsonl'}")
        if not model_path.exists():
            self.skipTest(f"missing local Qwen model: {model_path}")

        missing_libraries = [
            agent_name
            for agent_name in AGENT_NAMES
            if not (
                (data_dir / agent_name / "professional_library.jsonl").exists()
                and (data_dir / agent_name / "evaluation_library.jsonl").exists()
            )
        ]
        if missing_libraries:
            self.skipTest(f"missing knowledge libraries for: {', '.join(missing_libraries)}")
        missing_adapters = [
            agent_name
            for agent_name in AGENT_NAMES
            if not (lora_output_dir / agent_name / "adapter" / "adapter_config.json").exists()
        ]
        if missing_adapters:
            self.skipTest(f"missing LoRA adapters for: {', '.join(missing_adapters)}")

        engine = LocalQwenEngine(model_name_or_path=model_path)
        models = {
            agent_name: KnowledgeLibraryVoteModel(
                agent_name=agent_name,
                engine=engine,
                data_dir=data_dir,
                lora_output_dir=lora_output_dir,
            )
            for agent_name in AGENT_NAMES
        }
        report = run_four_agent_voting_on_test(
            dataset_dir=dataset_dir,
            models=models,
            limit=STANDARD_TEST_LIMIT,
            on_prediction=print_voting_prediction_detail,
        )
        output_file = project_root / "train" / "four_agent_lora_voting_100_report.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.assertEqual(report.total, STANDARD_TEST_LIMIT)
        self.assertEqual(len(report.predictions), STANDARD_TEST_LIMIT)
        self.assertEqual([vote.agent_name for vote in report.predictions[0].votes], list(AGENT_NAMES))
        self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
