from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT, run_qwen_gsm8k_baseline
from b_magent.library import EvolutionLibrary
from b_magent.local_qwen import (
    DEFAULT_QWEN_MODEL,
    LocalQwenAgentModel,
    LocalQwenEngine,
    NUMERIC_ANSWER_INSTRUCTION,
)
from b_magent.models import LibraryRecord
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


class FixedRecordingVoteModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions_seen: list[str] = []
        self.guidance_seen: list[str] = []

    def train_batch(self, batch: object) -> None:
        return None

    def generate(self, question: str) -> str:
        self.questions_seen.append(question)
        return f"reasoning for {question} #### {self.answer}"

    def generate_with_server_guidance(self, question: str, server_guidance: str) -> str:
        self.guidance_seen.append(server_guidance)
        return self.generate(question)


class RecordingServerRoutingModel:
    def __init__(self, diagnostic: str) -> None:
        self.diagnostic = diagnostic
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.diagnostic


class ConcurrentVoteModel:
    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier

    def train_batch(self, batch: object) -> None:
        return None

    def generate(self, question: str) -> str:
        self.barrier.wait(timeout=2)
        return "#### 42"


class CapturingKnowledgeEngine:
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}
        self.lock = threading.Lock()

    def generate(self, prompt: str, adapter_path: Path) -> str:
        agent_name = adapter_path.parent.name
        with self.lock:
            self.prompts[agent_name] = prompt
        return "#### 42"


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
        return self.generate_with_server_guidance(question, "")

    def generate_with_server_guidance(self, question: str, server_guidance: str) -> str:
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
            "Server evaluation guidance:\n"
            f"{server_guidance or '(none)'}\n\n"
            f"Output constraint:\n{NUMERIC_ANSWER_INSTRUCTION}"
        )
        return self.engine.generate(prompt, adapter_path=self.adapter_path)

    @property
    def adapter_path(self) -> Path:
        return self.lora_output_dir / self.agent_name / "adapter"


class KnowledgeServerRoutingModel:
    def __init__(
        self,
        engine: LocalQwenEngine,
        memory_limit: int = 3,
    ) -> None:
        self.engine = engine
        self.memory_limit = memory_limit

    def generate(self, prompt: str) -> str:
        return self.engine.generate(prompt)


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


def _is_lora_adapter_ready(adapter_path: Path) -> bool:
    return (adapter_path / "adapter_config.json").exists()


def _load_library_records(path: Path) -> list[LibraryRecord]:
    records: list[LibraryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(LibraryRecord.from_dict(json.loads(line)))
    return records


class FourAgentVotingTestCase(unittest.TestCase):
    def test_server_routing_does_not_append_guidance_to_agent_question(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_no_guidance_route_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            question = "What is 20 + 22?"
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": question, "answer": "#### 42"}) + "\n",
                encoding="utf-8",
            )
            models = {agent_name: FixedRecordingVoteModel("42") for agent_name in AGENT_NAMES}
            server_model = RecordingServerRoutingModel(
                json.dumps(
                    {
                        "difficulty": "easy",
                        "key_steps": ["add the two values"],
                        "risk_steps": ["check arithmetic"],
                        "capability_tags": ["addition", "arithmetic"],
                        "risk_tags": ["verification"],
                    }
                )
            )
            server_tag_records = [
                LibraryRecord(
                    agent_name=agent_name,
                    library_type="agent_training_tags",
                    source_task="addition training",
                    summary="training tags",
                    detail="source_library_type=professional",
                    tags=[agent_name, "agent-training-tags", "professional", "addition", "arithmetic"],
                )
                for agent_name in AGENT_NAMES
            ]

            report = run_four_agent_voting_on_test(
                dataset_dir,
                models=models,
                server_model=server_model,
                server_training_tag_records=server_tag_records,
            )

            prediction = report.predictions[0]
            self.assertEqual(prediction.key_steps, ["add the two values"])
            for agent_name in prediction.selected_agents:
                self.assertEqual(models[agent_name].questions_seen, [question])
                self.assertEqual(models[agent_name].guidance_seen, [])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_uses_comprehensive_assessment_and_relevant_global_evaluation(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_comprehensive_route_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            question = "A $20 item is discounted by 50%. What is the final price?"
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": question, "answer": "#### 10"}) + "\n",
                encoding="utf-8",
            )
            server_model = RecordingServerRoutingModel(
                json.dumps(
                    {
                        "difficulty": "hard",
                        "key_steps": ["compute the percentage discount", "subtract it from the price"],
                        "risk_steps": ["do not return the discount amount as the final price"],
                        "capability_tags": ["money", "percentage", "subtraction"],
                        "risk_tags": ["multi-step", "verification"],
                    }
                )
            )
            models = {agent_name: FixedRecordingVoteModel("10") for agent_name in AGENT_NAMES}
            server_tag_records = [
                LibraryRecord(
                    agent_name=agent_name,
                    library_type="agent_training_tags",
                    source_task="A $30 product has a 20% discount. Gold final answer: 24",
                    summary="successful discount problem",
                    detail="source_library_type=professional",
                    tags=[
                        agent_name,
                        "agent-training-tags",
                        "professional",
                        "curated-success-experience",
                        "money",
                        "percentage",
                    ],
                )
                for agent_name in AGENT_NAMES
            ]
            unrelated_global_records = [
                LibraryRecord(
                    agent_name="qwen_server_agent",
                    library_type="global_evaluation",
                    source_task=f"geometry task {index}",
                    summary=f"unrelated geometry lesson {index}",
                    detail="check area",
                    tags=["geometry"],
                )
                for index in range(5)
            ]
            relevant_record = LibraryRecord(
                agent_name="qwen_server_agent",
                library_type="global_evaluation",
                source_task="money percentage discount task",
                summary="relevant money discount lesson",
                detail="Distinguish the discount amount from the final price.",
                tags=["money", "percentage", "verification"],
            )

            report = run_four_agent_voting_on_test(
                dataset_dir,
                models=models,
                server_model=server_model,
                server_training_tag_records=server_tag_records,
                prior_global_evaluation_records=[*unrelated_global_records, relevant_record],
            )

            prediction = report.predictions[0]
            self.assertEqual(prediction.difficulty, "hard")
            self.assertIn("compute the percentage discount", prediction.key_steps)
            self.assertIn("do not return the discount amount as the final price", prediction.risk_steps)
            self.assertTrue({"money", "percentage", "subtraction", "multi-step", "verification"} <= set(prediction.routing_tags))
            self.assertIn("relevant money discount lesson", server_model.prompts[0])
            for agent_name in prediction.selected_agents:
                self.assertEqual(models[agent_name].guidance_seen, [])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_routing_prefers_success_evidence_over_error_evidence(self) -> None:
        from train.four_agent_private_train import _build_agent_tag_index, select_agents_by_server_tags

        def record(agent_name: str, outcome: str) -> LibraryRecord:
            return LibraryRecord(
                agent_name=agent_name,
                library_type="agent_training_tags",
                source_task="Question: A jacket costs $20. What is its price?",
                summary="training tags",
                detail="source_library_type=professional",
                tags=[agent_name, "agent-training-tags", "professional", outcome, "money"],
            )

        profiles = _build_agent_tag_index(
            [
                record("qwen_agent_1", "error-reflection-experience"),
                record("qwen_agent_2", "evaluated-experience"),
                record("qwen_agent_3", "private-training"),
                record("qwen_agent_4", "curated-success-experience"),
            ]
        )

        selected, _ = select_agents_by_server_tags(
            "money arithmetic",
            profiles,
            question="The price is $20.",
        )

        self.assertEqual(selected, ["qwen_agent_2", "qwen_agent_3", "qwen_agent_4"])

    def test_local_qwen_adapter_cache_passes_path_string_to_peft(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_adapter_cache_test_"))
        try:
            adapter_path = temp_dir / "adapter"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_path / "adapter_model.safetensors").write_text("weights-v1", encoding="utf-8")
            engine = LocalQwenEngine()
            engine._model = object()
            loaded_paths: list[object] = []

            class FakePeftModel:
                @staticmethod
                def from_pretrained(model: object, path: object) -> object:
                    loaded_paths.append(path)
                    return {"model": model, "path": path}

            with patch.dict("sys.modules", {"peft": type("FakePeftModule", (), {"PeftModel": FakePeftModel})}):
                first = engine._load_adapter_model(adapter_path)
                second = engine._load_adapter_model(adapter_path)

            self.assertIs(first, second)
            self.assertEqual(loaded_paths, [str(adapter_path.resolve())])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_local_qwen_auto_device_map_avoids_accelerate_dispatch(self) -> None:
        engine = LocalQwenEngine(device_map="auto")

        self.assertIsNone(engine._resolve_device_map())

    def test_local_qwen_explicit_device_map_is_preserved(self) -> None:
        engine = LocalQwenEngine(device_map="balanced")

        self.assertEqual(engine._resolve_device_map(), "balanced")

    def test_b_magent_reset_clears_all_training_experience_by_default(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_reset_test_"))
        try:
            data_dir = temp_dir / "data"
            lora_output_dir = data_dir / "lora_adapters"
            report_file = temp_dir / "training_report.json"
            for agent_name in AGENT_NAMES:
                agent_dir = data_dir / agent_name
                agent_dir.mkdir(parents=True)
                (agent_dir / "professional_library.jsonl").write_text("professional\n", encoding="utf-8")
                (agent_dir / "evaluation_library.jsonl").write_text("evaluation\n", encoding="utf-8")
                (agent_dir / "private_data.jsonl").write_text("private\n", encoding="utf-8")
            server_dir = data_dir / "qwen_server_agent"
            server_dir.mkdir(parents=True)
            (server_dir / "global_evaluation_library.jsonl").write_text("global\n", encoding="utf-8")
            (server_dir / "agent_training_tags.jsonl").write_text("tags\n", encoding="utf-8")
            (lora_output_dir / "qwen_agent_1").mkdir(parents=True)
            (lora_output_dir / "qwen_agent_1" / "state.json").write_text("{}", encoding="utf-8")
            report_file.write_text("{}", encoding="utf-8")

            reset_b_magent_training_state(
                data_dir,
                lora_output_dir=lora_output_dir,
                report_files=(report_file,),
            )

            for agent_name in AGENT_NAMES:
                agent_dir = data_dir / agent_name
                self.assertFalse((agent_dir / "professional_library.jsonl").exists())
                self.assertFalse((agent_dir / "private_data.jsonl").exists())
                self.assertFalse((agent_dir / "evaluation_library.jsonl").exists())
            self.assertFalse((server_dir / "global_evaluation_library.jsonl").exists())
            self.assertFalse((server_dir / "agent_training_tags.jsonl").exists())
            self.assertFalse(lora_output_dir.exists())
            self.assertFalse(report_file.exists())
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

    def test_server_routes_test_question_to_three_tag_matched_agents(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_server_routed_vote_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            question = "What is 20 + 22?"
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": question, "answer": "20 + 22 = 42. #### 42"}) + "\n",
                encoding="utf-8",
            )

            models = {
                "qwen_agent_1": FixedRecordingVoteModel("41"),
                "qwen_agent_2": FixedRecordingVoteModel("42"),
                "qwen_agent_3": FixedRecordingVoteModel("42"),
                "qwen_agent_4": FixedRecordingVoteModel("0"),
            }
            server_model = RecordingServerRoutingModel(
                "server COT hidden; observed errors: arithmetic calculation, final-answer check, verification"
            )
            server_tag_records = [
                LibraryRecord(
                    agent_name="qwen_agent_1",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_1 tags",
                    detail="",
                    tags=["arithmetic", "final-answer", "verification"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_2",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_2 tags",
                    detail="",
                    tags=["arithmetic", "final-answer"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_3",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_3 tags",
                    detail="",
                    tags=["verification"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_4",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_4 tags",
                    detail="",
                    tags=["structure"],
                ),
            ]
            prior_global_records = [
                LibraryRecord(
                    agent_name="qwen_server_agent",
                    library_type="global_evaluation",
                    source_task="training",
                    summary="verify arithmetic and final answer consistency",
                    detail="",
                    tags=["global-evaluation", "verification", "final-answer"],
                )
            ]

            report = run_four_agent_voting_on_test(
                dataset_dir,
                models=models,
                server_model=server_model,
                server_training_tag_records=server_tag_records,
                prior_global_evaluation_records=prior_global_records,
            )

            prediction = report.predictions[0]
            self.assertEqual(prediction.selected_agents, ["qwen_agent_1", "qwen_agent_2", "qwen_agent_3"])
            self.assertEqual([vote.agent_name for vote in prediction.votes], prediction.selected_agents)
            self.assertEqual(prediction.votes[0].predicted_answer, "41")
            self.assertEqual(prediction.votes[1].predicted_answer, "42")
            self.assertEqual(prediction.votes[2].predicted_answer, "42")
            self.assertEqual(prediction.final_answer, "42")
            self.assertTrue(prediction.correct)
            self.assertEqual(models["qwen_agent_4"].questions_seen, [])
            self.assertEqual(models["qwen_agent_1"].questions_seen, [question])
            self.assertIn(question, server_model.prompts[0])
            self.assertIn("Prior aggregated evaluation experience", server_model.prompts[0])
            self.assertIn("verification", prediction.matched_tags["qwen_agent_1"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_routed_vote_preserves_original_agent_order_for_selected_agents(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_server_routed_order_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": "What is 20 + 22?", "answer": "#### 99"}) + "\n",
                encoding="utf-8",
            )

            models = {
                "qwen_agent_1": FixedRecordingVoteModel("1"),
                "qwen_agent_2": FixedRecordingVoteModel("2"),
                "qwen_agent_3": FixedRecordingVoteModel("3"),
                "qwen_agent_4": FixedRecordingVoteModel("99"),
            }
            server_model = RecordingServerRoutingModel("arithmetic final-answer verification")
            server_tag_records = [
                LibraryRecord(
                    agent_name="qwen_agent_1",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_1 tags",
                    detail="",
                    tags=["structure"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_2",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_2 tags",
                    detail="",
                    tags=["arithmetic"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_3",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_3 tags",
                    detail="",
                    tags=["final-answer"],
                ),
                LibraryRecord(
                    agent_name="qwen_agent_4",
                    library_type="agent_training_tags",
                    source_task="training",
                    summary="qwen_agent_4 tags",
                    detail="",
                    tags=["arithmetic", "final-answer", "verification"],
                ),
            ]

            report = run_four_agent_voting_on_test(
                dataset_dir,
                models=models,
                server_model=server_model,
                server_training_tag_records=server_tag_records,
            )

            prediction = report.predictions[0]
            self.assertEqual(prediction.selected_agents, ["qwen_agent_2", "qwen_agent_3", "qwen_agent_4"])
            self.assertEqual([vote.agent_name for vote in prediction.votes], prediction.selected_agents)
            self.assertEqual(prediction.final_answer, "99")
            self.assertEqual(models["qwen_agent_1"].questions_seen, [])
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

    def test_four_agents_generate_each_question_in_parallel(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_parallel_vote_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": "What is 20 + 22?", "answer": "#### 42"}) + "\n",
                encoding="utf-8",
            )
            barrier = threading.Barrier(len(AGENT_NAMES))
            models = {
                agent_name: ConcurrentVoteModel(barrier)
                for agent_name in AGENT_NAMES
            }

            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.total, 1)
            self.assertEqual(report.correct, 1)
            self.assertEqual([vote.agent_name for vote in report.predictions[0].votes], list(AGENT_NAMES))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parallel_agents_use_only_their_own_professional_library(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_private_library_vote_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            question = "What is 20 + 22?"
            (dataset_dir / "test.jsonl").write_text(
                json.dumps({"question": question, "answer": "#### 42"}) + "\n",
                encoding="utf-8",
            )
            data_dir = temp_dir / "agent_data"
            lora_output_dir = temp_dir / "lora_adapters"
            engine = CapturingKnowledgeEngine()
            models: dict[str, KnowledgeLibraryVoteModel] = {}
            for agent_name in AGENT_NAMES:
                model = KnowledgeLibraryVoteModel(
                    agent_name=agent_name,
                    engine=engine,  # type: ignore[arg-type]
                    data_dir=data_dir,
                    lora_output_dir=lora_output_dir,
                )
                model.professional_library.add_record(
                    LibraryRecord(
                        agent_name=agent_name,
                        library_type="professional",
                        source_task=question,
                        summary=f"private-marker-{agent_name}",
                        detail="Use the agent's own arithmetic strategy.",
                    )
                )
                models[agent_name] = model

            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.correct, 1)
            for agent_name, prompt in engine.prompts.items():
                self.assertIn(f"private-marker-{agent_name}", prompt)
                for other_agent in set(AGENT_NAMES) - {agent_name}:
                    self.assertNotIn(f"private-marker-{other_agent}", prompt)
                self.assertEqual(models[agent_name].adapter_path.parent.name, agent_name)
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

    def test_knowledge_library_vote_model_uses_agent_adapter(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_adapter_vote_test_"))
        try:
            data_dir = temp_dir / "data"
            lora_output_dir = temp_dir / "lora_adapters"
            agent_name = "qwen_agent_1"
            agent_dir = data_dir / agent_name
            agent_dir.mkdir(parents=True)
            (agent_dir / "professional_library.jsonl").write_text("", encoding="utf-8")
            (agent_dir / "evaluation_library.jsonl").write_text("", encoding="utf-8")

            adapter_dir = lora_output_dir / agent_name / "adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            model = KnowledgeLibraryVoteModel(
                agent_name=agent_name,
                engine=LocalQwenEngine(),
                data_dir=data_dir,
                lora_output_dir=lora_output_dir,
            )
            self.assertEqual(model.adapter_path, adapter_dir)
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
        server_tag_file = data_dir / "qwen_server_agent" / "agent_training_tags.jsonl"
        global_eval_file = data_dir / "qwen_server_agent" / "global_evaluation_library.jsonl"
        if not server_tag_file.exists():
            self.skipTest(f"missing server agent tag library: {server_tag_file}")

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
        server_tag_records = _load_library_records(server_tag_file)
        prior_global_records = _load_library_records(global_eval_file) if global_eval_file.exists() else []
        try:
            report = run_four_agent_voting_on_test(
                dataset_dir=dataset_dir,
                models=models,
                limit=STANDARD_TEST_LIMIT,
                on_prediction=print_voting_prediction_detail,
                server_model=KnowledgeServerRoutingModel(engine),
                server_training_tag_records=server_tag_records,
                prior_global_evaluation_records=prior_global_records,
            )
        except RuntimeError as exc:
            if "_spropack" in str(exc):
                self.skipTest(f"local scipy/transformers environment cannot load Qwen: {exc}")
            raise
        output_file = project_root / "train" / "four_agent_lora_voting_100_report.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.assertEqual(report.total, STANDARD_TEST_LIMIT)
        self.assertEqual(len(report.predictions), STANDARD_TEST_LIMIT)
        self.assertEqual(len(report.predictions[0].votes), 3)
        self.assertEqual(report.predictions[0].votes[0].agent_name, report.predictions[0].selected_agents[0])
        self.assertTrue(report.predictions[0].server_diagnostic)
        self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
