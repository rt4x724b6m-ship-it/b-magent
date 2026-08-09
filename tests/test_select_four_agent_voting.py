from __future__ import annotations

import json
import random
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
    TRAINING_EVALUATION_LIMIT,
    format_voting_prediction_detail,
    print_voting_prediction_detail,
    reset_b_magent_training_state,
    routed_vote,
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
    def test_routed_vote_uses_matching_second_and_third_ranked_answers(self) -> None:
        votes = [
            type("Vote", (), {"predicted_answer": "11", "tag_match_score": 0.9})(),
            type("Vote", (), {"predicted_answer": "42", "tag_match_score": 0.8})(),
            type("Vote", (), {"predicted_answer": "42", "tag_match_score": 0.7})(),
        ]

        self.assertEqual(routed_vote(votes), "42")

    def test_routed_vote_uses_top_ranked_answer_when_second_and_third_disagree(self) -> None:
        votes = [
            type("Vote", (), {"predicted_answer": "11", "tag_match_score": 0.9})(),
            type("Vote", (), {"predicted_answer": "42", "tag_match_score": 0.8})(),
            type("Vote", (), {"predicted_answer": "7", "tag_match_score": 0.7})(),
        ]

        self.assertEqual(routed_vote(votes), "11")

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

    def test_six_agents_vote_final_answer_on_test_dataset(self) -> None:
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
                "qwen_agent_5": FixedVoteModel(["42", "7", "14"]),
                "qwen_agent_6": FixedVoteModel(["0", "8", "15"]),
            }
            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.total, 3)
            self.assertEqual(report.correct, 2)
            self.assertEqual(report.accuracy, 2 / 3)
            self.assertEqual([vote.agent_name for vote in report.predictions[0].votes], list(AGENT_NAMES))
            self.assertEqual(report.predictions[0].final_answer, "42")
            self.assertTrue(report.predictions[0].correct)

            # The second row is a 3-3 tie; agent order resolves it to 7.
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
                "qwen_agent_5": FixedVoteModel(["16"]),
                "qwen_agent_6": FixedVoteModel(["16"]),
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
                "qwen_agent_5": FixedRecordingVoteModel("0"),
                "qwen_agent_6": FixedRecordingVoteModel("0"),
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
                "qwen_agent_5": FixedRecordingVoteModel("5"),
                "qwen_agent_6": FixedRecordingVoteModel("6"),
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

    def test_voting_uses_full_split_when_limit_is_zero(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_full_vote_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            rows = [
                {"question": f"q{index}", "answer": f"#### {index}"}
                for index in range(STANDARD_TEST_LIMIT + 1)
            ]
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            models = {agent_name: FixedVoteModel([str(index) for index in range(len(rows))]) for agent_name in AGENT_NAMES}

            report = run_four_agent_voting_on_test(dataset_dir, models=models, limit=0)

            self.assertEqual(report.total, len(rows))
            self.assertEqual(report.correct, len(rows))
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

        data_list = [
            json.loads(line)
            for line in (dataset_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        random.seed(2024)
        random.shuffle(data_list)
        data_list = data_list[:500]

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
        with tempfile.TemporaryDirectory(prefix="b_magent_sampled_gsm8k_") as sampled_data_dir:
            sampled_dataset_dir = Path(sampled_data_dir)
            (sampled_dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in data_list) + "\n",
                encoding="utf-8",
            )
            try:
                report = run_four_agent_voting_on_test(
                    dataset_dir=sampled_dataset_dir,
                    models=models,
                    limit=TRAINING_EVALUATION_LIMIT,
                    on_prediction=print_voting_prediction_detail,
                    server_model=KnowledgeServerRoutingModel(engine),
                    server_training_tag_records=server_tag_records,
                    prior_global_evaluation_records=prior_global_records,
                )
            except RuntimeError as exc:
                if "_spropack" in str(exc):
                    self.skipTest(f"local scipy/transformers environment cannot load Qwen: {exc}")
                raise
        output_file = project_root / "train" / "four_agent_lora_voting_500_report.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.assertEqual(report.total, TRAINING_EVALUATION_LIMIT)
        self.assertEqual(len(report.predictions), TRAINING_EVALUATION_LIMIT)
        self.assertEqual(len(report.predictions[0].votes), 3)
        self.assertEqual(report.predictions[0].votes[0].agent_name, report.predictions[0].selected_agents[0])
        self.assertTrue(report.predictions[0].server_diagnostic)
        self.assertTrue(output_file.exists())


# ──────────────────────────────────────────────────────────────────────────────
# TravelPlanner 适配测试
# ──────────────────────────────────────────────────────────────────────────────

class TravelPlannerTaggingTestCase(unittest.TestCase):
    """tagging.py — 旅行标签、别名和权重测试。"""

    def setUp(self) -> None:
        from b_magent.tagging import ROUTING_TAGS, ROUTING_TAG_IMPORTANCE, extract_math_task_tags
        self.ROUTING_TAGS = ROUTING_TAGS
        self.ROUTING_TAG_IMPORTANCE = ROUTING_TAG_IMPORTANCE
        self.extract = extract_math_task_tags

    def test_travel_tags_present_in_routing_tags(self) -> None:
        expected = {
            "itinerary-planning", "transportation", "accommodation", "attraction",
            "restaurant", "budget-constraint", "local-constraint", "multi-city",
            "multi-day", "route-optimization", "feasibility-check", "commonsense-travel",
        }
        missing = expected - set(self.ROUTING_TAGS)
        self.assertEqual(missing, set(), f"Missing travel tags: {missing}")

    def test_travel_tag_weights_are_higher_than_generic_math_tags(self) -> None:
        # itinerary-planning should outweigh arithmetic
        self.assertGreater(
            self.ROUTING_TAG_IMPORTANCE.get("itinerary-planning", 0),
            self.ROUTING_TAG_IMPORTANCE.get("arithmetic", 1),
        )
        self.assertGreater(
            self.ROUTING_TAG_IMPORTANCE.get("budget-constraint", 0),
            self.ROUTING_TAG_IMPORTANCE.get("arithmetic", 1),
        )

    def test_travel_query_hits_budget_and_commonsense_tags(self) -> None:
        query = "Plan a 3-day trip from NYC to Chicago with a budget of $1500"
        tags = self.extract(query)
        self.assertIn("budget-constraint", tags)
        self.assertIn("commonsense-travel", tags)

    def test_travel_query_hits_transportation_tag(self) -> None:
        query = "Book a flight from Boston to Denver, departure at 8am"
        tags = self.extract(query)
        self.assertIn("transportation", tags)

    def test_travel_query_hits_accommodation_tag(self) -> None:
        query = "Find a hotel for 2 nights in San Francisco"
        tags = self.extract(query)
        self.assertIn("accommodation", tags)

    def test_travel_query_hits_feasibility_tag(self) -> None:
        query = "Check if the schedule is feasible given the flight connection time"
        tags = self.extract(query)
        self.assertIn("feasibility-check", tags)

    def test_travel_query_hits_itinerary_planning_tag(self) -> None:
        query = "Create a day-by-day itinerary for my trip"
        tags = self.extract(query)
        self.assertIn("itinerary-planning", tags)

    def test_math_tags_still_work_after_travel_tag_additions(self) -> None:
        # Use explicit arithmetic operators so the equation-based rules fire
        query = "Janet earns $50 per hour. She works 8*5 hours. How much does she earn?"
        tags = self.extract(query)
        self.assertIn("money", tags)
        self.assertIn("multiplication", tags)


class TravelPlannerLoraTestCase(unittest.TestCase):
    """lora.py — travel SFT 样本构建与 Gold plan 剥离测试。"""

    def setUp(self) -> None:
        from b_magent.lora import (
            build_lora_example, strip_gold_annotations, _is_travel_task,
            _build_travel_supervision_target,
        )
        from b_magent.models import Draft, EvaluationScores, PeerEvaluation, SelfImprovement
        self.build_lora_example = build_lora_example
        self.strip_gold_annotations = strip_gold_annotations
        self._is_travel_task = _is_travel_task
        self._build_travel_supervision_target = _build_travel_supervision_target
        self.Draft = Draft
        self.PeerEvaluation = PeerEvaluation
        self.EvaluationScores = EvaluationScores
        self.SelfImprovement = SelfImprovement

    def _make_draft(self, agent_name: str = "qwen_agent_1") -> object:
        return self.Draft(
            agent_name=agent_name,
            specialty="通用智能体",
            answer="Day 1: fly to Chicago",
            thought_trace=["parse query", "plan transport"],
            tool_calls=[],
            private_training_used=[],
            professional_memory_used=[],
            evaluation_alerts_used=[],
        )

    def _make_improvement(self, revised: str = "Day 1: fly. Day 2: museum.") -> object:
        return self.SelfImprovement(
            agent_name="qwen_agent_1",
            applied_suggestions=["add accommodation"],
            revised_answer=revised,
            professional_updates=[],
            reflection="Added accommodation to each day.",
        )

    def _make_evaluation(self) -> object:
        return self.PeerEvaluation(
            evaluator="qwen_agent_2",
            target="qwen_agent_1",
            suggestions=["add accommodation"],
            rationale="Missing accommodation on day 2.",
            evaluation_memory_used=[],
            scores=self.EvaluationScores(correctness=0.9, safety=1.0, efficiency=0.8),
        )

    def test_is_travel_task_detects_origin_header(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 3\nQuery: Plan my trip"
        self.assertTrue(self._is_travel_task(task))

    def test_is_travel_task_detects_query_header(self) -> None:
        task = "Query: Plan a 3-day trip to Seattle"
        self.assertTrue(self._is_travel_task(task))

    def test_is_travel_task_detects_gold_plan_header(self) -> None:
        task = "Gold plan: [day1: fly, day2: museum]"
        self.assertTrue(self._is_travel_task(task))

    def test_is_travel_task_returns_false_for_math(self) -> None:
        task = "Solve this GSM8K training problem.\nQuestion: How many apples?\nGold final answer: 5"
        self.assertFalse(self._is_travel_task(task))

    def test_strip_gold_annotations_removes_gold_plan(self) -> None:
        task = (
            "Origin: NYC  Destination: Chicago  Days: 3\n"
            "Query: Plan my trip\n"
            "Gold plan: [day1: fly, day2: museum, day3: return]"
        )
        stripped = self.strip_gold_annotations(task)
        self.assertNotIn("Gold plan:", stripped)
        self.assertIn("Query: Plan my trip", stripped)

    def test_strip_gold_annotations_still_removes_gold_reasoning(self) -> None:
        task = (
            "Question: How many apples?\n"
            "Gold reasoning: Janet has 3 apples...\n"
            "more reasoning\n"
            "Gold final answer: 3"
        )
        stripped = self.strip_gold_annotations(task)
        self.assertNotIn("Gold reasoning:", stripped)
        self.assertNotIn("more reasoning", stripped)
        self.assertNotIn("Gold final answer:", stripped)
        self.assertIn("Question:", stripped)

    def test_strip_gold_annotations_removes_gold_image_elements(self) -> None:
        task = "Image: /path/img.png\nQuestion: What color?\nGold image elements: {}"
        stripped = self.strip_gold_annotations(task)
        self.assertNotIn("Gold image elements:", stripped)

    def test_build_lora_example_travel_uses_gold_plan_as_output(self) -> None:
        task = (
            "Origin: NYC  Destination: Chicago  Days: 2\n"
            "Query: Plan my trip\n"
            "Gold plan: Day 1: fly. Day 2: museum."
        )
        example = self.build_lora_example(
            task,
            self._make_draft(),
            [self._make_evaluation()],
            self._make_improvement(),
        )
        self.assertIn("travel planning agent", example.instruction.lower())
        self.assertIn("Day 1: fly", example.output)
        self.assertNotIn("####", example.output)
        self.assertEqual(example.image, "")

    def test_build_lora_example_travel_falls_back_to_revised_answer(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 2\nQuery: Plan my trip"
        revised = "Day 1: flight. Day 2: sightseeing."
        example = self.build_lora_example(
            task,
            self._make_draft(),
            [self._make_evaluation()],
            self._make_improvement(revised=revised),
        )
        self.assertEqual(example.output, revised)

    def test_build_lora_example_math_unchanged(self) -> None:
        task = (
            "Solve this GSM8K training problem.\n"
            "Question: How many apples?\n"
            "Gold reasoning: 3 apples.\n"
            "Gold final answer: 3"
        )
        from b_magent.models import SelfImprovement
        improvement = SelfImprovement(
            agent_name="qwen_agent_1",
            applied_suggestions=[],
            revised_answer="The answer is 3.\n#### 3",
            professional_updates=[],
            reflection="",
        )
        example = self.build_lora_example(task, self._make_draft(), [self._make_evaluation()], improvement)
        self.assertIn("####", example.output)
        self.assertIn("math problem", example.instruction.lower())


class TravelPlannerAgentTestCase(unittest.TestCase):
    """agent.py — Query: 识别、JSON私有数据提取、Gold plan: 剥离、反思模板测试。"""

    def _extract_task_question(self, task: str) -> str:
        import importlib, b_magent.agent as m
        importlib.reload(m)
        return m._extract_task_question(task)

    def _extract_private_question(self, item: str) -> str:
        import b_magent.agent as m
        return m._extract_private_question(item)

    def _strip_gold_annotations(self, task: str) -> str:
        import b_magent.agent as m
        return m._strip_gold_annotations(task)

    def _build_private_training_reflection(self, specialty: str, batch: list) -> str:
        import b_magent.agent as m
        return m._build_private_training_reflection(specialty, batch)

    def test_extract_task_question_handles_query_field(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 3\nQuery: Plan a 3-day trip\nGold plan: ..."
        self.assertEqual(self._extract_task_question(task), "Plan a 3-day trip")

    def test_extract_task_question_still_handles_question_field(self) -> None:
        task = "Solve this GSM8K problem.\nQuestion: How many apples?\nGold final answer: 3"
        self.assertEqual(self._extract_task_question(task), "How many apples?")

    def test_extract_private_question_parses_json_travel_sample(self) -> None:
        sample = json.dumps({
            "question": "Plan a trip from NYC to Chicago",
            "answer": "Day 1: fly.",
            "level": "easy",
            "org": "NYC",
            "dest": "Chicago",
            "days": 3,
        })
        result = self._extract_private_question(sample)
        self.assertEqual(result, "Plan a trip from NYC to Chicago")

    def test_extract_private_question_still_handles_pipe_format(self) -> None:
        item = "GSM8K sample | question: How many apples? | reasoning_answer: 3 apples | final_answer: 3"
        result = self._extract_private_question(item)
        self.assertEqual(result, "How many apples?")

    def test_strip_gold_annotations_removes_gold_plan_in_agent(self) -> None:
        task = "Origin: NYC\nQuery: Plan trip\nGold plan: [day1: fly]"
        stripped = self._strip_gold_annotations(task)
        self.assertNotIn("Gold plan:", stripped)
        self.assertIn("Query: Plan trip", stripped)

    def test_private_training_reflection_travel_mentions_budget_and_constraints(self) -> None:
        batch = [json.dumps({
            "question": "Plan a 3-day trip from NYC to Chicago",
            "answer": "Day 1: fly.",
            "level": "easy",
            "org": "NYC",
            "dest": "Chicago",
            "days": 3,
        })]
        reflection = self._build_private_training_reflection("通用智能体", batch)
        self.assertIn("budget feasibility", reflection)
        self.assertIn("constraint", reflection)
        self.assertIn("day-by-day", reflection)

    def test_private_training_reflection_math_unchanged(self) -> None:
        batch = ["GSM8K sample | question: How many apples? | reasoning_answer: 3 apples | final_answer: 3"]
        reflection = self._build_private_training_reflection("通用智能体", batch)
        self.assertIn("final-answer format", reflection)
        self.assertIn("verify calculations", reflection)

    def test_private_training_reflection_empty_batch(self) -> None:
        reflection = self._build_private_training_reflection("通用智能体", [])
        self.assertIn("no private sample", reflection)


class TravelPlannerLibraryTestCase(unittest.TestCase):
    """library.py — _semantic_terms 停用词和 gold 剥离测试。"""

    def _semantic_terms(self, text: str) -> set:
        from b_magent.library import _semantic_terms
        return _semantic_terms(text)

    def test_gold_plan_content_not_in_terms(self) -> None:
        task = (
            "Origin: NYC  Destination: Chicago  Days: 3\n"
            "Query: Plan a trip\n"
            "Gold plan: [day1: fly to chicago, stay at marriott hotel]"
        )
        terms = self._semantic_terms(task)
        # gold plan place names should be stripped
        self.assertNotIn("marriott", terms)

    def test_travel_stop_words_excluded(self) -> None:
        text = "Please help me plan a trip from NYC to Chicago"
        terms = self._semantic_terms(text)
        # generic travel stop-words should be filtered
        for stop in ("plan", "trip", "please", "help"):
            self.assertNotIn(stop, terms, f"stop word '{stop}' should be excluded")

    def test_origin_and_destination_retained_as_retrieval_terms(self) -> None:
        text = "Origin: Seattle  Destination: Portland  Days: 2\nQuery: Book a flight"
        terms = self._semantic_terms(text)
        self.assertIn("seattle", terms)
        self.assertIn("portland", terms)

    def test_gold_final_answer_excluded_from_terms(self) -> None:
        task = "Question: How many apples?\nGold final answer: 42"
        terms = self._semantic_terms(task)
        self.assertNotIn("42", terms)

    def test_math_gold_reasoning_excluded_from_terms(self) -> None:
        task = "Question: How many apples?\nGold reasoning: Janet has 3 apples plus 2 more.\nGold final answer: 5"
        terms = self._semantic_terms(task)
        self.assertNotIn("janet", terms)


class TravelPlannerSelfEvolutionTestCase(unittest.TestCase):
    """self_evolution.py — 旅行/数学任务反思模板分支测试。"""

    def _make_event(self, task: str, is_correct: bool | None = None) -> object:
        from b_magent.self_evolution import EvolutionInput
        return EvolutionInput(
            agent_name="qwen_agent_1",
            specialty="通用智能体",
            task=task,
            answer="some answer",
            is_correct=is_correct,
        )

    def _reflect(self, task: str, suggestions: list, is_correct: bool | None = None) -> str:
        from b_magent.self_evolution import _build_professional_reflection
        event = self._make_event(task, is_correct)
        return _build_professional_reflection(event, suggestions)

    def test_travel_reflection_mentions_budget_feasibility(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 3\nQuery: Plan a 3-day budget trip"
        reflection = self._reflect(task, ["check budget"])
        self.assertIn("budget feasibility", reflection)

    def test_travel_reflection_mentions_constraint_satisfaction(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 3\nQuery: Plan trip with dietary constraints"
        reflection = self._reflect(task, [])
        self.assertIn("constraint satisfaction", reflection)

    def test_travel_reflection_does_not_say_verify_final_answer(self) -> None:
        task = "Origin: NYC  Destination: Chicago  Days: 3\nQuery: Plan my trip"
        reflection = self._reflect(task, [])
        self.assertNotIn("verify the final answer", reflection)

    def test_math_reflection_still_mentions_verify_final_answer(self) -> None:
        task = "Solve this GSM8K problem.\nQuestion: How many apples?\nGold final answer: 5"
        reflection = self._reflect(task, [])
        self.assertIn("verify the final answer", reflection)

    def test_travel_reflection_with_correct_outcome_labels_curated(self) -> None:
        task = "Origin: BOS  Destination: SEA  Days: 4\nQuery: Plan a 4-day trip"
        reflection = self._reflect(task, [], is_correct=True)
        self.assertIn("success", reflection)
        self.assertIn("curated-by-evaluation", reflection)

    def test_travel_reflection_with_wrong_outcome_labels_error(self) -> None:
        task = "Origin: BOS  Destination: SEA  Days: 4\nQuery: Plan a 4-day trip"
        reflection = self._reflect(task, [], is_correct=False)
        self.assertIn("error", reflection)
        self.assertIn("reflection-from-error", reflection)


class TravelPlannerVotingEndToEndTestCase(unittest.TestCase):
    """四智能体投票在 TravelPlanner 数据上的端到端集成测试。"""

    def setUp(self) -> None:
        import tempfile
        self.temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_travel_vote_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_travel_split(self, name: str, rows: list[dict]) -> Path:
        split_dir = self.temp_dir / "TravelPlanner"
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / name
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return split_dir

    def test_travel_voting_runs_and_produces_correct_report(self) -> None:
        from train.four_agent_private_train import run_four_agent_voting_on_test

        rows = [
            {
                "query": "Plan a 3-day trip from NYC to Chicago with a budget of $1500.",
                "annotated_plan": "Day 1: fly to Chicago. Day 2: visit museums. Day 3: return.",
                "level": "easy",
                "org": "NYC",
                "dest": "Chicago",
                "days": 3,
                "visiting_city_number": 1,
                "date": ["2022-03-16", "2022-03-17", "2022-03-18"],
                "people_number": 1,
                "local_constraint": {},
                "budget": 1500,
            },
            {
                "query": "Plan a 2-day trip from Boston to Seattle.",
                "annotated_plan": "Day 1: fly to Seattle. Day 2: explore and return.",
                "level": "easy",
                "org": "Boston",
                "dest": "Seattle",
                "days": 2,
                "visiting_city_number": 1,
                "date": ["2022-04-01", "2022-04-02"],
                "people_number": 2,
                "local_constraint": {},
                "budget": 1000,
            },
        ]
        dataset_dir = self._write_travel_split("test_test.json", rows)

        # Model that echoes the gold plan back verbatim for correct scoring
        class TravelEchoModel:
            def __init__(self, plans: list[str]) -> None:
                self.plans = plans
                self.idx = 0
            def train_batch(self, _batch: object) -> None:
                return None
            def generate(self, _question: str) -> str:
                plan = self.plans[self.idx % len(self.plans)]
                self.idx += 1
                return plan

        gold_plans = [r["annotated_plan"] for r in rows]
        models = {agent_name: TravelEchoModel(gold_plans) for agent_name in AGENT_NAMES}

        report = run_four_agent_voting_on_test(dataset_dir, models=models, split="test")

        self.assertEqual(report.total, 2)
        self.assertEqual(report.evaluation_split, "test")
        self.assertEqual(len(report.predictions), 2)
        # Each prediction should carry the correct question text
        self.assertIn("NYC to Chicago", report.predictions[0].question)
        self.assertIn("Boston to Seattle", report.predictions[1].question)

    def test_travel_voting_uses_travel_tags_for_routing(self) -> None:
        from train.four_agent_private_train import run_four_agent_voting_on_test

        rows = [{
            "query": "Plan a 3-day trip from NYC to Chicago with a budget of $1500.",
            "annotated_plan": "Day 1: fly. Day 2: hotel. Day 3: return.",
            "level": "easy",
            "org": "NYC",
            "dest": "Chicago",
            "days": 3,
            "visiting_city_number": 1,
            "date": ["2022-03-16", "2022-03-17", "2022-03-18"],
            "people_number": 1,
            "local_constraint": {},
            "budget": 1500,
        }]
        dataset_dir = self._write_travel_split("test_test.json", rows)

        server_model = RecordingServerRoutingModel(
            json.dumps({
                "difficulty": "medium",
                "key_steps": ["select flight", "book hotel", "plan attractions"],
                "risk_steps": ["check budget constraint"],
                "capability_tags": ["itinerary-planning", "budget-constraint", "transportation"],
                "risk_tags": ["feasibility-check"],
            })
        )
        server_tag_records = [
            LibraryRecord(
                agent_name=agent_name,
                library_type="agent_training_tags",
                source_task="travel planning task",
                summary="travel training tags",
                detail="source_library_type=professional",
                tags=[
                    agent_name, "agent-training-tags", "professional",
                    "itinerary-planning", "budget-constraint", "transportation",
                ],
            )
            for agent_name in AGENT_NAMES
        ]

        class FixedTravelModel:
            def train_batch(self, batch: object) -> None: return None
            def generate(self, question: str) -> str: return "Day 1: fly. Day 2: hotel."

        models = {agent_name: FixedTravelModel() for agent_name in AGENT_NAMES}

        report = run_four_agent_voting_on_test(
            dataset_dir,
            models=models,
            split="test",
            server_model=server_model,
            server_training_tag_records=server_tag_records,
        )

        prediction = report.predictions[0]
        # Server routing diagnostic should contain travel tags
        self.assertIn("itinerary-planning", prediction.routing_tags)
        self.assertIn("budget-constraint", prediction.routing_tags)
        # 3 agents should be selected
        self.assertEqual(len(prediction.selected_agents), 3)

    def test_travel_strip_gold_plan_in_format_training_task(self) -> None:
        """format_training_task produces a Gold plan: line that strip_gold_annotations removes."""
        from b_magent.datasets import TravelPlannerSample
        from train.four_agent_private_train import format_training_task
        from b_magent.lora import strip_gold_annotations

        sample = TravelPlannerSample(
            question="Plan a 3-day trip from NYC to Chicago.",
            answer="Day 1: fly. Day 2: museum. Day 3: return.",
            final_answer="Day 1: fly. Day 2: museum. Day 3: return.",
            level="easy",
            org="NYC",
            dest="Chicago",
            days=3,
        )
        task = format_training_task(sample)
        self.assertIn("Gold plan:", task)
        self.assertIn("Origin:", task)
        self.assertIn("Query:", task)

        stripped = strip_gold_annotations(task)
        self.assertNotIn("Gold plan:", stripped)
        self.assertIn("Query:", stripped)


if __name__ == "__main__":
    unittest.main()
