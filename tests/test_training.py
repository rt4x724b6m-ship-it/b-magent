from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from baseline.qwen_gsm8k import STANDARD_TEST_LIMIT
from train.four_agent_private_train import (
    STANDARD_PRIVATE_TRAIN_SIZE,
    build_participant_schedule,
    export_report,
    main,
    parse_args,
    run_b_magent_training_entry,
    run_four_agent_private_training,
)


class ReleasingBackend:
    def __init__(self) -> None:
        self.release_calls = 0

    def solve(
        self,
        agent_name: str,
        specialty: str,
        task: str,
        private_training: list[str],
        professional_memory: list[str],
        evaluation_alerts: list[str],
    ) -> tuple[str, list[str]]:
        return "answer #### 0", []

    def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):  # type: ignore[no-untyped-def]
        from b_magent.models import PeerEvaluation

        return PeerEvaluation(evaluator_name, target_draft.agent_name, ["ok"], "ok", [])

    def release_model_memory(self) -> None:
        self.release_calls += 1


class NoopLoraManager:
    def update_from_round(self, task, drafts, peer_reviews, self_improvements):  # type: ignore[no-untyped-def]
        return []


class FourAgentPrivateTrainingTestCase(unittest.TestCase):
    def test_four_agents_train_separately_for_three_rounds(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_training_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": "q0", "answer": "a0 #### 0"},
                {"question": "q1", "answer": "a1 #### 1"},
                {"question": "q2", "answer": "a2 #### 2"},
                {"question": "q3", "answer": "a3 #### 3"},
                {"question": "q4", "answer": "a4 #### 4"},
                {"question": "q5", "answer": "a5 #### 5"},
                {"question": "q6", "answer": "a6 #### 6"},
                {"question": "q7", "answer": "a7 #### 7"},
            ]
            test_rows = [
                {"question": "q0", "answer": "a0 #### 0"},
                {"question": "q1", "answer": "a1 #### 1"},
                {"question": "q2", "answer": "a2 #### 2"},
                {"question": "unseen", "answer": "missing #### 99"},
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            report = run_four_agent_private_training(
                dataset_dir=dataset_dir,
                rounds=3,
                batches_per_round=32,
                batch_size=1,
                private_train_size=2,
            )

            self.assertEqual(report.rounds, 3)
            self.assertEqual(report.batches_per_round, 32)
            self.assertEqual(report.batch_size, 1)
            self.assertEqual(report.train_total, 8)
            self.assertEqual(report.test_total, 4)
            self.assertEqual(len(report.agents), 4)
            self.assertEqual(
                [agent.agent_name for agent in report.agents],
                ["qwen_agent_1", "qwen_agent_2", "qwen_agent_3", "qwen_agent_4"],
            )
            self.assertEqual([agent.private_train_samples for agent in report.agents], [2, 2, 2, 2])

            for agent in report.agents:
                self.assertEqual(len(agent.rounds), 3)
                self.assertEqual(agent.rounds[-1].trained_batches, 96)
                self.assertEqual(agent.rounds[-1].test_total, 4)
                self.assertGreaterEqual(agent.final_accuracy, 0.0)
                self.assertLessEqual(agent.final_accuracy, 1.0)

            output_file = temp_dir / "train" / "report.json"
            export_report(report, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["rounds"], 3)
            self.assertEqual(payload["batches_per_round"], 32)
            self.assertIn("final_accuracy", payload["agents"][0])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_training_evaluates_first_100_official_test_questions_by_default(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_training_limit_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"a{i} #### {i}"}
                for i in range(4)
            ]
            test_rows = [
                {"question": f"test-{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_TEST_LIMIT + 1)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            report = run_four_agent_private_training(
                dataset_dir=dataset_dir,
                rounds=1,
                batches_per_round=1,
                batch_size=1,
                private_train_size=1,
            )

            self.assertEqual(report.test_total, STANDARD_TEST_LIMIT)
            self.assertTrue(all(agent.rounds[-1].test_total == STANDARD_TEST_LIMIT for agent in report.agents))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_training_defaults_to_200_private_samples_per_agent_without_overlap(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_training_private_size_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_PRIVATE_TRAIN_SIZE * 4)
            ]
            test_rows = [{"question": "test-0", "answer": "a0 #### 0"}]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            report = run_four_agent_private_training(
                dataset_dir=dataset_dir,
                rounds=1,
                batches_per_round=1,
                batch_size=1,
            )

            self.assertEqual(
                [agent.private_train_samples for agent in report.agents],
                [STANDARD_PRIVATE_TRAIN_SIZE] * 4,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_training_rejects_too_few_samples_for_default_private_size(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_training_private_size_error_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"a{i} #### {i}"}
                for i in range(STANDARD_PRIVATE_TRAIN_SIZE * 4 - 1)
            ]
            test_rows = [{"question": "test-0", "answer": "a0 #### 0"}]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in test_rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "need at least 800 training samples"):
                run_four_agent_private_training(
                    dataset_dir=dataset_dir,
                    rounds=1,
                    batches_per_round=1,
                    batch_size=1,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cli_can_disable_distillation_while_keeping_lora(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py", "--disable-distillation"]):
            args = parse_args()

        self.assertTrue(args.enable_lora)
        self.assertFalse(args.enable_distillation)

    def test_b_magent_training_evenly_splits_prepared_train_dataset_to_agents(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_even_private_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"reasoning {i} #### {i}"}
                for i in range(10)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )

            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=1,
                random_seed=1,
                backend=None,
            )

            self.assertEqual(
                report.private_dataset_counts,
                {
                    "qwen_agent_1": 3,
                    "qwen_agent_2": 3,
                    "qwen_agent_3": 2,
                    "qwen_agent_4": 2,
                },
            )
            all_private_questions = []
            for agent_name, expected_count in report.private_dataset_counts.items():
                private_file = temp_dir / "data" / agent_name / "private_data.jsonl"
                self.assertTrue(private_file.exists())
                lines = private_file.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), expected_count)
                all_private_questions.extend(lines)
            self.assertEqual(len(all_private_questions), 10)
            self.assertEqual(len(set(all_private_questions)), 10)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_auto_rounds_cover_even_private_splits(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_auto_rounds_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"reasoning {i} #### {i}"}
                for i in range(10)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )

            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=None,
                private_batch_size=1,
                random_seed=1,
                backend=None,
            )

            self.assertEqual(report.rounds, 5)
            trained_slots = {}
            for round_report in report.training_rounds:
                for agent_name in round_report.participants:
                    trained_slots[agent_name] = trained_slots.get(agent_name, 0) + 1
            self.assertEqual(trained_slots, report.private_dataset_counts)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_releases_backend_memory_before_adapter_training(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_release_memory_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"train-{i}", "answer": f"reasoning {i} #### {i}"}
                for i in range(4)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            backend = ReleasingBackend()

            run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=2,
                private_batch_size=1,
                random_seed=1,
                backend=backend,
                lora_manager=NoopLoraManager(),
            )

            self.assertEqual(backend.release_calls, 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_participant_schedule_uses_distinct_agents_per_round(self) -> None:
        schedule = build_participant_schedule(
            {
                "qwen_agent_1": 3,
                "qwen_agent_2": 3,
                "qwen_agent_3": 2,
                "qwen_agent_4": 2,
            },
            private_batch_size=1,
        )

        self.assertEqual(len(schedule), 5)
        self.assertTrue(all(len(set(pair)) == 2 for pair in schedule))

    def test_cli_defaults_enable_lora_distillation_and_200_rounds(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py"]):
            args = parse_args()

        self.assertEqual(args.rounds, 200)
        self.assertTrue(args.dataset_dir.is_absolute())
        self.assertTrue(args.output.is_absolute())
        self.assertTrue(args.lora_output_dir.is_absolute())
        self.assertTrue(args.enable_lora)
        self.assertTrue(args.enable_distillation)

    def test_cli_disabling_lora_also_disables_distillation(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py", "--disable-lora"]):
            args = parse_args()

        self.assertFalse(args.enable_lora)
        self.assertFalse(args.enable_distillation)

    def test_b_magent_main_resets_training_state_before_training(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_main_reset_test_"))
        try:
            output_file = temp_dir / "report.json"
            lora_output_dir = temp_dir / "lora_adapters"
            with (
                patch(
                    "sys.argv",
                    [
                        "four_agent_private_train.py",
                        "--mode",
                        "b-magent",
                        "--backend",
                        "demo",
                        "--dataset-dir",
                        str(temp_dir / "gsm8k"),
                        "--output",
                        str(output_file),
                        "--lora-output-dir",
                        str(lora_output_dir),
                    ],
                ),
                patch("train.four_agent_private_train.reset_b_magent_training_state") as reset,
                patch("train.four_agent_private_train.run_b_magent_training_entry") as run_training,
                patch("train.four_agent_private_train.export_json_report"),
            ):
                run_training.return_value.to_dict.return_value = {
                    "agents": [],
                    "rounds": 0,
                    "professional_records": {},
                    "evaluation_records": {},
                    "lora_updates": {},
                    "distillation_enabled": False,
                    "distilled_adapter_paths": {},
                    "distillation_updates": {},
                }
                run_training.return_value.agents = []
                run_training.return_value.rounds = 0
                run_training.return_value.distillation_enabled = False

                main()

            reset.assert_called_once()
            self.assertTrue(reset.call_args.kwargs["reset_evaluation_libraries"])
            self.assertEqual(reset.call_args.kwargs["lora_output_dir"], lora_output_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
