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
from b_magent.agent import QwenAgent
from train.four_agent_private_train import (
    AGENT_NAMES,
    DEFAULT_DATASET_DIR,
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
    def __init__(self) -> None:
        self.flush_calls = 0

    def update_from_round(self, task, drafts, peer_reviews, self_improvements):  # type: ignore[no-untyped-def]
        return []

    def flush_pending(self, agent_names):  # type: ignore[no-untyped-def]
        self.flush_calls += 1
        raise AssertionError("training entry must not bypass the LoRA accumulation threshold")


class FinalizingLoraManager:
    def __init__(self) -> None:
        self.flush_calls = 0

    def update_from_round(self, task, drafts, peer_reviews, self_improvements):  # type: ignore[no-untyped-def]
        raise AssertionError("a completed resume must not run another training round")

    def flush_pending(self, agent_names):  # type: ignore[no-untyped-def]
        self.flush_calls += 1
        from b_magent.lora import LoraUpdate

        return [LoraUpdate("qwen_agent_4", "dataset", "adapter", 1, True)]


class FourAgentPrivateTrainingTestCase(unittest.TestCase):
    def test_visual_training_progress_checkpoint_takes_priority_and_is_resettable(self) -> None:
        from train.train import (
            TRAINING_PROGRESS_FILE,
            load_training_progress,
            reset_b_magent_training_state,
            save_training_progress,
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_progress_checkpoint_test_"))
        try:
            save_training_progress(temp_dir, 99)

            self.assertEqual(load_training_progress(temp_dir), 99)
            reset_b_magent_training_state(temp_dir, lora_output_dir=None)
            self.assertFalse((temp_dir / TRAINING_PROGRESS_FILE).exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_visual_training_main_resumes_without_resetting_state(self) -> None:
        from train.train import main as visual_training_main

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_visual_main_resume_test_"))
        try:
            dataset_dir = temp_dir / "TravelPlanner"
            dataset_dir.mkdir()
            (dataset_dir / "train.csv").write_text(
                "query,annotated_plan,reference_information\nq,plan,reference\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "train.py",
                        "--mode",
                        "b-magent",
                        "--backend",
                        "demo",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--resume",
                    ],
                ),
                patch("train.train.reset_b_magent_training_state") as reset,
                patch("train.train.load_training_progress", return_value=99),
                patch("train.train.run_b_magent_training_entry") as run_training,
                patch("train.train.export_json_report"),
            ):
                run_training.return_value.agents = []
                run_training.return_value.rounds = 200

                visual_training_main()

            reset.assert_not_called()
            self.assertEqual(run_training.call_args.kwargs["start_round"], 99)
            self.assertTrue(run_training.call_args.kwargs["preserve_private_datasets"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multiline_private_samples_are_stored_and_loaded_as_single_rows(self) -> None:
        from train.train import (
            DEFAULT_TRAINING_ROUNDS,
            run_b_magent_training_entry as run_visual_training_entry,
        )

        self.assertEqual(DEFAULT_TRAINING_ROUNDS, 200)

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_multiline_private_test_"))
        try:
            dataset_dir = temp_dir / "TravelPlanner"
            dataset_dir.mkdir()
            (dataset_dir / "train.csv").write_text(
                "query,annotated_plan,reference_information\n"
                '"trip one","day 1\nday 2","shared heading\nsource one"\n'
                '"trip two","day 1\nday 3","shared heading\nsource two"\n'
                '"trip three","day 1\nday 4","shared heading\nsource three"\n'
                '"trip four","day 1\nday 5","shared heading\nsource four"\n',
                encoding="utf-8",
            )
            data_dir = temp_dir / "data"

            report = run_visual_training_entry(
                dataset_dir=dataset_dir,
                data_dir=data_dir,
                rounds=1,
                backend=None,
            )

            self.assertEqual(sum(report.private_dataset_counts.values()), 4)
            private_datasets = []
            for agent_name in AGENT_NAMES:
                private_file = data_dir / agent_name / "private_data.jsonl"
                self.assertEqual(len(private_file.read_text(encoding="utf-8").splitlines()), 1)
                private_data = QwenAgent(agent_name, "test", data_dir)._load_private_data()
                self.assertEqual(len(private_data), 1)
                private_datasets.append(set(private_data))
            for index, private_dataset in enumerate(private_datasets):
                for other_private_dataset in private_datasets[index + 1 :]:
                    self.assertTrue(private_dataset.isdisjoint(other_private_dataset))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_completed_resume_flushes_pending_lora_without_another_round(self) -> None:
        from train.train import run_b_magent_training_entry as run_visual_training_entry

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_completed_resume_test_"))
        try:
            dataset_dir = temp_dir / "dataset"
            data_dir = temp_dir / "data"
            dataset_dir.mkdir()
            train_rows = [
                {"question": f"q{index}", "answer": f"a{index} #### {index}"}
                for index in range(len(AGENT_NAMES))
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            for agent_name, row in zip(AGENT_NAMES, train_rows):
                agent_dir = data_dir / agent_name
                agent_dir.mkdir(parents=True)
                (agent_dir / "private_data.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )
            lora_manager = FinalizingLoraManager()

            report = run_visual_training_entry(
                dataset_dir=dataset_dir,
                data_dir=data_dir,
                rounds=1,
                start_round=1,
                preserve_private_datasets=True,
                lora_manager=lora_manager,  # type: ignore[arg-type]
            )

            self.assertEqual(lora_manager.flush_calls, 1)
            self.assertEqual(report.training_rounds, [])
            self.assertEqual(report.lora_updates["qwen_agent_4"], 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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

    def test_cli_keeps_lora_enabled_by_default(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py"]):
            args = parse_args()

        self.assertTrue(args.enable_lora)

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

    def test_visual_training_uses_first_200_samples_per_agent(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_visual_private_limit_test_"))
        try:
            dataset_dir = temp_dir / "data" / "infographicsvqa"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {
                    "dataset": "infographicsvqa",
                    "id": str(index),
                    "image": f"images/{index}.png",
                    "question": f"visual-{index}",
                    "answers": [str(index)],
                }
                for index in range(STANDARD_PRIVATE_TRAIN_SIZE * len(AGENT_NAMES) + 5)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )

            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "agent_data",
                rounds=1,
                backend=None,
            )

            self.assertEqual(report.train_total, STANDARD_PRIVATE_TRAIN_SIZE * len(AGENT_NAMES))
            self.assertEqual(
                report.private_dataset_counts,
                {agent_name: STANDARD_PRIVATE_TRAIN_SIZE for agent_name in AGENT_NAMES},
            )
            for agent_index, agent_name in enumerate(AGENT_NAMES):
                rows = [
                    json.loads(line)
                    for line in (temp_dir / "agent_data" / agent_name / "private_data.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                start = agent_index * STANDARD_PRIVATE_TRAIN_SIZE
                self.assertEqual(rows[0]["question"], f"visual-{start}")
                self.assertEqual(rows[-1]["question"], f"visual-{start + STANDARD_PRIVATE_TRAIN_SIZE - 1}")
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

    def test_b_magent_downlinks_global_experience_before_next_round(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_global_downlink_test_"))
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

            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=2,
                private_batch_size=1,
                random_seed=1,
                backend=None,
            )

            self.assertEqual(report.training_rounds[0].global_downlinks, 0)
            self.assertEqual(report.training_rounds[0].global_uploads, 1)
            self.assertEqual(report.training_rounds[1].global_downlinks, 2)
            self.assertEqual(report.training_rounds[1].global_uploads, 1)
            self.assertTrue((temp_dir / "data" / "qwen_server_agent" / "global_evaluation_library.jsonl").exists())
            second_round_evaluators = set(report.training_rounds[1].evaluators)
            for agent_name in AGENT_NAMES:
                evaluation_path = temp_dir / "data" / agent_name / "evaluation_library.jsonl"
                evaluation_text = evaluation_path.read_text(encoding="utf-8") if evaluation_path.exists() else ""
                if agent_name in second_round_evaluators:
                    self.assertIn("global-downlink", evaluation_text)
                    self.assertIn("source_global_experience_id=qwen_server_agent:", evaluation_text)
                else:
                    self.assertNotIn("global-downlink", evaluation_text)
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
            lora_manager = NoopLoraManager()

            run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=2,
                private_batch_size=1,
                random_seed=1,
                backend=backend,
                lora_manager=lora_manager,
            )

            self.assertEqual(backend.release_calls, 0)
            self.assertEqual(lora_manager.flush_calls, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_participant_schedule_uses_distinct_agents_per_round(self) -> None:
        schedule = build_participant_schedule(
            {
                "qwen_agent_1": 3,
                "qwen_agent_2": 3,
                "qwen_agent_3": 2,
                "qwen_agent_4": 2,
                "qwen_agent_5": 2,
                "qwen_agent_6": 2,
            },
            private_batch_size=1,
            random_seed=7,
        )

        self.assertTrue(all(len(group) == 3 for group in schedule))
        self.assertTrue(all(len(set(group)) == 3 for group in schedule))
        self.assertEqual(
            schedule,
            build_participant_schedule(
                {name: 3 if name in {"qwen_agent_1", "qwen_agent_2"} else 2 for name in AGENT_NAMES},
                private_batch_size=1,
                random_seed=7,
            ),
        )

    def test_cli_defaults_enable_lora_and_auto_cover_private_data(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py"]):
            args = parse_args()

        self.assertEqual(args.rounds, 0)
        self.assertTrue(args.dataset_dir.is_absolute())
        self.assertEqual(args.dataset_dir, DEFAULT_DATASET_DIR)
        self.assertTrue(args.output.is_absolute())
        self.assertTrue(args.lora_output_dir.is_absolute())
        self.assertTrue(args.enable_lora)
        self.assertEqual(args.private_batch_size, 4)
        self.assertEqual(args.lora_threshold, 50)
        self.assertEqual(args.lora_train_batch_size, 1)
        self.assertEqual(args.lora_min_training_examples, 16)
        self.assertEqual(args.lora_gradient_accumulation_steps, 1)
        self.assertFalse(args.llm_experience_tags)

    def test_cli_can_disable_lora(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py", "--disable-lora"]):
            args = parse_args()

        self.assertFalse(args.enable_lora)

    def test_b_magent_main_resets_training_state_before_training(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_main_reset_test_"))
        try:
            output_file = temp_dir / "report.json"
            lora_output_dir = temp_dir / "lora_adapters"
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text(
                json.dumps({"question": "q", "answer": "#### 1"}) + "\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "four_agent_private_train.py",
                        "--mode",
                        "b-magent",
                        "--backend",
                        "demo",
                        "--answer-validator",
                        "local",
                        "--dataset-dir",
                        str(dataset_dir),
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
                }
                run_training.return_value.agents = []
                run_training.return_value.rounds = 0

                main()

            reset.assert_called_once()
            self.assertTrue(reset.call_args.kwargs["reset_evaluation_libraries"])
            self.assertEqual(reset.call_args.kwargs["lora_output_dir"], lora_output_dir)
            self.assertIn(output_file, reset.call_args.kwargs["report_files"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_main_preserves_training_state_with_resume_flag(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_main_resume_reset_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text(
                json.dumps({"question": "q", "answer": "#### 1"}) + "\n",
                encoding="utf-8",
            )
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
                        str(dataset_dir),
                        "--resume",
                    ],
                ),
                patch("train.four_agent_private_train.reset_b_magent_training_state") as reset,
                patch("train.four_agent_private_train.load_training_progress", return_value=12),
                patch("train.four_agent_private_train.run_b_magent_training_entry") as run_training,
                patch("train.four_agent_private_train.export_json_report"),
            ):
                run_training.return_value.agents = []
                run_training.return_value.rounds = 0

                main()

            reset.assert_not_called()
            self.assertEqual(run_training.call_args.kwargs["start_round"], 12)
            self.assertTrue(run_training.call_args.kwargs["preserve_private_datasets"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_main_does_not_reset_state_when_training_data_is_missing(self) -> None:
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
                    "/tmp/missing-infographicsvqa-data",
                ],
            ),
            patch("train.four_agent_private_train.reset_b_magent_training_state") as reset,
        ):
            with self.assertRaisesRegex(ValueError, "training state was not cleared"):
                main()

        reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
