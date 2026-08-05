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
    AGENT_NAMES,
    STANDARD_PRIVATE_TRAIN_SIZE,
    build_participant_schedule,
    build_training_manifest,
    export_report,
    main,
    parse_args,
    run_b_magent_training_entry,
    run_four_agent_private_training,
    save_training_progress,
    validate_resume_manifest,
)
from b_magent.agent import QwenAgent


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


class SixAgentPrivateTrainingTestCase(unittest.TestCase):
    def test_private_training_excludes_the_current_task_answer(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_private_leak_test_"))
        try:
            agent_name = "qwen_agent_1"
            agent_dir = temp_dir / agent_name
            agent_dir.mkdir(parents=True)
            (agent_dir / "private_data.jsonl").write_text(
                "GSM8K sample | question: same question | reasoning_answer: secret #### 7 | final_answer: 7\n"
                "GSM8K sample | question: other question | reasoning_answer: safe #### 8 | final_answer: 8\n",
                encoding="utf-8",
            )
            agent = QwenAgent(agent_name, "math", temp_dir)

            batch = agent.train_private_data(
                "Solve this GSM8K training problem.\nQuestion: same question\nGold final answer: 7",
                batch_size=1,
            )

            self.assertEqual(len(batch), 1)
            self.assertIn("other question", batch[0])
            self.assertNotIn("secret", batch[0])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resume_manifest_rejects_changed_training_data(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_manifest_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            train_file = dataset_dir / "train.jsonl"
            train_file.write_text('{"question":"q1","answer":"#### 1"}\n', encoding="utf-8")
            with patch("sys.argv", ["four_agent_private_train.py", "--dataset-dir", str(dataset_dir)]):
                args = parse_args()
            progress_file = temp_dir / "training_progress.json"
            manifest = build_training_manifest(args)
            save_training_progress(progress_file, 1, 2, manifest)
            train_file.write_text('{"question":"q2","answer":"#### 2"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train_sha256"):
                validate_resume_manifest(progress_file, build_training_manifest(args))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resume_manifest_allows_lora_optimizer_stability_tuning(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_manifest_lora_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text('{"question":"q","answer":"#### 1"}\n', encoding="utf-8")
            with patch("sys.argv", ["four_agent_private_train.py", "--dataset-dir", str(dataset_dir)]):
                args = parse_args()
            progress_file = temp_dir / "training_progress.json"
            manifest = build_training_manifest(args)
            manifest["lora_gradient_accumulation_steps"] = 4
            manifest["lora_learning_rate"] = 1e-4
            save_training_progress(progress_file, 1, 2, manifest)

            validate_resume_manifest(progress_file, build_training_manifest(args))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_backend_failure_does_not_clear_existing_training_state(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_preload_safety_test_"))
        try:
            dataset_dir = temp_dir / "gsm8k"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text(
                '{"question":"q1","answer":"#### 1"}\n',
                encoding="utf-8",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "four_agent_private_train.py",
                        "--mode",
                        "b-magent",
                        "--dataset-dir",
                        str(dataset_dir),
                    ],
                ),
                patch(
                    "train.four_agent_private_train.build_b_magent_backend",
                    side_effect=RuntimeError("model load failed"),
                ),
                patch("train.four_agent_private_train.reset_b_magent_training_state") as reset,
            ):
                with self.assertRaisesRegex(RuntimeError, "model load failed"):
                    main()

            reset.assert_not_called()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_default_split_reserves_thirty_percent_and_private_sets_do_not_overlap(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_thirty_percent_split_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            rows = [
                {"question": f"unique-question-{i}", "answer": f"reasoning #### {i}"}
                for i in range(20)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=1,
                random_seed=13,
            )

            private_rows = []
            for agent_name in AGENT_NAMES:
                private_rows.extend(
                    line
                    for line in (temp_dir / "data" / agent_name / "private_data.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
            self.assertEqual(report.reserved_total, 6)
            self.assertEqual(report.train_total, 14)
            self.assertEqual(sum(report.private_dataset_counts.values()), 14)
            self.assertEqual(len(private_rows), len(set(private_rows)))
            reserved_rows = Path(report.reserved_data_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(reserved_rows), 6)
            self.assertTrue(set(reserved_rows).isdisjoint(private_rows))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_b_magent_keeps_reserved_samples_out_of_private_training(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_reserved_split_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            rows = [
                {"question": f"q{i}", "answer": f"reasoning #### {i}"}
                for i in range(12)
            ]
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            report = run_b_magent_training_entry(
                dataset_dir=dataset_dir,
                data_dir=temp_dir / "data",
                rounds=1,
                random_seed=7,
                reserved_size=4,
            )

            private_text = "\n".join(
                (temp_dir / "data" / name / "private_data.jsonl").read_text(encoding="utf-8")
                for name in AGENT_NAMES
            )
            self.assertEqual(report.train_total, 8)
            self.assertEqual(report.reserved_total, 4)
            reserved_text = Path(report.reserved_data_path).read_text(encoding="utf-8")
            self.assertTrue(reserved_text)
            self.assertTrue(all(f"question: q{i} |" not in private_text for i in range(12) if f"question: q{i} |" in reserved_text))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_six_agents_train_separately_for_three_rounds(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_training_test_"))
        try:
            dataset_dir = temp_dir / "data" / "gsm8k"
            dataset_dir.mkdir(parents=True)
            train_rows = [
                {"question": f"q{i}", "answer": f"a{i} #### {i}"}
                for i in range(12)
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
            self.assertEqual(report.train_total, 12)
            self.assertEqual(report.test_total, 4)
            self.assertEqual(len(report.agents), 6)
            self.assertEqual(
                [agent.agent_name for agent in report.agents],
                list(AGENT_NAMES),
            )
            self.assertEqual([agent.private_train_samples for agent in report.agents], [2] * 6)

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
                for i in range(6)
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
                for i in range(STANDARD_PRIVATE_TRAIN_SIZE * 6)
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
                [STANDARD_PRIVATE_TRAIN_SIZE] * 6,
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
                for i in range(STANDARD_PRIVATE_TRAIN_SIZE * 6 - 1)
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

            with self.assertRaisesRegex(ValueError, "need at least 1200 training samples"):
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
                reserved_size=0,
            )

            self.assertEqual(
                report.private_dataset_counts,
                {
                    "qwen_agent_1": 2,
                    "qwen_agent_2": 2,
                    "qwen_agent_3": 2,
                    "qwen_agent_4": 2,
                    "qwen_agent_5": 1,
                    "qwen_agent_6": 1,
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
                reserved_size=0,
            )

            self.assertEqual(report.rounds, 4)
            trained_slots = {}
            for round_report in report.training_rounds:
                for agent_name in round_report.participants:
                    trained_slots[agent_name] = trained_slots.get(agent_name, 0) + 1
            self.assertTrue(
                all(trained_slots.get(name, 0) >= count for name, count in report.private_dataset_counts.items())
            )
            self.assertEqual(sum(trained_slots.values()) % 3, 0)
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
            self.assertEqual(report.training_rounds[1].global_downlinks, 6)
            self.assertEqual(report.training_rounds[1].global_uploads, 1)
            self.assertTrue((temp_dir / "data" / "qwen_server_agent" / "global_evaluation_library.jsonl").exists())
            for agent_name in AGENT_NAMES:
                evaluation_text = (temp_dir / "data" / agent_name / "evaluation_library.jsonl").read_text(
                    encoding="utf-8"
                )
                self.assertIn("global-downlink", evaluation_text)
                self.assertIn("source_global_experience_id=qwen_server_agent:", evaluation_text)
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

            self.assertEqual(backend.release_calls, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_participant_schedule_uses_distinct_agents_per_round(self) -> None:
        schedule = build_participant_schedule(
            {
                "qwen_agent_1": 3,
                "qwen_agent_2": 3,
                "qwen_agent_3": 2,
                "qwen_agent_4": 2,
                "qwen_agent_5": 1,
                "qwen_agent_6": 1,
            },
            private_batch_size=1,
        )

        self.assertEqual(len(schedule), 4)
        self.assertTrue(all(len(set(group)) == 3 for group in schedule))

    def test_cli_defaults_enable_lora_and_auto_cover_private_data(self) -> None:
        with patch("sys.argv", ["four_agent_private_train.py"]):
            args = parse_args()

        self.assertEqual(args.rounds, 0)
        self.assertTrue(args.dataset_dir.is_absolute())
        self.assertTrue(args.output.is_absolute())
        self.assertTrue(args.lora_output_dir.is_absolute())
        self.assertTrue(args.enable_lora)
        self.assertEqual(args.lora_threshold, 10)
        self.assertEqual(args.lora_train_batch_size, 4)
        self.assertEqual(args.lora_gradient_accumulation_steps, 1)
        self.assertEqual(args.lora_epochs, 2.0)
        self.assertEqual(args.lora_learning_rate, 2e-5)
        self.assertEqual(args.lora_min_evaluation_score, 0.85)
        self.assertIsNone(args.reserved_size)
        self.assertEqual(args.reserved_ratio, 0.3)

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
                patch("train.four_agent_private_train.validate_resume_manifest") as validate_manifest,
                patch("train.four_agent_private_train.run_b_magent_training_entry") as run_training,
                patch("train.four_agent_private_train.export_json_report"),
            ):
                run_training.return_value.agents = []
                run_training.return_value.rounds = 0

                main()

            reset.assert_not_called()
            validate_manifest.assert_called_once()
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
    build_training_manifest,
