from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from train.four_agent_private_train import export_report, run_four_agent_private_training


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
            )

            self.assertEqual(report.rounds, 3)
            self.assertEqual(report.batches_per_round, 32)
            self.assertEqual(report.batch_size, 1)
            self.assertEqual(report.train_total, 8)
            self.assertEqual(report.test_total, 4)
            self.assertEqual(len(report.agents), 4)
            self.assertEqual(
                [agent.agent_name for agent in report.agents],
                ["qwen_planner", "qwen_executor", "qwen_reviewer", "qwen_verifier"],
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


if __name__ == "__main__":
    unittest.main()
