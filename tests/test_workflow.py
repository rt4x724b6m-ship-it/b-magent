from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.datasets import GSM8KDataset
from b_magent.seed import seed_agent_libraries
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


class WorkflowTestCase(unittest.TestCase):
    def test_self_evolution_randomly_splits_roles_and_writes_libraries(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_test_"))
        try:
            agents = build_default_agents(temp_dir)
            seed_agent_libraries(agents)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            task = "three-agent self-evolution with private training and evaluator suggestions"
            report = workflow.run(task)
            self.assertGreater(len(report.participants), 0)
            self.assertGreater(len(report.evaluators), 0)
            self.assertTrue(set(report.participants).isdisjoint(report.evaluators))
            self.assertEqual(len(report.drafts), len(report.participants))
            self.assertEqual(len(report.self_improvements), len(report.participants))
            self.assertEqual(len(report.evaluation_evolutions), len(report.evaluators))
            self.assertEqual(len(report.peer_reviews), len(report.participants) * len(report.evaluators))

            for draft in report.drafts:
                self.assertTrue(draft.private_training_used)
                self.assertTrue(draft.thought_trace)

            for review in report.peer_reviews:
                self.assertTrue(review.suggestions)
                self.assertFalse(hasattr(review, "score"))

            for improvement in report.self_improvements:
                self.assertTrue(improvement.professional_updates)

            for evolution in report.evaluation_evolutions:
                self.assertTrue(evolution.evaluation_updates)

            output_file = temp_dir / "data" / "report.json"
            workflow.export_report(report, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], task)
            self.assertNotIn("score", payload["peer_reviews"][0])

            for agent_name in ("qwen_planner", "qwen_executor", "qwen_reviewer"):
                professional_file = temp_dir / "data" / agent_name / "professional_library.jsonl"
                evaluation_file = temp_dir / "data" / agent_name / "evaluation_library.jsonl"
                self.assertTrue(professional_file.exists())
                self.assertTrue(evaluation_file.exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_gsm8k_train_jsonl_feeds_private_training(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_gsm8k_test_"))
        try:
            gsm8k_dir = temp_dir / "data" / "gsm8k"
            gsm8k_dir.mkdir(parents=True)
            sample = {
                "question": "Natalia sold clips to 48 friends in April, then half as many in May. How many total?",
                "answer": "May sales are 48 / 2 = 24. Total is 48 + 24 = 72. #### 72",
            }
            (gsm8k_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

            dataset = GSM8KDataset(gsm8k_dir)
            loaded = dataset.load()
            self.assertEqual(loaded[0].final_answer, "72")

            agents = build_default_agents(temp_dir)
            seed_agent_libraries(agents)
            workflow = MultiAgentWorkflow(agents, random_seed=7)
            report = workflow.run("solve GSM8K-style math problems")

            used_private_training = [
                item
                for draft in report.drafts
                for item in draft.private_training_used
            ]
            self.assertTrue(any("GSM8K sample" in item for item in used_private_training))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_gsm8k_raw_jsonl_can_be_split_into_train_and_test(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_gsm8k_split_test_"))
        try:
            gsm8k_dir = temp_dir / "data" / "gsm8k"
            gsm8k_dir.mkdir(parents=True)
            source_file = gsm8k_dir / "raw.jsonl"
            rows = [
                {"question": "q1", "answer": "a1 #### 1"},
                {"question": "q2", "answer": "a2 #### 2"},
                {"question": "q3", "answer": "a3 #### 3"},
                {"question": "q4", "answer": "a4 #### 4"},
                {"question": "q5", "answer": "a5 #### 5"},
            ]
            source_file.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = GSM8KDataset(gsm8k_dir)
            counts = dataset.split_raw_jsonl(source_file, test_ratio=0.4, seed=3)
            self.assertEqual(counts, {"train": 3, "test": 2, "skipped": 0})
            self.assertTrue((gsm8k_dir / "train.jsonl").exists())
            self.assertTrue((gsm8k_dir / "test.jsonl").exists())
            self.assertEqual(len(dataset.load("train")), 3)
            self.assertEqual(len(dataset.load("test")), 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
