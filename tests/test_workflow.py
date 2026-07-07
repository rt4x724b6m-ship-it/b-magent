from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.datasets import GSM8KDataset
from b_magent.models import Draft, PeerEvaluation
from b_magent.seed import seed_agent_libraries
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


class WorkflowTestCase(unittest.TestCase):
    def test_gold_answer_is_hidden_from_solver_and_evaluator_prompts(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.solve_tasks: list[str] = []
                self.review_tasks: list[str] = []

            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                self.solve_tasks.append(task)
                return "answer #### 1", []

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                self.review_tasks.append(task)
                return PeerEvaluation(evaluator_name, target_draft.agent_name, ["check"], "r", [])

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_gold_prompt_test_"))
        try:
            backend = RecordingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            workflow.run("Question: q\nGold reasoning: hidden\nGold final answer: 2")

            self.assertTrue(backend.solve_tasks)
            self.assertTrue(backend.review_tasks)
            self.assertTrue(all("Gold final answer" not in task for task in backend.solve_tasks))
            self.assertTrue(all("Gold reasoning" not in task for task in backend.review_tasks))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_self_improvement_scores_revised_answer_without_leaking_gold(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_correctness_test_"))
        try:
            agent = build_default_agents(temp_dir)[0]
            draft = Draft("qwen_agent_1", "方案规划", "wrong calculation #### 1", [], [], [], [])
            review = PeerEvaluation("qwen_agent_3", "qwen_agent_1", ["check arithmetic"], "r", [])

            improvement = agent.self_improve("Task\nGold final answer: 2", draft, [review])

            self.assertFalse(improvement.is_correct)
            self.assertNotIn("#### 2", improvement.revised_answer)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_self_improvement_normalizes_integer_decimal_correctness(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_decimal_correctness_test_"))
        try:
            agent = build_default_agents(temp_dir)[0]
            draft = Draft("qwen_agent_1", "方案规划", "calculation #### 2.0", [], [], [], [])

            improvement = agent.self_improve("Task\nGold final answer: 2", draft, [])

            self.assertTrue(improvement.is_correct)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_two_random_trainers_and_two_evaluators_evolve_separate_libraries(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_test_"))
        try:
            agents = build_default_agents(temp_dir)
            seed_agent_libraries(agents)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            task = "four-agent self-evolution with private training and evaluator suggestions"
            report = workflow.run(task)
            expected_agents = ["qwen_agent_1", "qwen_agent_2", "qwen_agent_3", "qwen_agent_4"]
            self.assertEqual(len(report.participants), 2)
            self.assertEqual(len(report.evaluators), 2)
            self.assertEqual(sorted(report.participants + report.evaluators), expected_agents)
            self.assertEqual(set(report.participants).isdisjoint(report.evaluators), True)
            self.assertEqual(len(report.drafts), 2)
            self.assertEqual(len(report.self_improvements), 2)
            self.assertEqual(len(report.evaluation_evolutions), 2)
            self.assertEqual(len(report.peer_reviews), 4)

            for draft in report.drafts:
                self.assertTrue(draft.private_training_used)
                self.assertTrue(draft.thought_trace)

            for review in report.peer_reviews:
                self.assertIn(review.evaluator, report.evaluators)
                self.assertIn(review.target, report.participants)
                self.assertTrue(review.suggestions)
                self.assertTrue(review.evaluation_memory_used)
                self.assertFalse(hasattr(review, "score"))

            for improvement in report.self_improvements:
                self.assertIn(improvement.agent_name, report.participants)
                self.assertTrue(improvement.professional_updates)

            for evolution in report.evaluation_evolutions:
                self.assertIn(evolution.agent_name, report.evaluators)
                self.assertTrue(evolution.evaluation_updates)
                self.assertEqual(len(evolution.synthesized_suggestions), 4)
                detail = evolution.evaluation_updates[0].detail
                self.assertIn("prior_evaluation_memory=", detail)
                self.assertIn("own_review_rationales=", detail)
                self.assertIn("own_review_scores=", detail)

            output_file = temp_dir / "data" / "report.json"
            workflow.export_report(report, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], task)
            self.assertNotIn("score", payload["peer_reviews"][0])

            for agent_name in expected_agents:
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
