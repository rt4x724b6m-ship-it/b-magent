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
from b_magent.trajectory import mask_draft_for_evaluation
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


class WorkflowTestCase(unittest.TestCase):
    def test_masked_trajectory_keeps_answer_quality_signals(self) -> None:
        draft = Draft(
            "qwen_agent_1",
            "通用智能体",
            "1. Compute public result\n2. Check consistency\n#### 42",
            ["SECRET_RAW_TRACE"],
            ["SECRET_PRIVATE_SAMPLE"],
            ["memory one"],
            ["alert one"],
        )

        masked = mask_draft_for_evaluation(draft)
        serialized = json.dumps(masked.__dict__, ensure_ascii=False)

        self.assertIn("insight_solution_structure", serialized)
        self.assertIn("Numbered steps: 2", serialized)
        self.assertIn("Final marker present: True", serialized)
        self.assertIn("federated_answer_summary", masked.answer)
        self.assertIn("final_answer: 42", masked.answer)
        self.assertNotIn("Compute public result", masked.answer)
        self.assertNotIn("SECRET_RAW_TRACE", serialized)
        self.assertNotIn("SECRET_PRIVATE_SAMPLE", serialized)

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
            seed_agent_libraries(agents)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            workflow.run("Question: q\nGold reasoning: hidden\nGold final answer: 2")

            self.assertTrue(backend.solve_tasks)
            self.assertTrue(backend.review_tasks)
            self.assertTrue(all("Gold final answer" not in task for task in backend.solve_tasks))
            self.assertTrue(all("Gold reasoning" not in task for task in backend.review_tasks))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_peer_evaluation_receives_fot_trajectory_instead_of_private_data(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.review_drafts: list[Draft] = []

            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                return "answer #### 1", ["raw private-derived trace detail"]

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                self.review_drafts.append(target_draft)
                return PeerEvaluation(evaluator_name, target_draft.agent_name, ["check"], "r", [])

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_mask_eval_test_"))
        try:
            backend = RecordingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            private_file = temp_dir / "data" / "qwen_agent_1" / "private_data.jsonl"
            private_file.parent.mkdir(parents=True, exist_ok=True)
            private_file.write_text("SECRET_PRIVATE_SAMPLE\n", encoding="utf-8")
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            workflow.run("Question: q", participant_names=["qwen_agent_1", "qwen_agent_2"])

            self.assertTrue(backend.review_drafts)
            for review_draft in backend.review_drafts:
                serialized = json.dumps(review_draft.__dict__, ensure_ascii=False)
                self.assertEqual(review_draft.private_training_used, [])
                self.assertIn("federated_answer_summary", review_draft.answer)
                self.assertNotIn("answer #### 1", review_draft.answer)
                self.assertTrue(any(item.startswith("insight_") for item in review_draft.thought_trace))
                self.assertNotIn("SECRET_PRIVATE_SAMPLE", serialized)
                self.assertNotIn("raw private-derived trace detail", serialized)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evaluation_evolution_summarizes_target_results_without_raw_answer_text(self) -> None:
        class RecordingBackend:
            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                return (
                    "PRIVATE_DERIVED_REASONING_SENTENCE\n"
                    "1. public step\n"
                    "#### 17",
                    ["SECRET_RAW_THOUGHT_TRACE"],
                )

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                serialized = json.dumps(target_draft.__dict__, ensure_ascii=False)
                assert "SECRET_RAW_THOUGHT_TRACE" not in serialized
                assert "SECRET_PRIVATE_SAMPLE" not in serialized
                assert "PRIVATE_DERIVED_REASONING_SENTENCE" not in serialized
                assert "federated_answer_summary" in target_draft.answer
                return PeerEvaluation(evaluator_name, target_draft.agent_name, ["add verification"], "r", [])

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_eval_evolve_mask_test_"))
        try:
            backend = RecordingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            private_file = temp_dir / "data" / "qwen_agent_1" / "private_data.jsonl"
            private_file.parent.mkdir(parents=True, exist_ok=True)
            private_file.write_text("SECRET_PRIVATE_SAMPLE\n", encoding="utf-8")
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            report = workflow.run("Question: q", participant_names=["qwen_agent_1", "qwen_agent_2"])

            evaluation_details = "\n".join(
                record.detail
                for evolution in report.evaluation_evolutions
                for record in evolution.evaluation_updates
            )
            self.assertIn("revised_answer_summary=", evaluation_details)
            self.assertIn("final_answer=17", evaluation_details)
            self.assertNotIn("PRIVATE_DERIVED_REASONING_SENTENCE", evaluation_details)
            self.assertNotIn("SECRET_RAW_THOUGHT_TRACE", evaluation_details)
            self.assertNotIn("SECRET_PRIVATE_SAMPLE", evaluation_details)
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

    def test_self_improvement_regenerates_answer_without_gold_leakage(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.improve_tasks: list[str] = []
                self.improve_suggestions: list[list[str]] = []

            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                return "draft answer #### 1", []

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                return PeerEvaluation(evaluator_name, target_draft.agent_name, ["check arithmetic"], "r", [])

            def improve_answer(
                self,
                agent_name,
                specialty,
                task,
                draft,
                suggestions,
                professional_memory,
                evaluation_alerts,
            ):
                self.improve_tasks.append(task)
                self.improve_suggestions.append(suggestions)
                return "ideal rewritten answer #### 2", "regenerated from feedback"

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_regenerate_test_"))
        try:
            backend = RecordingBackend()
            agent = build_default_agents(temp_dir, backend=backend)[0]
            draft = Draft("qwen_agent_1", "方案规划", "wrong calculation #### 1", [], [], [], [])
            review = PeerEvaluation("qwen_agent_3", "qwen_agent_1", ["check arithmetic"], "r", [])

            improvement = agent.self_improve(
                "Question: q\nGold reasoning: hidden\nGold final answer: 2",
                draft,
                [review],
            )

            self.assertEqual(improvement.revised_answer, "ideal rewritten answer #### 2")
            self.assertEqual(improvement.reflection, "regenerated from feedback")
            self.assertTrue(improvement.is_correct)
            self.assertEqual(backend.improve_suggestions, [["check arithmetic"]])
            self.assertTrue(backend.improve_tasks)
            self.assertNotIn("Gold final answer", backend.improve_tasks[0])
            self.assertNotIn("Gold reasoning", backend.improve_tasks[0])
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
            self.assertIsNotNone(report.global_experience)
            assert report.global_experience is not None
            self.assertEqual(report.global_experience.server_name, "qwen_server_agent")
            self.assertEqual(sorted(report.global_experience.source_evaluators), sorted(report.evaluators))
            self.assertEqual(report.global_experience.source_update_count, 2)
            self.assertTrue(report.global_experience.global_updates)
            global_detail = report.global_experience.global_updates[0].detail
            self.assertIn("uploaded_consensus_evaluation_experience=", global_detail)

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
                self.assertIn("review_scores_peer_comparisons_and_target_results=", detail)
                self.assertIn("target=", detail)
                self.assertIn("revised_answer_summary=", detail)

            output_file = temp_dir / "data" / "report.json"
            workflow.export_report(report, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], task)
            self.assertNotIn("score", payload["peer_reviews"][0])
            self.assertIn("global_experience", payload)
            self.assertEqual(payload["global_experience"]["server_name"], "qwen_server_agent")

            for agent_name in expected_agents:
                professional_file = temp_dir / "data" / agent_name / "professional_library.jsonl"
                evaluation_file = temp_dir / "data" / agent_name / "evaluation_library.jsonl"
                self.assertTrue(professional_file.exists())
                self.assertTrue(evaluation_file.exists())
            self.assertTrue((temp_dir / "data" / "qwen_server_agent" / "global_evaluation_library.jsonl").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_agent_uses_same_backend_for_global_aggregation(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.global_calls: list[dict[str, object]] = []

            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                return "1. solve\n#### 3", ["public trace"]

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                return PeerEvaluation(
                    evaluator_name,
                    target_draft.agent_name,
                    ["verify final numeric answer"],
                    "r",
                    evaluation_memory,
                )

            def aggregate_global_experience(
                self,
                server_name,
                task,
                peer_reviews,
                evaluation_evolutions,
                consensus_evaluation_records,
                prior_global_memory,
            ):
                self.global_calls.append(
                    {
                        "server_name": server_name,
                        "peer_review_count": len(peer_reviews),
                        "evolution_count": len(evaluation_evolutions),
                        "consensus_count": len(consensus_evaluation_records),
                        "prior_global_memory": prior_global_memory,
                    }
                )
                return "global synthesized review lesson"

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_server_backend_test_"))
        try:
            backend = RecordingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            seed_agent_libraries(agents)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            report = workflow.run("solve numeric task", participant_names=["qwen_agent_1", "qwen_agent_2"])

            self.assertEqual(len(backend.global_calls), 1)
            self.assertEqual(backend.global_calls[0]["server_name"], "qwen_server_agent")
            self.assertEqual(backend.global_calls[0]["peer_review_count"], 4)
            self.assertEqual(backend.global_calls[0]["evolution_count"], 2)
            self.assertGreaterEqual(backend.global_calls[0]["consensus_count"], 1)
            self.assertIsNotNone(report.global_experience)
            assert report.global_experience is not None
            self.assertEqual(report.global_experience.synthesized_experience, "global synthesized review lesson")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_server_does_not_upload_when_evaluator_suggestions_disagree(self) -> None:
        class DisagreeingBackend:
            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                return "1. solve\n#### 3", ["public trace"]

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                if evaluator_name == "qwen_agent_3":
                    suggestions = ["verify final numeric answer"]
                else:
                    suggestions = ["rewrite as legal risk memo"]
                return PeerEvaluation(evaluator_name, target_draft.agent_name, suggestions, "r", evaluation_memory)

            def aggregate_global_experience(self, *args, **kwargs):
                raise AssertionError("server should not aggregate globally without evaluator consensus")

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_disagree_upload_test_"))
        try:
            backend = DisagreeingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            report = workflow.run("solve numeric task", participant_names=["qwen_agent_1", "qwen_agent_2"])

            self.assertIsNotNone(report.global_experience)
            assert report.global_experience is not None
            self.assertEqual(report.global_experience.source_update_count, 0)
            self.assertEqual(report.global_experience.global_updates, [])
            self.assertEqual(len(report.evaluation_evolutions), 2)
            for evolution in report.evaluation_evolutions:
                self.assertEqual(evolution.evaluation_updates, [])
            for agent_name in ("qwen_agent_3", "qwen_agent_4"):
                evaluation_file = temp_dir / "data" / agent_name / "evaluation_library.jsonl"
                self.assertFalse(evaluation_file.read_text(encoding="utf-8").strip())
            self.assertFalse((temp_dir / "data" / "qwen_server_agent" / "global_evaluation_library.jsonl").read_text(encoding="utf-8").strip())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evolved_library_summaries_are_reused_next_round(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.solve_memories: list[list[str]] = []
                self.review_memories: list[list[str]] = []

            def solve(self, agent_name, specialty, task, private_training, professional_memory, evaluation_alerts):
                self.solve_memories.append(professional_memory)
                return "1. solve\n#### 3", ["public trace"]

            def suggest_improvements(self, evaluator_name, target_draft, task, evaluation_memory):
                self.review_memories.append(evaluation_memory)
                return PeerEvaluation(
                    evaluator_name,
                    target_draft.agent_name,
                    ["verify final numeric answer"],
                    "r",
                    evaluation_memory,
                )

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_reuse_evolved_test_"))
        try:
            backend = RecordingBackend()
            agents = build_default_agents(temp_dir, backend=backend)
            workflow = MultiAgentWorkflow(agents, random_seed=7)

            workflow.run("solve numeric task", participant_names=["qwen_agent_1", "qwen_agent_2"])
            workflow.run("solve numeric task", participant_names=["qwen_agent_1", "qwen_agent_2"])

            flattened_solve_memories = "\n".join(item for batch in backend.solve_memories for item in batch)
            flattened_review_memories = "\n".join(item for batch in backend.review_memories for item in batch)
            self.assertIn("solving lesson", flattened_solve_memories)
            self.assertIn("review lesson", flattened_review_memories)
            self.assertIn("verify final numeric answer", flattened_review_memories)
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
