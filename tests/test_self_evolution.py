from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.self_evolution import EvolutionInput, SelfEvolutionLibrary, evolve_all_agents


class SelfEvolutionLibraryTestCase(unittest.TestCase):
    def test_professional_and_evaluation_libraries_are_stored_separately(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_evolution_test_"))
        try:
            event = EvolutionInput(
                agent_name="qwen_agent_1",
                specialty="general-agent",
                task="solve GSM8K word problem",
                answer="Break the word problem into variables and equations.",
                thought_trace=["identify unknown", "translate sentence to equation"],
                peer_suggestions=["state boundary conditions", "verify arithmetic"],
                evaluator_suggestions=["check final numeric answer", "avoid vague feedback"],
                evaluator_rationales=["The reviewed answer skipped the final numeric check."],
                evaluation_memory_used=["Always verify the final numeric answer before suggesting style changes."],
                evaluation_scores=["target=qwen_agent_2: correctness=0.8, safety=1.0, efficiency=0.7"],
            )
            library = SelfEvolutionLibrary(temp_dir, event.agent_name)
            result = library.evolve_from_round(event)

            professional_file = temp_dir / "qwen_agent_1" / "professional_library.jsonl"
            evaluation_file = temp_dir / "qwen_agent_1" / "evaluation_library.jsonl"
            self.assertTrue(professional_file.exists())
            self.assertTrue(evaluation_file.exists())
            self.assertNotEqual(professional_file, evaluation_file)

            self.assertIsNotNone(result.professional_record)
            self.assertIsNotNone(result.evaluation_record)
            self.assertEqual(result.professional_record.library_type, "professional")
            self.assertEqual(result.evaluation_record.library_type, "evaluation")
            self.assertIn("solving lesson", result.professional_record.summary)
            self.assertIn("review lesson", result.evaluation_record.summary)
            self.assertIn("verify", result.professional_record.summary)
            self.assertIn("numeric answer", result.evaluation_record.summary)
            self.assertIn("prior_evaluation_memory=", result.evaluation_record.detail)
            self.assertIn("own_review_rationales=", result.evaluation_record.detail)
            self.assertIn("correctness=0.8", result.evaluation_record.detail)
            self.assertIn("future_solving_lesson=", result.professional_record.detail)
            self.assertIn("future_review_lesson=", result.evaluation_record.detail)

            professional_records = library.search_professional("boundary")
            evaluation_records = library.search_evaluation("numeric")
            self.assertEqual(len(professional_records), 1)
            self.assertEqual(len(evaluation_records), 1)
            self.assertEqual(professional_records[0].library_type, "professional")
            self.assertEqual(evaluation_records[0].library_type, "evaluation")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evolve_all_agents_writes_private_dual_libraries(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_evolution_all_test_"))
        try:
            events = [
                EvolutionInput(
                    agent_name="qwen_agent_1",
                    specialty="general-agent",
                    task="task",
                    answer="answer",
                    peer_suggestions=["improve answer structure"],
                    evaluator_suggestions=["evaluate answer clarity"],
                ),
                EvolutionInput(
                    agent_name="qwen_agent_4",
                    specialty="general-agent",
                    task="task",
                    answer="answer",
                    peer_suggestions=["improve checks"],
                    evaluator_suggestions=["evaluate calculation"],
                ),
            ]
            results = evolve_all_agents(temp_dir, events)
            self.assertEqual(len(results), 2)
            for event in events:
                self.assertTrue((temp_dir / event.agent_name / "professional_library.jsonl").exists())
                self.assertTrue((temp_dir / event.agent_name / "evaluation_library.jsonl").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_wrong_answers_still_create_error_lessons_for_experience_library(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_error_lesson_test_"))
        try:
            event = EvolutionInput(
                agent_name="qwen_agent_1",
                specialty="general-agent",
                task="solve numeric word problem",
                answer="wrong reasoning #### 1",
                thought_trace=["missed final check"],
                peer_suggestions=["verify final numeric answer before submitting"],
                is_correct=False,
            )
            library = SelfEvolutionLibrary(temp_dir, event.agent_name)
            record = library.evolve_professional(event)

            self.assertIn("error solving lesson", record.summary)
            self.assertIn("verify final numeric answer", record.summary)
            self.assertIn("future_solving_lesson=", record.detail)
            retrieved = library.search_professional("numeric verify")
            self.assertEqual(retrieved[0].summary, record.summary)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
