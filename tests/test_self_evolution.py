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
                agent_name="qwen_planner",
                specialty="planning",
                task="solve GSM8K word problem",
                answer="Break the word problem into variables and equations.",
                thought_trace=["identify unknown", "translate sentence to equation"],
                peer_suggestions=["state boundary conditions", "verify arithmetic"],
                evaluator_suggestions=["check final numeric answer", "avoid vague feedback"],
            )
            library = SelfEvolutionLibrary(temp_dir, event.agent_name)
            result = library.evolve_from_round(event)

            professional_file = temp_dir / "qwen_planner" / "professional_library.jsonl"
            evaluation_file = temp_dir / "qwen_planner" / "evaluation_library.jsonl"
            self.assertTrue(professional_file.exists())
            self.assertTrue(evaluation_file.exists())
            self.assertNotEqual(professional_file, evaluation_file)

            self.assertIsNotNone(result.professional_record)
            self.assertIsNotNone(result.evaluation_record)
            self.assertEqual(result.professional_record.library_type, "professional")
            self.assertEqual(result.evaluation_record.library_type, "evaluation")

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
                    agent_name="qwen_planner",
                    specialty="planning",
                    task="task",
                    answer="answer",
                    peer_suggestions=["improve plan"],
                    evaluator_suggestions=["review plan clarity"],
                ),
                EvolutionInput(
                    agent_name="qwen_verifier",
                    specialty="verification",
                    task="task",
                    answer="answer",
                    peer_suggestions=["improve checks"],
                    evaluator_suggestions=["review calculation"],
                ),
            ]
            results = evolve_all_agents(temp_dir, events)
            self.assertEqual(len(results), 2)
            for event in events:
                self.assertTrue((temp_dir / event.agent_name / "professional_library.jsonl").exists())
                self.assertTrue((temp_dir / event.agent_name / "evaluation_library.jsonl").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
