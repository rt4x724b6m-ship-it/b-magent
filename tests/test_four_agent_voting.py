from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from train.three_agent_private_train import AGENT_NAMES, run_four_agent_voting_on_test


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


class FourAgentVotingTestCase(unittest.TestCase):
    def test_four_agents_vote_final_answer_on_test_dataset(self) -> None:
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
                "qwen_planner": FixedVoteModel(["42", "7", "9"]),
                "qwen_executor": FixedVoteModel(["42", "8", "11"]),
                "qwen_reviewer": FixedVoteModel(["41", "7", "12"]),
                "qwen_verifier": FixedVoteModel(["0", "8", "13"]),
            }
            report = run_four_agent_voting_on_test(dataset_dir, models=models)

            self.assertEqual(report.total, 3)
            self.assertEqual(report.correct, 2)
            self.assertEqual(report.accuracy, 2 / 3)
            self.assertEqual([vote.agent_name for vote in report.predictions[0].votes], list(AGENT_NAMES))
            self.assertEqual(report.predictions[0].final_answer, "42")
            self.assertTrue(report.predictions[0].correct)

            # The second row is a 2-2 tie: planner/reviewer vote 7, executor/verifier vote 8.
            # Ties are resolved by the first answer in AGENT_NAMES order.
            self.assertEqual(report.predictions[1].final_answer, "7")
            self.assertTrue(report.predictions[1].correct)

            self.assertEqual(report.predictions[2].final_answer, "9")
            self.assertFalse(report.predictions[2].correct)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
