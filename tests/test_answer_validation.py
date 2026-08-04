from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import httpx

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.agent import QwenAgent
from b_magent.answer_validation import (
    AnswerValidationResult,
    GPT56SolRequirementValidator,
)
from b_magent.backend import DemoQwenBackend
from b_magent.models import Draft
from b_magent.datasets import GSM8KDataset
from train.four_agent_private_train import AGENT_NAMES, run_four_agent_voting_on_test


class GPT56SolValidationTestCase(unittest.TestCase):
    @staticmethod
    def _valid_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "correct": True,
                        "requirements_met": ["answer supplied"],
                        "requirements_missed": [],
                        "unsupported_claims": [],
                        "rationale": "The answer is supported.",
                    }
                )
            },
            request=request,
        )

    def test_travelplanner_test_rows_load_without_annotated_plan(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_travel_test_loader_test_"))
        try:
            (temp_dir / "test.csv").write_text(
                "query,reference_information\n"
                '"Plan a trip","[{\'Description\': \'Hotels\', \'Content\': \'Hotel A costs 100\'}]"\n',
                encoding="utf-8",
            )

            samples = GSM8KDataset(temp_dir).load("test")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].answer, "")
            self.assertIn("Hotel A costs 100", samples[0].reference_information)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_semantic_validator_marks_compliant_test_answer_correct_despite_gold_mismatch(self) -> None:
        class FixedModel:
            def train_batch(self, batch: object) -> None:
                return None

            def generate(self, question: str) -> str:
                return "A semantically compliant alternative. #### 7"

        class CompliantValidator:
            def __init__(self) -> None:
                self.answers: list[str] = []

            def validate(self, task: str, answer: str) -> AnswerValidationResult:
                self.answers.append(answer)
                return AnswerValidationResult(
                    True,
                    ["user goal", "hard constraints"],
                    [],
                    [],
                    "The alternative satisfies the task requirements.",
                )

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_semantic_test_correctness_test_"))
        try:
            (temp_dir / "test.jsonl").write_text(
                json.dumps({"question": "Return any valid option.", "answer": "#### 99"}) + "\n",
                encoding="utf-8",
            )
            validator = CompliantValidator()

            report = run_four_agent_voting_on_test(
                temp_dir,
                models={name: FixedModel() for name in AGENT_NAMES},
                answer_validator=validator,
            )

            prediction = report.predictions[0]
            self.assertEqual(prediction.final_answer, "7")
            self.assertNotEqual(prediction.final_answer, prediction.gold_answer)
            self.assertTrue(prediction.correct)
            self.assertIn("semantically compliant alternative", validator.answers[0])
            self.assertEqual(prediction.requirements_met, ["user goal", "hard constraints"])
            self.assertEqual(
                prediction.answer_validation_rationale,
                "The alternative satisfies the task requirements.",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_validator_uses_sol_and_accepts_semantically_compliant_answer(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "correct": True,
                                            "requirements_met": ["three-day trip", "within budget"],
                                            "requirements_missed": [],
                                            "unsupported_claims": [],
                                            "rationale": "The answer meets every requested constraint.",
                                        }
                                    ),
                                }
                            ],
                        }
                    ]
                },
                request=request,
            )

        validator = GPT56SolRequirementValidator(
            "test-key",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        task = (
            "Task: Plan a three-day trip under $1000.\n"
            "Candidate reference information: Hotel A costs $100.\n"
            "Gold retrieval targets: hidden-source\n"
            "Gold reference response: hidden official wording"
        )

        result = validator.validate(task, "Use Hotel A and keep the total below $1000.")

        self.assertTrue(result.correct)
        self.assertEqual(requests[0]["model"], "gpt-5.6-sol")
        self.assertEqual(requests[0]["reasoning"], {"effort": "medium"})
        request_text = json.dumps(requests[0], ensure_ascii=False)
        self.assertNotIn("hidden official wording", request_text)
        self.assertNotIn("hidden-source", request_text)
        self.assertIn("Do not require wording", request_text)

    def test_validator_retries_an_empty_success_response(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(200, content=b"", request=request)
            return self._valid_response(request)

        validator = GPT56SolRequirementValidator(
            "test-key",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=1,
            retry_backoff=0,
        )

        result = validator.validate("Return an answer.", "answer")

        self.assertTrue(result.correct)
        self.assertEqual(attempts, 2)

    def test_validator_reports_gateway_response_after_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                502,
                text="upstream temporarily unavailable",
                headers={"content-type": "text/plain"},
                request=request,
            )

        validator = GPT56SolRequirementValidator(
            "test-key",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=1,
            retry_backoff=0,
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 502.*upstream temporarily unavailable"):
            validator.validate("Return an answer.", "answer")

    def test_gpt_verdict_controls_professional_experience_kind(self) -> None:
        class AlwaysCorrectValidator:
            def validate(self, task: str, answer: str) -> AnswerValidationResult:
                return AnswerValidationResult(True, ["all requirements"], [], [], "compliant")

        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_gpt_verdict_experience_test_"))
        try:
            agent = QwenAgent(
                "qwen_agent_1",
                "general-agent",
                temp_dir,
                backend=DemoQwenBackend(),
                answer_validator=AlwaysCorrectValidator(),
            )
            draft = Draft(
                agent_name=agent.name,
                specialty=agent.specialty,
                answer="A different but compliant plan.",
                thought_trace=[],
                private_training_used=[],
                professional_memory_used=[],
                evaluation_alerts_used=[],
            )

            improvement = agent.self_improve(
                "Task: Produce any plan satisfying the stated requirements.",
                draft,
                [],
            )

            self.assertTrue(improvement.is_correct)
            self.assertIn("curated-success-experience", improvement.professional_updates[0].tags)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
