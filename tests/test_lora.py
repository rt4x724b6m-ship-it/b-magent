from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.lora import (
    LoraEvolutionManager,
    LoraTrainingConfig,
    LoraUpdate,
    build_lora_example,
    tokenize_lora_row,
)
from b_magent.models import Draft, EvaluationScores, PeerEvaluation, SelfImprovement


class FakeLoraTrainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, Path]] = []

    def train(self, agent_name: str, dataset_path: Path, adapter_path: Path, config: LoraTrainingConfig) -> None:
        self.calls.append((agent_name, dataset_path, adapter_path))
        adapter_path.mkdir(parents=True, exist_ok=True)


def count_jsonl_rows_for_test(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class LoraEvolutionTestCase(unittest.TestCase):
    def test_sft_tokenization_masks_prompt_and_preserves_output_when_truncated(self) -> None:
        class CharacterTokenizer:
            eos_token_id = 0

            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                return {"input_ids": [ord(character) for character in text]}

        tokenized = tokenize_lora_row(
            CharacterTokenizer(),
            {"instruction": "solve", "input": "x" * 100, "output": "answer"},
            max_length=24,
        )

        self.assertLessEqual(len(tokenized["input_ids"]), 24)
        first_target = tokenized["labels"].index(ord("a"))
        self.assertTrue(all(label == -100 for label in tokenized["labels"][:first_target]))
        self.assertEqual(tokenized["labels"][first_target:], [ord(char) for char in "answer"] + [0])

    def test_builds_reflection_sft_example_from_trajectory_and_evaluations(self) -> None:
        draft = Draft(
            agent_name="qwen_agent_1",
            specialty="通用智能体",
            answer="old answer",
            thought_trace=["reasoned step"],
            private_training_used=["private"],
            professional_memory_used=[],
            evaluation_alerts_used=[],
            tool_calls=["calculator(1+1)"],
        )
        evaluation = PeerEvaluation(
            evaluator="qwen_agent_3",
            target="qwen_agent_1",
            suggestions=["fix final answer"],
            rationale="correctness feedback",
            evaluation_memory_used=[],
            scores=EvaluationScores(correctness=0.9, safety=1.0, efficiency=0.8),
        )
        improvement = SelfImprovement(
            agent_name="qwen_agent_1",
            applied_suggestions=["fix final answer"],
            revised_answer="improved answer",
            professional_updates=[],
        )

        example = build_lora_example("solve task", draft, [evaluation], improvement)

        self.assertEqual(example.agent_name, "qwen_agent_1")
        self.assertIn("solve task", example.input)
        self.assertIn("old answer", example.input)
        self.assertIn("calculator(1+1)", example.input)
        self.assertIn("fix final answer", example.input)
        self.assertIn("correctness=0.90", example.input)
        self.assertEqual(example.output, "improved answer")

    def test_lora_example_hides_gold_answer_from_training_input(self) -> None:
        draft = Draft(
            agent_name="qwen_agent_1",
            specialty="通用智能体",
            answer="draft #### 1",
            thought_trace=[],
            private_training_used=[],
            professional_memory_used=[],
            evaluation_alerts_used=[],
        )
        improvement = SelfImprovement("qwen_agent_1", ["fix"], "corrected #### 2", [])

        example = build_lora_example(
            "Question: q\nGold reasoning: first hidden line\nsecond hidden line\n#### 2\nGold final answer: 2",
            draft,
            [PeerEvaluation("qwen_agent_3", "qwen_agent_1", ["fix"], "r", [])],
            improvement,
        )

        self.assertNotIn("Gold reasoning", example.input)
        self.assertNotIn("Gold final answer", example.input)
        self.assertNotIn("second hidden line", example.input)
        self.assertNotIn("#### 2", example.input)
        self.assertIn("Question: q", example.input)

    def test_manager_accumulates_per_agent_datasets_and_trains_when_curated_examples_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-1.5B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                ),
                trainer=trainer,
            )
            drafts = [
                Draft("qwen_agent_1", "通用智能体", "a1", ["t1"], [], [], []),
                Draft("qwen_agent_2", "通用智能体", "a2", ["t2"], [], [], []),
            ]
            reviews = [
                PeerEvaluation("qwen_agent_3", "qwen_agent_1", ["s1"], "r1", []),
                PeerEvaluation("qwen_agent_4", "qwen_agent_1", ["s2"], "r2", []),
                PeerEvaluation("qwen_agent_3", "qwen_agent_2", ["s3"], "r3", []),
                PeerEvaluation("qwen_agent_4", "qwen_agent_2", ["s4"], "r4", []),
            ]
            improvements = [
                SelfImprovement("qwen_agent_1", ["s1", "s2"], "better a1", [], is_correct=None),
                SelfImprovement("qwen_agent_2", ["s3", "s4"], "better a2", [], is_correct=None),
            ]

            updates = manager.update_from_round("task", drafts, reviews, improvements)

            self.assertEqual([update.trained for update in updates], [True, True])
            self.assertEqual([call[0] for call in trainer.calls], ["qwen_agent_1", "qwen_agent_2"])
            for update in updates:
                self.assertIsInstance(update, LoraUpdate)
                dataset_rows = Path(update.dataset_path).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(dataset_rows), 1)
                self.assertEqual(json.loads(dataset_rows[0])["agent_name"], update.agent_name)
                self.assertTrue((Path(update.adapter_path) / "b_magent_lora_metadata.json").exists())
                state = json.loads((Path(update.dataset_path).parent / "lora_state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["version"], 1)
                self.assertEqual(state["pending_examples"], 0)

    def test_rejects_incorrect_or_duplicate_examples_before_lora_training(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_gate_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-1.5B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                ),
                trainer=trainer,
            )
            draft = Draft("qwen_agent_1", "通用智能体", "draft #### 1", ["t1"], [], [], [])
            reviews = [
                PeerEvaluation(
                    "qwen_agent_3",
                    "qwen_agent_1",
                    ["fix"],
                    "r",
                    [],
                    EvaluationScores(correctness=1.0, safety=1.0, efficiency=1.0),
                )
            ]

            bad_updates = manager.update_from_round(
                "Task\nGold final answer: 2",
                [draft],
                reviews,
                [SelfImprovement("qwen_agent_1", ["fix"], "still wrong #### 1", [])],
            )
            self.assertEqual(bad_updates[0].reason, "improved answer failed gold-answer correctness gate")
            self.assertFalse(manager.dataset_path("qwen_agent_1").exists())

            good = SelfImprovement("qwen_agent_1", ["fix"], "now correct #### 2", [])
            first_updates = manager.update_from_round("Task\nGold final answer: 2", [draft], reviews, [good])
            duplicate_updates = manager.update_from_round("Task\nGold final answer: 2", [draft], reviews, [good])

            self.assertTrue(first_updates[0].trained)
            self.assertEqual(duplicate_updates[0].reason, "duplicate SFT example")
            self.assertEqual(len(trainer.calls), 1)

    def test_lora_batches_updates_at_threshold_and_flushes_final_examples(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_curated_dataset_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-1.5B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=2,
                ),
                trainer=trainer,
            )
            reviews = [
                PeerEvaluation(
                    "qwen_agent_3",
                    "qwen_agent_1",
                    ["verify final answer"],
                    "r",
                    [],
                    EvaluationScores(correctness=1.0, safety=1.0, efficiency=1.0),
                )
            ]
            first = manager.update_from_round(
                "Task A\nGold final answer: 2",
                [Draft("qwen_agent_1", "通用智能体", "draft #### 1", ["t1"], [], [], [])],
                reviews,
                [SelfImprovement("qwen_agent_1", ["fix"], "correct #### 2", [])],
            )
            self.assertFalse(first[0].trained)
            self.assertEqual(first[0].pending_examples, 1)
            self.assertIn("1/2", first[0].reason)
            self.assertEqual(len(trainer.calls), 0)
            self.assertEqual(count_jsonl_rows_for_test(manager.dataset_path("qwen_agent_1")), 1)

            second = manager.update_from_round(
                "Task B\nGold final answer: 3",
                [Draft("qwen_agent_1", "通用智能体", "draft #### 1", ["t2"], [], [], [])],
                reviews,
                [SelfImprovement("qwen_agent_1", ["fix"], "correct #### 3", [])],
            )

            self.assertTrue(second[0].trained)
            self.assertEqual(second[0].examples, 2)
            self.assertEqual(second[0].pending_examples, 0)
            self.assertEqual(len(trainer.calls), 1)
            trained_rows = Path(second[0].dataset_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(trained_rows), 2)

            third = manager.update_from_round(
                "Task C\nGold final answer: 4",
                [Draft("qwen_agent_1", "通用智能体", "draft #### 1", ["t3"], [], [], [])],
                reviews,
                [SelfImprovement("qwen_agent_1", ["fix"], "correct #### 4", [])],
            )
            self.assertFalse(third[0].trained)

            flushed = manager.flush_pending(["qwen_agent_1"])
            self.assertTrue(flushed[0].trained)
            self.assertEqual(flushed[0].examples, 3)
            self.assertEqual(len(trainer.calls), 2)


if __name__ == "__main__":
    unittest.main()
