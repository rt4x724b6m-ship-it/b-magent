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
    configure_multimodal_image_budget,
    configure_processor_tokenizer,
    safe_lora_epochs,
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
    def test_releases_inference_model_before_lora_trainer_starts(self) -> None:
        events: list[str] = []

        class RecordingTrainer:
            def train(self, agent_name, dataset_path, adapter_path, config):  # type: ignore[no-untyped-def]
                events.append("train")

        with tempfile.TemporaryDirectory() as temp:
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
                    output_dir=Path(temp) / "lora",
                ),
                trainer=RecordingTrainer(),
                before_train=lambda: events.append("release"),
            )
            dataset_path = manager.dataset_path("qwen_agent_1")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps({"instruction": "solve", "input": "1+1", "output": "2"}) + "\n",
                encoding="utf-8",
            )

            manager.train_agent_on_curated_dataset("qwen_agent_1")

        self.assertEqual(events, ["release", "train"])

    def test_multimodal_image_budget_keeps_visual_tokens_below_sequence_limit(self) -> None:
        class ImageProcessor:
            patch_size = 14
            merge_size = 2
            min_pixels = 56 * 56
            max_pixels = 28 * 28 * 12800

        class VLProcessor:
            image_processor = ImageProcessor()

        processor = VLProcessor()
        max_pixels = configure_multimodal_image_budget(processor, 2048)

        self.assertEqual(max_pixels, 1536 * 14 * 14 * 4)
        self.assertEqual(processor.image_processor.max_pixels, max_pixels)
        self.assertLessEqual(max_pixels // (14 * 14 * 4), 2048 - 256)

    def test_multimodal_image_budget_does_not_enlarge_existing_cap(self) -> None:
        class ImageProcessor:
            patch_size = 14
            merge_size = 2
            min_pixels = 56 * 56
            max_pixels = 100_000

        class VLProcessor:
            image_processor = ImageProcessor()

        processor = VLProcessor()
        configure_multimodal_image_budget(processor, 2048)

        self.assertEqual(processor.image_processor.max_pixels, 100_000)

    def test_tiny_lora_datasets_have_a_safe_epoch_cap(self) -> None:
        self.assertEqual(safe_lora_epochs(1, 200.0), 3.0)
        self.assertEqual(safe_lora_epochs(16, 200.0), 5.0)
        self.assertEqual(safe_lora_epochs(50, 200.0), 10.0)

    def test_lora_does_not_train_below_safety_minimum(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_minimum_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="model",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                    min_training_examples=2,
                    require_correct_answer=False,
                ),
                trainer=trainer,
            )
            updates = manager.update_from_round(
                "task",
                [Draft("qwen_agent_1", "specialty", "draft", [], [], [], [])],
                [PeerEvaluation("qwen_agent_2", "qwen_agent_1", [], "ok", [])],
                [SelfImprovement("qwen_agent_1", [], "answer", [])],
            )

            self.assertFalse(updates[0].trained)
            self.assertIn("safety minimum", updates[0].reason)
            self.assertEqual(trainer.calls, [])

    def test_lora_lifecycle_is_stored_as_non_prompt_professional_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_audit_test_") as temp:
            root = Path(temp)
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="model",
                    output_dir=root / "lora",
                    threshold=50,
                    professional_library_dir=root / "data",
                    require_correct_answer=False,
                ),
                trainer=FakeLoraTrainer(),
            )
            manager.update_from_round(
                "audit task",
                [Draft("qwen_agent_1", "specialty", "draft", [], [], [], [])],
                [PeerEvaluation("qwen_agent_2", "qwen_agent_1", [], "ok", [])],
                [SelfImprovement("qwen_agent_1", [], "answer", [])],
            )

            payload = json.loads(
                (root / "data/qwen_agent_1/professional_library.jsonl").read_text(encoding="utf-8")
            )
            self.assertIn("lora-training-metadata", payload["tags"])
            self.assertIn("example_hash=", payload["detail"])

    def test_configures_tokenizer_wrapped_by_vl_processor(self) -> None:
        class TextTokenizer:
            pad_token = None
            eos_token = "<eos>"
            padding_side = "left"

        class VLProcessor:
            tokenizer = TextTokenizer()

        processor = VLProcessor()

        tokenizer = configure_processor_tokenizer(processor)

        self.assertIs(tokenizer, processor.tokenizer)
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "right")

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

    def test_sft_tokenization_keeps_start_of_long_answer(self) -> None:
        class CharacterTokenizer:
            eos_token_id = 0

            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                return {"input_ids": [ord(character) for character in text]}

        tokenized = tokenize_lora_row(
            CharacterTokenizer(),
            {"instruction": "solve", "input": "x", "output": "final-answer-followed-by-detail"},
            max_length=16,
        )
        targets = [token for token in tokenized["labels"] if token != -100]
        self.assertEqual(targets, [ord(character) for character in "final-an"])

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

    def test_visual_sft_example_uses_image_and_short_gold_answer(self) -> None:
        draft = Draft("qwen_agent_1", "visual", "wrong", [], [], [], [])
        improvement = SelfImprovement("qwen_agent_1", ["fix"], "verbose revision", [])
        example = build_lora_example(
            "Image: /tmp/chart.png\nQuestion: Which color?\nGold image elements: {}\n"
            "Gold reasoning: hidden\nGold final answer: Blue | blue",
            draft,
            [PeerEvaluation("qwen_agent_3", "qwen_agent_1", ["fix"], "r", [])],
            improvement,
        )
        self.assertEqual(example.image, "/tmp/chart.png")
        self.assertEqual(example.output, "Blue")
        self.assertNotIn("Gold final answer", example.input)
        self.assertNotIn("Gold image elements", example.input)
        self.assertIn("visual", example.instruction)

    def test_visual_gold_supervision_accepts_an_incorrect_draft(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_visual_gold_lora_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                ),
                trainer=trainer,
            )
            task = "Image: /tmp/chart.png\nQuestion: Which color?\nGold final answer: blue"
            draft = Draft("qwen_agent_1", "OCR specialist", "red", [], [], [], [])
            review = PeerEvaluation(
                "qwen_agent_3",
                "qwen_agent_1",
                ["recheck"],
                "wrong color",
                [],
                EvaluationScores(correctness=1.0, safety=1.0, efficiency=1.0),
            )

            updates = manager.update_from_round(
                task,
                [draft],
                [review],
                [SelfImprovement("qwen_agent_1", ["recheck"], "red", [], is_correct=False)],
            )

            self.assertTrue(updates[0].trained)
            row = json.loads(manager.dataset_path("qwen_agent_1").read_text(encoding="utf-8"))
            self.assertEqual(row["output"], "blue")
            self.assertIn("OCR specialist", row["instruction"])

    def test_visual_bare_short_answer_passes_gate_and_trains_lora(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_visual_lora_gate_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                ),
                trainer=trainer,
            )
            task = (
                "Image: /tmp/chart.png\n"
                "Question: Which platform has the heavy female audience?\n"
                "Gold final answer: Pinterest | pinterest"
            )
            draft = Draft("qwen_agent_1", "visual", "Pinterest", [], [], [], [])
            reviews = [
                PeerEvaluation(
                    "qwen_agent_3",
                    "qwen_agent_1",
                    ["keep the short answer"],
                    "correct visual answer",
                    [],
                    EvaluationScores(correctness=1.0, safety=1.0, efficiency=1.0),
                )
            ]
            improvement = SelfImprovement(
                "qwen_agent_1",
                ["keep the short answer"],
                "Pinterest",
                [],
            )

            updates = manager.update_from_round(task, [draft], reviews, [improvement])

            self.assertTrue(updates[0].trained)
            self.assertTrue(improvement.is_correct)
            self.assertEqual(len(trainer.calls), 1)
            row = json.loads(manager.dataset_path("qwen_agent_1").read_text(encoding="utf-8"))
            self.assertEqual(row["output"], "Pinterest")

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

    def test_verified_retrieval_label_trains_even_when_current_draft_is_not_grounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_verified_retrieval_lora_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                ),
                trainer=trainer,
            )
            task = (
                "TravelPlanner information-retrieval task.\n"
                "Task: Find a hotel.\n"
                "Candidate reference information:\n[source-1] Harbor Hotel costs 100.\n\n"
                "Gold retrieval targets:\nsource-1: Hotels\n\n"
                "Gold reference response:\nStay at Harbor Hotel for 100."
            )
            draft = Draft("qwen_agent_1", "general", "unsupported draft", [], [], [], [])
            review = PeerEvaluation(
                "qwen_agent_3",
                "qwen_agent_1",
                ["use source-1"],
                "ground the answer",
                [],
                EvaluationScores(correctness=1.0, safety=1.0, efficiency=1.0),
            )
            improvement = SelfImprovement("qwen_agent_1", ["use source-1"], "still unsupported", [])

            updates = manager.update_from_round(task, [draft], [review], [improvement])

            self.assertTrue(updates[0].trained)
            row = json.loads(manager.dataset_path("qwen_agent_1").read_text(encoding="utf-8").strip())
            self.assertIn("Summarize the evidence already retrieved by the server", row["instruction"])
            self.assertNotIn("Gold reference response", row["input"])
            self.assertIn("Relevant sources: source-1: Hotels", row["output"])
            self.assertIn("Stay at Harbor Hotel for 100.", row["output"])
            self.assertFalse(improvement.is_correct)

    def test_manager_accumulates_per_agent_datasets_and_trains_when_curated_examples_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b_magent_lora_test_") as temp:
            trainer = FakeLoraTrainer()
            manager = LoraEvolutionManager(
                LoraTrainingConfig(
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
                    output_dir=Path(temp) / "lora",
                    threshold=1,
                    require_correct_answer=False,
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
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
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
            self.assertEqual(
                bad_updates[0].reason,
                "improved answer did not pass a verifiable correctness or grounding gate",
            )
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
                    base_model_path="models/Qwen2.5-VL-7B-Instruct",
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
