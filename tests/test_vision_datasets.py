from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from tests import _project_path  # noqa: F401
except ImportError:
    import _project_path  # type: ignore[no-redef]  # noqa: F401
_project_path.add_project_root_to_sys_path()

from b_magent.datasets import MultimodalBenchmarkDataset, VisionQADataset, VisionQASample, load_project_dataset
from b_magent.local_qwen import _extract_image_path
from b_magent.lora import is_improved_answer_correct
from train.four_agent_private_train import (
    extract_prediction_answer,
    format_inference_question,
    format_training_task,
    infographic_anls,
)
from scripts.prepare_vision_datasets import normalized_split_name


class VisionDatasetTest(unittest.TestCase):
    def test_normalizes_image_paths_and_multiple_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "infographicsvqa"
            root.mkdir()
            (root / "train.jsonl").write_text(json.dumps({
                "question_id": 7,
                "image": "images/7.png",
                "question": "What is the total?",
                "answers": ["42", "forty two"],
            }) + "\n", encoding="utf-8")

            samples = VisionQADataset(root).load("train")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].answers, ("42", "forty two"))
            self.assertEqual(samples[0].image_path, str(root / "images/7.png"))

    def test_combines_both_benchmarks_and_honors_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in MultimodalBenchmarkDataset.DATASETS:
                folder = root / name
                folder.mkdir()
                (folder / "test.jsonl").write_text(json.dumps({
                    "image": "images/a.png", "question": name, "answer": "yes"
                }) + "\n", encoding="utf-8")

            samples = MultimodalBenchmarkDataset(root).load("test", limit=1)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].dataset, "mm-vet")

    def test_combined_training_excludes_mmvet(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in MultimodalBenchmarkDataset.DATASETS:
                folder = root / name
                folder.mkdir()
                (folder / "train.jsonl").write_text(json.dumps({
                    "image": "a.png", "question": name, "answer": "yes"
                }) + "\n", encoding="utf-8")

            samples = MultimodalBenchmarkDataset(root).load("train")

            self.assertEqual([sample.dataset for sample in samples], ["infographicsvqa"])

    def test_official_splits_are_not_merged_into_training(self) -> None:
        self.assertEqual(normalized_split_name("infographicsvqa", "train"), "train")
        self.assertEqual(normalized_split_name("infographicsvqa", "validation"), "validation")
        self.assertEqual(normalized_split_name("infographicsvqa", "test"), "test")
        self.assertEqual(normalized_split_name("mm-vet", "train"), "test")

    def test_direct_visual_dataset_is_not_misread_as_gsm8k(self) -> None:
        dataset = load_project_dataset(Path("/tmp/infographicsvqa"))
        self.assertIsInstance(dataset, VisionQADataset)

    def test_visual_training_task_includes_image_and_answers(self) -> None:
        sample = VisionQASample(
            question="How many?", answer="2", final_answer="2", answers=("2", "two"),
            image_path="/tmp/example.png", dataset="mm-vet", sample_id="1",
            image_elements={"summary": "two objects", "objects": ["objects"]},
        )
        task = format_training_task(sample)
        self.assertIn("Image: /tmp/example.png", task)
        self.assertIn("Gold final answer: 2 | two", task)
        self.assertIn("Gold image elements:", task)

    def test_extracts_only_existing_task_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "sample.png"
            image.touch()
            self.assertEqual(_extract_image_path(f"Image: {image}\nQuestion: What?"), image)
            self.assertIsNone(_extract_image_path("Image: /missing/image.png"))

    def test_visual_inference_keeps_image_and_text_answer(self) -> None:
        sample = VisionQASample(
            question="Is the light on?", answer="yes", final_answer="yes", answers=("yes",),
            image_path="/tmp/example.png", dataset="mm-vet",
        )
        self.assertIn("Return JSON with image_elements and final_answer.", format_inference_question(sample))
        self.assertEqual(extract_prediction_answer("Final answer: Yes.", sample), "yes")
        self.assertEqual(
            extract_prediction_answer('{"image_elements":{},"final_answer":"Yes."}', sample),
            "yes",
        )

    def test_lora_quality_gate_accepts_any_visual_reference(self) -> None:
        task = "Question: What is shown?\nGold final answer: bicycle | bike"
        self.assertTrue(is_improved_answer_correct(task, "Answer: Bike."))

    def test_infographic_anls_tolerates_small_ocr_errors(self) -> None:
        self.assertEqual(infographic_anls("Coca cola", ("Coca Cola",)), 1.0)
        self.assertAlmostEqual(infographic_anls("CocaCola", ("Coca Cola",)), 8 / 9)
        self.assertEqual(infographic_anls("unrelated", ("Coca Cola",)), 0.0)


if __name__ == "__main__":
    unittest.main()
