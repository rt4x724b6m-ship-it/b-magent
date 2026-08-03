from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.local_vlm_eval import clean_prediction, evaluate, normalize_answer, score_anls


class FixedVisionModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.image_paths: list[list[Path | str]] = []

    def generate_multimodal(self, prompt: str, image_paths: list[Path | str]) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        return next(self.outputs)


def test_answer_cleaning_and_normalization() -> None:
    assert clean_prediction("Answer: Pinterest\n") == "Pinterest"
    assert normalize_answer("LinkedIn, Facebook") == "linkedin facebook"
    assert score_anls("Pinteres", ["Pinterest"]) > 0.5


def test_evaluate_uses_image_and_scores_multiple_answers(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "sample.png")
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "sample-1",
            "image": "images/sample.png",
            "question": "Which platform?",
            "answers": ["Pinterest", "Pinterest platform"],
            "image_elements": {"visible_text": ["do not leak this field"]},
        }) + "\n",
        encoding="utf-8",
    )
    model = FixedVisionModel(["Answer: pinterest"])

    report = evaluate(dataset, model=model, model_path="mock", limit=1)

    assert report.total == 1
    assert report.correct == 1
    assert report.accuracy == 1.0
    assert report.normalized_accuracy == 1.0
    assert report.predictions[0].normalized_exact_match
    assert "Question: Which platform?" in model.prompts[0]
    assert model.image_paths == [[(image_dir / "sample.png").resolve()]]
    assert "do not leak" not in model.prompts[0]
