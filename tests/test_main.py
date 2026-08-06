from __future__ import annotations

import pytest

from main import DEFAULT_LORA_OUTPUT_DIR, load_library_records, parse_args, print_prediction
from train.six_agent_private_train import AgentVote, VotingPrediction


def test_print_prediction_shows_each_agent_result(capsys) -> None:
    prediction = VotingPrediction(
        index=0,
        question="What is shown?",
        gold_answer="new york",
        gold_answers=["New York", "NYC"],
        votes=[
            AgentVote("qwen_agent_1", "New York", "New York"),
            AgentVote("qwen_agent_2", "NYC", "nyc"),
            AgentVote("qwen_agent_3", "Boston", "Boston"),
        ],
        final_answer="New York",
        correct=True,
        selected_agents=["qwen_agent_1", "qwen_agent_2", "qwen_agent_3"],
    )

    print_prediction(prediction, total=1)

    output = capsys.readouterr().out
    assert "[1/1] correct" in output
    assert "final_answer=New York" in output
    assert "evaluated_answer=New York" in output
    assert "gold=New York, NYC" in output
    assert "qwen_agent_1: correct | answer=New York" in output
    assert "qwen_agent_2: correct | answer=nyc" in output
    assert "qwen_agent_3: wrong | answer=Boston" in output


def test_optional_global_library_can_be_empty(tmp_path) -> None:
    library = tmp_path / "global_evaluation_library.jsonl"
    library.write_text("", encoding="utf-8")

    assert load_library_records(library, allow_empty=True) == []


def test_required_scheduler_tag_library_cannot_be_empty(tmp_path) -> None:
    library = tmp_path / "agent_training_tags.jsonl"
    library.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Scheduler tag library is empty"):
        load_library_records(library)


def test_main_defaults_to_7b_model_and_separate_lora_directory(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py"])

    args = parse_args()

    assert args.model_path.name == "Qwen2.5-VL-7B-Instruct"
    assert args.lora_output_dir == DEFAULT_LORA_OUTPUT_DIR
    assert args.lora_output_dir.name == "lora_adapters_qwen2_5_vl_7b"
    assert args.split == "validation"
