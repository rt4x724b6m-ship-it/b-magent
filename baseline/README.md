# Qwen GSM8K baseline

This baseline runs the local Qwen2.5-VL-7B-Instruct model directly on
the first 100 questions of the official GSM8K test split. It does not use
networking, external tools, voting, or agent memory.

Input:

- `data/gsm8k/test.jsonl`

Output:

- `baseline/qwen_gsm8k_report.json`

Run:

```bash
python -m baseline.qwen_gsm8k --dataset-dir data/gsm8k --split test --output baseline/qwen_gsm8k_report.json
```

The default `--limit` is 100.

The model must already exist at `models/Qwen2.5-VL-7B-Instruct`, or you must pass
another local directory with `--model-path`. The script sets
`local_files_only=True` when loading Transformers weights, so a missing model
path fails immediately instead of downloading from Hugging Face.

The evaluator extracts final answers from `#### answer` when present, otherwise
it uses the last number in the model output.

# Untrained local VLM baseline

This evaluator runs the original local Qwen2.5-VL checkpoint directly on the
InfographicsVQA test split. It does not load LoRA adapters, agent training data,
memory, routing, or voting. Its normalized accuracy and ANLS use the same
normalization and Levenshtein scoring functions as `main.py`.

Run the first 100 test questions:

```bash
python -m baseline.local_vlm_eval
```

Run the full test split or override paths:

```bash
python -m baseline.local_vlm_eval \
  --limit 0 \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset data/infographicsvqa/test.jsonl \
  --output baseline/local_vlm_report.json
```

The JSON report includes every prediction, reference answer, per-sample ANLS,
latency, errors, normalized accuracy, and aggregate ANLS. The default checkpoint
is loaded with `local_files_only=True`, so the command never downloads a model.

## Validation first 500 samples

Run the standalone base Qwen2.5-VL-7B model on the first 500 labeled validation
questions and score with ANLS:

```bash
python -m baseline.qwen_validation_500
```

The defaults are `models/Qwen2.5-VL-7B-Instruct`,
`data/infographicsvqa/validation.jsonl`, and
`baseline/qwen_validation_500_report.json`. Override them with `--model-path`,
`--dataset`, or `--output`.
