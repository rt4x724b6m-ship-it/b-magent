# Qwen GSM8K baseline

This baseline runs the original local Qwen2.5-1.5B-Instruct model directly on
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

The model must already exist at `models/Qwen2.5-1.5B-Instruct`, or you must pass
another local directory with `--model-path`. The script sets
`local_files_only=True` when loading Transformers weights, so a missing model
path fails immediately instead of downloading from Hugging Face.

The evaluator extracts final answers from `#### answer` when present, otherwise
it uses the last number in the model output.
