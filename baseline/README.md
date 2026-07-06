# Qwen GSM8K baseline

This baseline runs one Qwen-style model directly on the GSM8K test split.

Input:

- `data/gsm8k/test.jsonl`

Output:

- `baseline/qwen_gsm8k_report.json`

Run:

```bash
python -m baseline.qwen_gsm8k --dataset-dir data/gsm8k --split test --output baseline/qwen_gsm8k_report.json
```

The current `EchoQwenModel` is an offline placeholder. Replace its `generate()`
method, or pass another object with `generate(question: str) -> str`, when the
real Qwen client is ready.

The evaluator extracts final answers from `#### answer` when present, otherwise
it uses the last number in the model output.
