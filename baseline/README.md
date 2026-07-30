# Qwen GSM8K baseline

This baseline runs the original local Qwen2.5-VL-3B-Instruct model directly on
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

The model must already exist at `models/Qwen2.5-VL-3B-Instruct`, or you must pass
another local directory with `--model-path`. The script sets
`local_files_only=True` when loading Transformers weights, so a missing model
path fails immediately instead of downloading from Hugging Face.

The evaluator extracts final answers from `#### answer` when present, otherwise
it uses the last number in the model output.

## Local multimodal image VQA evaluation

This evaluator loads the local `Qwen2.5-VL-3B-Instruct` weights and tests them
on the 100-example InfoGraphicsVQA test split. Only the original image and
question are sent to the model; precomputed `image_elements` in the JSONL are
deliberately ignored.

Run a quick 10-example check:

```bash
python -m baseline.local_vlm_eval --limit 10 --output baseline/local_vlm_report_10.json
```

Run the full test split:

```bash
python -m baseline.local_vlm_eval --limit 0 --output baseline/local_vlm_report.json
```

Useful options include `--model-path`, `--dataset`, `--dtype`, and
`--max-new-tokens`. The default `--max-pixels 1048576` bounds GPU memory use;
set it to `0` to use the model's original image resolution limit. The JSON
report contains every prediction and summarizes:

- exact accuracy;
- case/punctuation-normalized accuracy;
- ANLS, which gives partial credit for close OCR-style answers;
- inference errors and per-sample/average latency.
