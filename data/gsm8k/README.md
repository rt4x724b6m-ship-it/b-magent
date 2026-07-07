# GSM8K local data

This directory contains the official GSM8K train/test split from
`openai/gsm8k` (`main` configuration).

Expected files:

- `train.jsonl` with 7,473 samples
- `test.jsonl` with 1,319 samples
- optional `raw.jsonl` if you want the framework to split one local file

Expected row format:

```json
{"question": "question text", "answer": "reasoning text #### final_answer"}
```

The framework reads `train.jsonl` during each participant agent's private
training stage. It extracts the final answer from the text after `####` when
that marker is present.

To split one local raw file from code:

```python
from pathlib import Path
from b_magent.datasets import GSM8KDataset

dataset = GSM8KDataset(Path("data/gsm8k"))
dataset.split_raw_jsonl(Path("data/gsm8k/raw.jsonl"), test_ratio=0.2, seed=13)
```

This writes `train.jsonl` and `test.jsonl`. The default split is 80% train and
20% test.
