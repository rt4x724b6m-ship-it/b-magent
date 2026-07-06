# Four-agent private training

This runner trains four Qwen-style agents separately on different private
GSM8K training splits, then evaluates each agent on the shared test split.

Input:

- `data/gsm8k/train.jsonl`
- `data/gsm8k/test.jsonl`

Default schedule:

- 3 rounds
- 32 batches per round
- batch size 1

Run:

```bash
python -m train.three_agent_private_train --dataset-dir data/gsm8k --output train/four_agent_training_report.json
```

The current `MemoryQwenModel` is an offline placeholder. Replace it with a real
Qwen training wrapper that implements:

```python
train_batch(batch)
generate(question)
```

The four agents do not share training state in this runner. Their private data
is produced by splitting `train.jsonl` round-robin across:

- `qwen_planner`
- `qwen_executor`
- `qwen_reviewer`
- `qwen_verifier`
