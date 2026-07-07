# b_magent

`b_magent` is a local Python demo for Qwen-based multi-agent workflows.

It currently contains three runnable paths:

- Four-agent self-evolution training: equal agents draft, evaluate peers, revise, and write private JSONL memory.
- GSM8K local Qwen baseline: runs `Qwen2.5-1.5B-Instruct` on test questions.
- Four-agent GSM8K runner: local-Qwen self-evolution training, offline demo smoke tests, or local-Qwen voting.

## Environment

The project expects Python 3.12 and these main packages:

```bash
pip install -r requirements.txt
```

The local Qwen baseline uses this default model path:

```text
models/Qwen2.5-1.5B-Instruct
```

This workspace is offline. Put the full model files at that path, or pass
`--model-path /path/to/Qwen2.5-1.5B-Instruct`.

Check the setup:

```bash
python scripts/check_setup.py
```

## Run

Four-agent self-evolution demo:

```bash
python main.py --task "设计一个多智能体协作解决数学题的流程" --seed 1
```

Output:

```text
data/latest_report.json
```

Local Qwen GSM8K smoke test:

```bash
python -m baseline.qwen_gsm8k --dataset-dir data/gsm8k --split test --limit 1 --output baseline/qwen_gsm8k_smoke_report.json
```

Full default GSM8K baseline, first 100 test samples:

```bash
python -m baseline.qwen_gsm8k --dataset-dir data/gsm8k --split test --output baseline/qwen_gsm8k_report.json
```

Four-agent local-Qwen self-evolution training:

```bash
python -m train.four_agent_private_train --mode b-magent --backend local-qwen --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --rounds 3 --output train/b_magent_training_report.json
```

At startup this mode evenly splits the prepared `data/gsm8k/train.jsonl`
training set into each agent's private file under
`data/qwen_agent_*/private_data.jsonl`. If the row count is not divisible by
four, the remainder is assigned one row at a time from `qwen_agent_1` onward.
The default CLI run uses 100 training rounds. Use `--rounds 0` to auto-cover the
split dataset: each round trains two participating agents, so the automatic
round count is derived from the number of private examples and
`--private-batch-size`. Each participating agent loads only that small private
batch per round, which keeps prompts bounded.

Per-agent LoRA self-evolution and many-to-many distillation are enabled by
default. Disable them explicitly when you only want the JSONL experience-library
loop:

```bash
python -m train.four_agent_private_train --mode b-magent --backend local-qwen --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --disable-lora --disable-distillation
```

Default LoRA/distillation run:

```bash
python -m train.four_agent_private_train --mode b-magent --backend local-qwen --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --rounds 100
```

Per-agent adapters are written under `data/lora_adapters/qwen_agent_*/adapter`.
Each distilled adapter is written under
`data/lora_adapters/qwen_agent_*/distilled_adapter` and is preferred by that
same agent on later local-Qwen rounds when distillation is enabled. No common
adapter is stored.

Fast logic-only smoke test without loading model weights:

```bash
python -m train.four_agent_private_train --mode b-magent --backend demo --dataset-dir data/gsm8k --rounds 1 --disable-lora --disable-distillation
```

Four-agent local-Qwen voting:

```bash
python -m train.four_agent_private_train --dataset-dir data/gsm8k --local-qwen --test-limit 1 --output train/four_agent_qwen_voting_smoke_report.json
```

Vote with the four trained agents on the first 100 GSM8K test questions:

```bash
python -m train.four_agent_private_train --mode local-qwen-vote --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --test-limit 100 --lora-output-dir data/lora_adapters --output train/four_agent_trained_voting_100_report.json
```

## Data

GSM8K files are expected at:

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
```

Each row should contain `question` and `answer`. The reader extracts final
answers from the canonical `#### answer` marker.
