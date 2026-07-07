# b_magent training entry

`four_agent_private_train.py` is now the training entry for the `b_magent`
package agents by default.

Default mode:

```bash
python -m train.four_agent_private_train --dataset-dir data/gsm8k --rounds 3 --model-path models/Qwen2.5-1.5B-Instruct
```

This runs four equal `b_magent` agents. They share the same workflow and model
interface, but each agent writes only to its own professional evolution library
and evaluation evolution library:

- `qwen_agent_1`
- `qwen_agent_2`
- `qwen_agent_3`
- `qwen_agent_4`

For each round it converts one GSM8K training sample into a task, runs the
multi-agent self-evolution workflow, and writes professional/evaluation memory
records under `data/<agent_name>/`.

Before the first round, the prepared training set at `data/gsm8k/train.jsonl`
is evenly split into `data/qwen_agent_*/private_data.jsonl`. Every row is used
once; when the total is not divisible by four, earlier agents receive one extra
row. The training report includes `private_dataset_counts`.

The default CLI run uses 100 training rounds. Use `--rounds 0` to auto-compute
enough rounds to cover the split private data. Each round has two task agents,
so 800 private examples with `--private-batch-size 1` becomes 400 rounds.
Increase `--private-batch-size` to consume more private examples per
participating agent per round without loading the full private split into one
prompt.

LoRA self-evolution and many-to-many distillation are enabled by default. Use
`--disable-lora --disable-distillation` when you only want the JSONL
experience-library loop:

```bash
python -m train.four_agent_private_train --dataset-dir data/gsm8k --rounds 100 --model-path models/Qwen2.5-1.5B-Instruct
```

The distillation stage keeps every agent's adapter as a teacher and trains each
agent's own `data/lora_adapters/qwen_agent_*/distilled_adapter` with SFT plus
KL-divergence to the weighted average teacher distribution. Teacher weights are
normalized from each agent's trained LoRA example count. No common adapter is
stored.

By default `--mode b-magent` uses `--backend local-qwen`, so each solve and
evaluation step calls the configured Qwen model. For a fast logic-only smoke
test that does not load a model, pass `--backend demo`.

Default output:

```text
train/b_magent_training_report.json
```

The old four-agent placeholder runner is still available:

```bash
python -m train.four_agent_private_train --mode placeholder --dataset-dir data/gsm8k --output train/four_agent_training_report.json
```

The local-Qwen voting runner is also still available:

```bash
python -m train.four_agent_private_train --mode local-qwen-vote --dataset-dir data/gsm8k --test-limit 1 --output train/four_agent_qwen_voting_report.json
```
