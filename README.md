# b_magent

`b_magent` 是一个本地四智能体自进化实验系统，主要用于 GSM8K 数学题训练、四智能体协作求解、互评、自我反思、LoRA 增量训练和多智能体蒸馏。

系统默认使用 4 个同构 Qwen 智能体：

- `qwen_agent_1`
- `qwen_agent_2`
- `qwen_agent_3`
- `qwen_agent_4`

每一轮训练中，系统选择 2 个智能体作为参与者解题，另外 2 个智能体作为评价者互评。参与者根据私有数据、专业经验库和评价经验库生成答案；评价者根据自己的评价经验库给出建议；参与者再根据建议做自我改进，并把经验写回自己的专业库；评价者也会把本轮评价经验写回自己的评价库。

## 环境

推荐使用可迁移的 conda 环境文件：

```bash
conda env create -f environment-portable.yml
conda activate cxh
```

环境文件说明：

- `environment.yml`：完整导出，包含本机路径 `prefix`
- `environment-portable.yml`：推荐使用，已去掉本机路径
- `environment-from-history.yml`：只包含 conda 显式安装历史，较干净但可能不够完整
- `conda-explicit.txt`：精确复刻 conda 包 URL，适合同系统 Linux 机器

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python scripts/check_setup.py
```

本地 Qwen 默认模型路径：

```text
models/Qwen2.5-1.5B-Instruct
```

该项目按离线本地模型运行，不会自动下载模型。可以通过 `--model-path` 指定其他本地模型目录。

## 数据

GSM8K 数据默认放在：

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
```

每行是一个 JSON 对象，至少包含：

- `question`
- `answer`

`b_magent.datasets.GSM8KDataset.extract_final_answer()` 会从标准 GSM8K 的 `#### answer` 标记后提取最终答案。

## 系统运行逻辑

### 1. 主入口

普通 demo 入口是 [main.py](/home/cxh/b_magent/main.py)：

1. `parse_args()` 解析 `--task`、`--output`、`--seed`
2. `build_default_agents()` 创建 4 个 `QwenAgent`
3. `seed_agent_libraries()` 初始化每个智能体的专业库和评价库
4. `MultiAgentWorkflow.run()` 执行一轮自进化
5. `MultiAgentWorkflow.export_report()` 导出 JSON 报告

运行：

```bash
python main.py --task "设计一个多智能体协作解决数学题的流程" --seed 1
```

默认输出：

```text
data/latest_report.json
```

### 2. 四智能体单轮协作流程

核心类是 [b_magent/workflow.py](/home/cxh/b_magent/b_magent/workflow.py) 里的 `MultiAgentWorkflow`。

`MultiAgentWorkflow.run(task, participant_names=None)` 的流程：

1. `_select_participants()` 选择 2 个参与解题的智能体
2. 剩余 2 个智能体自动成为评价者
3. 每个参与者调用 `QwenAgent.train_private_data()` 读取私有训练数据并写入专业库
4. 每个参与者调用 `QwenAgent.solve_task()` 生成初稿 `Draft`
5. 每个评价者调用 `QwenAgent.evaluate_peer()` 评价所有参与者初稿，生成 `PeerEvaluation`
6. 每个参与者调用 `QwenAgent.self_improve()` 根据评价建议生成改进答案，并写入专业经验库
7. 每个评价者调用 `QwenAgent.evolve_evaluation_library()` 把本轮评价经验写入评价经验库
8. 返回 `EvolutionReport`

每轮结构可以理解为：

```text
4 agents
  -> 2 participants solve
  -> 2 evaluators review
  -> participants update professional library
  -> evaluators update evaluation library
  -> optional LoRA update
```

### 3. 智能体内部逻辑

核心类是 [b_magent/agent.py](/home/cxh/b_magent/b_magent/agent.py) 里的 `QwenAgent`。

主要函数：

- `QwenAgent.__init__()`：绑定智能体名称、角色、数据目录、后端模型，并创建三个本地存储对象
- `train_private_data(task, batch_size=None)`：读取私有数据，按批次取样，把训练摘要写入 `professional_library.jsonl`
- `solve_task(task, private_training)`：检索专业库和评价库，调用后端 `solve()` 生成答案和推理轨迹
- `evaluate_peer(task, draft)`：检索评价库，调用后端 `suggest_improvements()` 生成互评建议
- `self_improve(task, draft, evaluations)`：合并互评建议，生成改进答案，校验 gold answer，并调用 `SelfEvolutionLibrary.evolve_professional()`
- `evolve_evaluation_library(task, all_evaluations)`：汇总自己的评价行为，调用 `SelfEvolutionLibrary.evolve_evaluation()`
- `_load_private_data()`：优先读取 `data/<agent>/private_data.jsonl`，其次读取 txt，再退回 GSM8K train，再退回内置样例
- `_next_private_batch()`：根据 `private_batch_size` 循环取每轮私有样本

辅助函数：

- `_extract_gold_final_answer()`：从任务文本里的 `Gold final answer:` 提取标准答案
- `_extract_final_answer()`：从模型输出里的 `####` 或最后一个数字提取预测答案
- `_strip_gold_annotations()`：解题时去掉 gold reasoning 和 gold final answer，避免泄漏答案

### 4. 经验库逻辑

经验库存储在每个智能体目录下：

```text
data/qwen_agent_*/professional_library.jsonl
data/qwen_agent_*/evaluation_library.jsonl
data/qwen_agent_*/private_data.jsonl
```

相关文件：

- [b_magent/library.py](/home/cxh/b_magent/b_magent/library.py)
- [b_magent/self_evolution.py](/home/cxh/b_magent/b_magent/self_evolution.py)
- [b_magent/seed.py](/home/cxh/b_magent/b_magent/seed.py)

关键函数：

- `EvolutionLibrary.add_record()`：向 JSONL 经验库追加一条 `LibraryRecord`
- `EvolutionLibrary.all_records()`：读取全部经验记录
- `EvolutionLibrary.search(query, limit=3)`：基于任务文本和标签做简单关键词检索
- `SelfEvolutionLibrary.evolve_professional()`：把参与者的答案、推理轨迹、互评建议沉淀为专业经验
- `SelfEvolutionLibrary.evolve_evaluation()`：把评价者的建议、理由、评分沉淀为评价经验
- `seed_agent_libraries()`：首次运行时为每个智能体写入基础专业经验和评价经验

## 训练入口

主要训练入口是 [train/four_agent_private_train.py](/home/cxh/b_magent/train/four_agent_private_train.py)。

### b-magent 自进化训练

默认模式是 `--mode b-magent`，默认后端是 `local-qwen`，默认启用 LoRA 和蒸馏。

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend local-qwen \
  --model-path models/Qwen2.5-1.5B-Instruct \
  --dataset-dir data/gsm8k \
  --rounds 200 \
  --output train/b_magent_training_report.json
```

快速逻辑 smoke test，不加载模型：

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend demo \
  --dataset-dir data/gsm8k \
  --rounds 1 \
  --disable-lora
```

只运行经验库自进化，不做 LoRA 和蒸馏：

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend local-qwen \
  --model-path models/Qwen2.5-1.5B-Instruct \
  --dataset-dir data/gsm8k \
  --disable-lora
```

`run_b_magent_training_entry()` 的完整训练逻辑：

1. 用 `GSM8KDataset.load("train")` 读取训练集
2. 用 `write_even_agent_private_datasets()` 把训练集平均拆到 4 个智能体的 `private_data.jsonl`
3. 用 `build_participant_schedule()` 生成每轮两个参与者的排班
4. 用 `expand_participant_schedule()` 根据 `--rounds` 扩展排班
5. 用 `build_default_agents()` 创建智能体并初始化经验库
6. 每轮用 `format_gsm8k_training_task()` 构造带 gold 信息的训练任务
7. 调用 `MultiAgentWorkflow.run()` 完成四智能体自进化
8. 如果启用 LoRA，调用 `LoraEvolutionManager.update_from_round()`
9. 汇总为 `BMagentTrainingReport` 并通过 `export_json_report()` 写出

### 参数含义

常用参数：

- `--mode b-magent`：运行四智能体自进化训练
- `--backend local-qwen`：使用本地 Qwen 模型
- `--backend demo`：使用确定性 demo 后端，不加载模型
- `--rounds`：训练轮数，传 `0` 时自动覆盖平均拆分后的私有训练数据
- `--private-batch-size`：每轮参与者读取多少条私有样本
- `--enable-lora` / `--disable-lora`：开启或关闭 LoRA
- `--lora-threshold`：兼容旧命令的保留参数；现在只要智能体有精选 SFT 数据集样本就触发 LoRA 训练

## 后端模型逻辑

### Demo 后端

[b_magent/backend.py](/home/cxh/b_magent/b_magent/backend.py) 的 `DemoQwenBackend` 用于不加载模型的确定性测试。

关键函数：

- `DemoQwenBackend.solve()`：根据任务、私有样本、专业经验和评价约束拼出一个可复现答案
- `DemoQwenBackend.suggest_improvements()`：返回固定结构的改进建议和评分

### 本地 Qwen 后端

[b_magent/local_qwen.py](/home/cxh/b_magent/b_magent/local_qwen.py) 封装本地 Qwen 推理。

关键类和函数：

- `QwenGenerationConfig`：控制 `max_new_tokens`、`temperature`、`top_p`、`do_sample`
- `LocalQwenEngine.__init__()`：保存模型路径、设备、dtype、生成参数
- `LocalQwenEngine.generate(prompt, adapter_path=None)`：加载模型，套 chat template，必要时加载 LoRA adapter，然后生成文本
- `LocalQwenEngine._load()`：懒加载 `AutoTokenizer` 和 `AutoModelForCausalLM`
- `LocalQwenEngine._load_adapter_model()`：用 PEFT 加载 agent adapter，并根据文件指纹刷新缓存
- `LocalQwenEngine.unload()`：释放模型和 CUDA 缓存
- `LocalQwenEvolutionBackend.solve()`：把任务、私有样本和经验库组织成 prompt，调用本地 Qwen 解题
- `LocalQwenEvolutionBackend.suggest_improvements()`：让本地 Qwen 给出 3 到 5 条评价建议和分数
- `LocalQwenEvolutionBackend.release_model_memory()`：LoRA 训练前释放推理模型显存
- `LocalQwenAgentModel.generate()`：投票评测时每个智能体生成自己的答案，优先使用该智能体的 LoRA adapter

LoRA adapter 选择顺序：

1. 如果 `data/lora_adapters/<agent>/adapter` 可用，使用该智能体自己的 LoRA adapter
2. 如果不存在，就使用原始 base model

## LoRA 自进化逻辑

LoRA 逻辑在 [b_magent/lora.py](/home/cxh/b_magent/b_magent/lora.py)。

默认输出：

```text
data/lora_adapters/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters/qwen_agent_*/current_sft_dataset.jsonl
data/lora_adapters/qwen_agent_*/lora_state.json
data/lora_adapters/qwen_agent_*/adapter/
```

核心类：

- `LoraTrainingConfig`：LoRA 训练配置，包含 base model、阈值、序列长度、学习率、LoRA rank 等
- `LoraSFTExample`：一条 SFT 样本，包含 `instruction`、`input`、`output`
- `LoraUpdate`：一次 LoRA 更新结果
- `AgentLoraState`：每个智能体的 LoRA 状态
- `PeftSFTLoraTrainer`：实际 PEFT/Transformers LoRA SFT 训练器
- `LoraEvolutionManager`：从自进化轮次中筛选样本并触发训练

关键函数：

- `LoraEvolutionManager.update_from_round()`：从一轮 `Draft`、`PeerEvaluation`、`SelfImprovement` 中生成 LoRA 样本
- `add_example_if_usable()`：检查样本是否可用，包括是否有评价、评分是否达标、gold answer 是否正确、是否重复
- `train_agent_on_curated_dataset()`：只要有精选 SFT 数据集样本，就用精选 SFT 数据集训练 adapter
- `build_lora_example()`：把任务、轨迹、评价报告和改进答案转换成 SFT 样本
- `format_trajectory()`：格式化智能体推理轨迹和工具调用
- `format_evaluation_report()`：格式化评价者评分、建议和理由
- `format_lora_prompt()`：把 SFT 样本拼成训练 prompt
- `append_lora_example()`：追加写入精选数据集 `sft_dataset.jsonl`
- `copy_lora_dataset()`：把精选数据集复制为本轮训练用的 `current_sft_dataset.jsonl`
- `hash_lora_example()`：对样本做哈希去重
- `is_improved_answer_correct()`：当任务含 gold answer 时，检查改进答案是否正确
- `write_lora_metadata()`：训练完成后写 adapter 元数据

LoRA 训练目标：

```text
冻结 base model，只训练每个智能体自己的 LoRA adapter。
训练样本来自“原始回答 + 互评反馈 + 自我改进后的答案”。
```

## 多智能体蒸馏逻辑

蒸馏逻辑在 [b_magent/distillation.py](/home/cxh/b_magent/b_magent/distillation.py)。

默认输出：

```text
data/lora_adapters/qwen_agent_*/distillation_dataset.jsonl
data/lora_adapters/qwen_agent_*/distillation_state.json
data/lora_adapters/qwen_agent_*/distilled_adapter/
```

核心类：

- `DistillationConfig`：蒸馏训练配置
- `TeacherAdapter`：一个教师 adapter 的路径、权重、样本数和版本
- `DistillationUpdate`：一次蒸馏更新结果
- `DistillationState`：每个智能体的蒸馏状态
- `PeftManyToManyDistillationTrainer`：多教师 KL 蒸馏训练器
- `DistillationManager`：收集教师、刷新蒸馏数据集、触发蒸馏训练

关键函数：

- `DistillationConfig.from_lora_config()`：从 LoRA 配置派生蒸馏配置
- `DistillationManager.update_from_lora_updates()`：根据 LoRA 更新决定哪些智能体需要刷新蒸馏数据
- `collect_teachers()`：收集所有已经训练过的 agent LoRA adapter 作为教师
- `normalize_teacher_weights()`：按教师样本数归一化教师权重
- `refresh_agent_dataset()`：把该智能体的 SFT 数据同步到蒸馏数据集
- `maybe_train_agent_distillation()`：达到阈值后训练该智能体的 `distilled_adapter`
- `append_distillation_row()`：追加蒸馏样本
- `hash_distillation_source()`：蒸馏样本去重
- `write_distillation_metadata()`：写入蒸馏元数据和损失公式

蒸馏目标：

```text
每个智能体保留自己的 private distilled adapter。
其他智能体的 adapter 只作为 teacher，不保存公共 adapter。
```

损失形式：

```text
p_T = sum_i alpha_i * p_i, sum_i alpha_i = 1
L_KD = T^2 * D_KL(p_T || p_S)
L = sft_weight * L_SFT + kd_weight * L_KD
```

## 投票评测逻辑

四智能体投票评测入口同样在 [train/four_agent_private_train.py](/home/cxh/b_magent/train/four_agent_private_train.py)。

运行：

```bash
python -m train.four_agent_private_train \
  --mode local-qwen-vote \
  --model-path models/Qwen2.5-1.5B-Instruct \
  --dataset-dir data/gsm8k \
  --test-limit 100 \
  --lora-output-dir data/lora_adapters \
  --output train/four_agent_trained_voting_100_report.json
```

关键函数：

- `build_four_local_qwen_agents()`：共享一个 `LocalQwenEngine`，构造 4 个 `LocalQwenAgentModel`
- `run_four_agent_voting_on_test()`：对 test set 中每道题让 4 个智能体分别回答
- `majority_vote()`：对 4 个答案做多数投票；平票时保留先出现的答案
- `format_voting_prediction_detail()`：格式化每道题的投票结果

## Baseline 逻辑

单模型 Qwen baseline 在 [baseline/qwen_gsm8k.py](/home/cxh/b_magent/baseline/qwen_gsm8k.py)。

运行 smoke test：

```bash
python -m baseline.qwen_gsm8k \
  --dataset-dir data/gsm8k \
  --split test \
  --limit 1 \
  --output baseline/qwen_gsm8k_smoke_report.json
```

运行前 100 条测试：

```bash
python -m baseline.qwen_gsm8k \
  --dataset-dir data/gsm8k \
  --split test \
  --output baseline/qwen_gsm8k_report.json
```

关键函数：

- `build_local_qwen_baseline_model()`：构建本地 Qwen baseline 模型
- `run_qwen_gsm8k_baseline()`：逐题生成答案并统计准确率
- `extract_numeric_answer()`：优先从 `####` 后提取答案，否则取最后一个数字
- `normalize_answer()`：去掉逗号和句点，把整数小数规范化
- `export_report()`：写出 baseline JSON 报告

## 重要数据结构

数据结构定义在 [b_magent/models.py](/home/cxh/b_magent/b_magent/models.py)。

- `LibraryRecord`：经验库中的一条记录
- `Draft`：参与者初稿，包括答案、推理轨迹、私有样本、经验库检索结果和工具调用
- `EvaluationScores`：评价分数，包含 correctness、safety、efficiency
- `PeerEvaluation`：评价者对某个参与者的建议、理由和分数
- `SelfImprovement`：参与者根据互评建议生成的改进结果
- `EvaluationEvolution`：评价者更新评价库的结果
- `EvolutionReport`：一轮完整自进化报告

训练报告结构在 [train/four_agent_private_train.py](/home/cxh/b_magent/train/four_agent_private_train.py)：

- `BMagentTrainingRound`：一轮 b-magent 训练摘要
- `BMagentTrainingReport`：多轮 b-magent 训练报告
- `VotingPrediction`：单题投票结果
- `VotingReport`：投票评测汇总
- `MultiAgentTrainingReport`：旧 placeholder 训练报告

## 产物目录

常见运行产物：

```text
data/latest_report.json
train/b_magent_training_report.json
baseline/qwen_gsm8k_report.json
train/four_agent_trained_voting_100_report.json
```

每个智能体的经验库：

```text
data/qwen_agent_*/private_data.jsonl
data/qwen_agent_*/professional_library.jsonl
data/qwen_agent_*/evaluation_library.jsonl
```

每个智能体的 LoRA 和蒸馏产物：

```text
data/lora_adapters/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters/qwen_agent_*/lora_state.json
data/lora_adapters/qwen_agent_*/adapter/
data/lora_adapters/qwen_agent_*/distillation_dataset.jsonl
data/lora_adapters/qwen_agent_*/distillation_state.json
data/lora_adapters/qwen_agent_*/distilled_adapter/
```

## 推荐运行顺序

1. 检查环境：

```bash
python scripts/check_setup.py
```

2. 先跑不加载模型的逻辑测试：

```bash
python -m train.four_agent_private_train --mode b-magent --backend demo --dataset-dir data/gsm8k --rounds 1 --disable-lora
```

3. 再跑本地 Qwen 自进化训练：

```bash
python -m train.four_agent_private_train --mode b-magent --backend local-qwen --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --rounds 200
```

4. 最后用训练后的 4 个智能体投票评测：

```bash
python -m train.four_agent_private_train --mode local-qwen-vote --model-path models/Qwen2.5-1.5B-Instruct --dataset-dir data/gsm8k --test-limit 100 --lora-output-dir data/lora_adapters --output train/four_agent_trained_voting_100_report.json
```
