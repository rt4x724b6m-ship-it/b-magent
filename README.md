# b_magent

`b_magent` 是一个本地四智能体自进化实验系统，主要用于 GSM8K 数学题训练、四智能体协作求解、互评、自我反思和 LoRA 增量训练。当前仓库只保留直接驱动经验库和 LoRA 更新的主流程。

系统默认使用 4 个同构 Qwen 智能体：

- `qwen_agent_1`
- `qwen_agent_2`
- `qwen_agent_3`
- `qwen_agent_4`

每一轮训练中，系统选择 2 个智能体作为参与者解题，另外 2 个智能体作为评价者互评。参与者根据私有数据、专业经验库和评价经验库生成答案；评价者根据自己的评价经验库给出建议；参与者再根据建议做自我改进，并把经验写回自己的专业库；评价者也会把本轮评价经验写回自己的评价库。

## 环境

推荐直接用仓库中的可迁移 Conda 环境文件创建完整环境：

```bash
git clone https://github.com/rt4x724b6m-ship-it/b-magent.git
cd b-magent
conda env create -f environment.yml
conda activate b-magent
```

以后仓库更新了依赖，可以在项目目录同步现有环境：

```bash
git pull
conda env update -n b-magent -f environment.yml --prune
```

`environment.yml` 是项目的主环境声明，只包含可跨设备解析的直接依赖，不包含
本机 `prefix`、Conda 缓存路径或平台相关的 build 字符串。仓库里的
`environment-portable.yml` 和 `environment-from-history.yml` 是旧环境快照，仅供
排查历史环境，不建议用于新设备安装。

如果不使用 Conda，也可以在 Python 3.12 虚拟环境中用 pip 安装同一组依赖：

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
8. `qwen_server_agent` 聚合本轮互评经验，写入全局评价经验库
9. 返回 `EvolutionReport`

每轮结构可以理解为：

```text
4 agents
  -> 2 participants solve
  -> 2 evaluators review
  -> participants update professional library
  -> evaluators update evaluation library
  -> server agent aggregate global evaluation experience
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

专业经验自反思标签：

- `self_improve()` 完成答案修订和反思后，会再次让当前 agent 根据任务、原答案、改进答案、评价建议和反思生成 1 到 5 个经验标签。
- agent 优先复用 `arithmetic`、`final-answer`、`verification`、`boundary`、`structure`，也可以生成更具体的可复用标签。
- 标签入库前统一转换为小写 `kebab-case`，并执行去重、长度、数量和保留标签检查。
- agent 未实现标签生成或未返回有效 JSON 时，系统继续使用固定关键词标签，不影响训练流程。
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

默认模式是 `--mode b-magent`，默认后端是 `local-qwen`，默认启用 LoRA。每次启动该模式时，脚本会先清理上一轮生成的 agent 专业库、私有数据、评价库、server 全局评价库和 LoRA adapter 目录，再开始新的训练。

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

只运行经验库自进化，不做 LoRA：

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
7. 训练轮次开始前调用 `downlink_global_evaluation_experience()`，把 server 全局评价经验下发到各 agent 评价库
8. 调用 `MultiAgentWorkflow.run()` 完成四智能体自进化，并由 `qwen_server_agent` 聚合本轮全局评价经验
9. 如果启用 LoRA，调用 `LoraEvolutionManager.update_from_round()`
10. 汇总为 `BMagentTrainingReport` 并通过 `export_json_report()` 写出

### 参数含义

常用参数：

- `--mode b-magent`：运行四智能体自进化训练
- `--backend local-qwen`：使用本地 Qwen 模型
- `--backend demo`：使用确定性 demo 后端，不加载模型
- `--rounds`：训练轮数，传 `0` 时自动覆盖平均拆分后的私有训练数据
- `--private-batch-size`：每轮参与者读取多少条私有样本
- `--enable-lora` / `--disable-lora`：开启或关闭 LoRA
- `--lora-output-dir`：每个智能体的 LoRA SFT 数据集和 adapter 输出目录，默认 `data/lora_adapters`
- `--lora-threshold`：每个智能体累计多少条新精选样本后刷新一次 LoRA，默认 `10`；训练结束会刷新不足阈值的剩余样本
- `--lora-max-seq-length`：LoRA 训练最大序列长度，默认 `1024`
- `--lora-train-batch-size`：单卡 LoRA batch size，默认 `4`
- `--lora-gradient-accumulation-steps`：LoRA 梯度累积步数，默认 `1`
- `--lora-epochs`：每次 LoRA SFT 的 epoch 数，默认 `1.0`
- `--lora-learning-rate`：LoRA 学习率，默认 `2e-4`
- `--lora-min-evaluation-score`：接受 LoRA 样本所需的最低评价分数，默认 `0.6`
- `--allow-uncorrect-lora-labels`：当任务包含 gold answer 时，允许错误的自我改进答案进入 LoRA SFT 数据集
- `--seed`：传给 b-magent workflow 的随机种子

兼容/评测模式参数：

- `--mode placeholder`：运行旧的离线 `MemoryQwenModel` 占位训练器
- `--mode local-qwen-vote`：使用 4 个本地 Qwen agent 对 test set 做投票评测
- `--local-qwen`：`--mode local-qwen-vote` 的旧别名
- `--batches-per-round`、`--batch-size`、`--private-train-size`：仅用于旧 placeholder 模式
- `--test-limit`：用于 baseline/投票评测的测试样本数量，默认沿用 `STANDARD_TEST_LIMIT`

### Server 全局评价经验

四智能体 workflow 会额外创建一个 server agent：

```text
data/qwen_server_agent/global_evaluation_library.jsonl
```

每轮结束时，`qwen_server_agent` 会聚合所有评价者对参与者初稿的互评经验，写入全局评价经验库。下一轮开始前，训练入口会检索与当前任务相关的全局经验，并以 `global-downlink` 记录追加到每个 agent 的 `evaluation_library.jsonl`，避免每个 agent 只在自己的局部评价轨迹里演化。

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
data/qwen_server_agent/global_evaluation_library.jsonl
```

每个智能体的 LoRA 产物：

```text
data/lora_adapters/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters/qwen_agent_*/current_sft_dataset.jsonl
data/lora_adapters/qwen_agent_*/lora_state.json
data/lora_adapters/qwen_agent_*/adapter/
```

## 两层任务编排设计（规划中）

当前训练主线使用 GSM8K 数学题，样本结构为 `question`、`answer` 和
`final_answer`。这种单题单答案数据适合验证求解、自我反思和 LoRA 更新，
但不包含任务拆分、子任务依赖或客户端选择的监督信号。

下一阶段计划扩展为两层架构：

```text
用户任务
  -> 第一层：任务拆分服务器
  -> 子任务、依赖关系、所需能力
  -> 第二层：按能力选择客户端智能体
  -> 客户端执行、评价、反思与 LoRA 更新
```

截至目前，`FirstLayerServer` 仍是架构占位，不会实际拆分任务、维护任务图
或调度客户端；四个 `qwen_agent_*` 也仍是同构的通用智能体。以下数据集是
后续设计和数据接入的候选，不表示它们已经下载、转换或用于训练。

### 候选数据集

| 目标 | 数据集 | 可用信息 | 推荐用途 |
| --- | --- | --- | --- |
| 第一层任务拆分 | [TaskBench](https://huggingface.co/datasets/microsoft/Taskbench) | Tool Graph、子任务与工具依赖 | 构建任务图、学习子任务排序和并行关系 |
| 第一层规划补充 | [UltraTool](https://github.com/JoeYing1019/UltraTool) | 独立的自然语言多步骤计划与工具任务 | 训练或评测先规划、后执行的流程 |
| 第二层客户端选择 | [MetaTool / ToolE](https://github.com/HowieHwong/MetaTool) | 查询、目标工具、工具描述、相似候选与多工具场景 | 将工具视为客户端能力，训练候选智能体排序和拒选 |
| 客户端能力扩展 | [ToolBench](https://github.com/OpenBMB/ToolBench) | 真实 API 描述与多步调用路径 | 生成跨领域客户端能力卡和多步执行样本 |
| 高质量函数调用补充 | [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 经验证的复杂函数调用对话和 API 模式 | 补充复杂参数、并行调用和多工具组合 |
| 工具/路由评测 | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | 候选函数选择、并行/多函数、拒选及多轮场景 | 验证客户端路由和调用格式的正确性 |
| 长程交互评测 | [tau-bench](https://github.com/sierra-research/tau2-bench) | 领域策略、工具集和多轮任务 | 验证拆分后委派的端到端完成度 |
| 泛化保留测试 | [GAIA](https://huggingface.co/gaia-benchmark/GAIA) | 多模态、工具型通用助理任务 | 作为最终泛化评测，不作为主要拆分监督数据 |

建议的最小数据闭环是先接入 `TaskBench + MetaTool`：前者提供
`subtasks + depends_on`，后者提供 `candidate_agents + selected_agent` 的监督
范式。随后再使用 `ToolBench` 和 `ToolACE` 扩展客户端能力覆盖面，并以
`BFCL`、`tau-bench` 和 `GAIA` 分别评估路由、长程执行和泛化能力。

### 目标样本格式

后续数据适配不应继续局限于 GSM8K 的 `question/answer` 格式，而应保留任务图
与路由标签，例如：

```json
{
  "task": "用户原始任务",
  "subtasks": [
    {
      "id": "s1",
      "instruction": "子任务描述",
      "depends_on": [],
      "required_capabilities": ["retrieval", "web-search"],
      "candidate_agents": ["agent_retrieval", "agent_general"],
      "selected_agent": "agent_retrieval",
      "expected_tool_or_action": "search"
    }
  ]
}
```

客户端需要先定义清晰、可重叠但可区分的能力卡；否则即使引入上述数据，多个
同构智能体仍可能退化为随机分派或投票。公开候选数据主要为英文，转换为中文时
应保留工具名、参数名和依赖边的原始语义。

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
