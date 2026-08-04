  # b_magent

`b_magent` 是一个本地六智能体自进化实验系统。每轮从六个客户端智能体中随机选择三个参与训练，剩余三个负责评价。

系统默认使用 6 个同构 Qwen 智能体：

- `qwen_agent_1`
- `qwen_agent_2`
- `qwen_agent_3`
- `qwen_agent_4`
- `qwen_agent_5`
- `qwen_agent_6`

每一轮训练中，系统随机选择 3 个智能体作为参与者，另外 3 个智能体作为评价者。随机选择可通过 `random_seed` 复现。

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
models/Qwen2.5-VL-7B-Instruct
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

TravelPlanner 等数据集使用“服务器检索、智能体总结”的训练契约。第二层服务器负责历史关键信息命中、网页搜索和候选证据准备；LoRA 与专业经验库只强化基于已有证据进行总结、整合、去重、冲突处理和约束保持的能力：

- 模型输入只包含用户问题和候选参考信息，不包含官方答案和官方相关来源标签。
- `reference_information` 会被解析为带 `source-N` 标识的候选资料，作为第二层服务器已检索到的证据，保留与官方回答相关的资料和部分困难负样本。
- LoRA 监督输出使用“官方相关来源 + 官方总结/计划”，不使用未验证的模型自生成答案作为标签。
- 当数据集没有官方标签时，默认 `require_correct_answer=True` 会拒绝该 LoRA 样本；只有显式关闭正确性门槛才能做实验性训练。
- 智能体当前回答只有同时满足官方回答覆盖率、证据 grounding 精度和数字覆盖率时，才会标记为 `curated-success-experience`；否则仅保存为错误反思经验。
- LoRA 默认上下文窗口为 `4096`，用于保留候选证据、任务约束和监督总结输出；训练指令明确禁止客户端再次搜索。
- 专业经验以 `summarization`、`information-synthesis`、`constraint-preservation` 等标签沉淀，第二层服务器据此匹配三个智能体并统一整合结果。

当前 TravelPlanner 训练集检查结果为 45/45 条样本包含候选参考信息和相关来源标签，模型可见输入中没有官方答案原文泄漏。最终检索准确率仍需要在独立验证集上计算 Recall@K、MRR、grounded precision 和 hallucination rate。

### GPT-5.6 Sol 回答验证

生产训练默认使用 `gpt-5.6-sol` 通过 OpenAI Responses API 判定智能体回答是否正确。判定是语义和约束导向的：只要回答达成用户要求的结果、满足所有显式硬约束，且重要事实能由候选证据支持，即认为正确；不要求措辞、顺序、格式或具体选择与官方答案完全一致。

训练和测试共用这一判定标准。训练时，GPT 结论决定回答进入成功专业经验还是错误反思经验；测试时，GPT 结论直接决定 `VotingPrediction.correct`，即使回答与官方参考的文字或具体方案不同，只要符合题目要求就计为正确。测试报告同时保存 `requirements_met`、`requirements_missed`、`unsupported_claims` 和 `answer_validation_rationale`。

TravelPlanner 的 validation/test 即使没有 `annotated_plan` 也会保留题目和 `reference_information` 供 GPT 语义验证。当前可加载数量为 train 45、validation 180、test 1000，所有样本均包含候选参考信息。

```bash
export OPENAI_API_KEY="your-openai-api-key"
# 可选：自定义兼容的 API 基础地址和推理强度
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_VALIDATOR_REASONING_EFFORT="medium"
```

常规训练无需额外参数，因为 `--answer-validator` 默认是 `gpt-5.6-sol`。API 请求失败或结构化判定无法解析时会直接停止训练，不会把故障误当成错误回答。评估理由、已满足要求、遗漏要求和无依据声明会一起写入专业经验轨迹。

离线测试或确定性调试可以显式回退到本地规则：

```bash
python -m train.four_agent_private_train --mode b-magent --answer-validator local
```

默认模式是 `--mode b-magent`，默认后端是 `local-qwen`，默认启用 LoRA。每次启动该模式时，脚本会先清理上一轮生成的 agent 专业库、私有数据、评价库、server 全局评价库和 LoRA adapter 目录，再开始新的训练。

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend local-qwen \
  --model-path models/Qwen2.5-VL-7B-Instruct \
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
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset-dir data/gsm8k \
  --disable-lora
```

`run_b_magent_training_entry()` 的完整训练逻辑：

1. 用 `GSM8KDataset.load("train")` 读取训练集
2. 用 `write_even_agent_private_datasets()` 把训练集平均拆到 4 个智能体的 `private_data.jsonl`
3. 用 `build_participant_schedule()` 生成每轮两个参与者的排班
4. 用 `expand_participant_schedule()` 根据 `--rounds` 扩展排班
5. 用 `build_default_agents()` 创建智能体并初始化经验库
6. 每轮用 `format_gsm8k_training_task()` 构造基于已检索证据的总结任务；模型可见输入会隐藏 gold 标签
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
- `--lora-output-dir`：每个智能体的 LoRA SFT 数据集和 adapter 输出目录，默认 `data/lora_adapters_qwen2_5_vl_7b`
- `--lora-threshold`：每个智能体累计多少条新精选样本后刷新一次 LoRA，默认 `10`；训练结束会刷新不足阈值的剩余样本
- `--lora-max-seq-length`：LoRA 训练最大序列长度，默认 `4096`
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

同一轮中的私有训练、专业经验进化和评价经验进化会生成任务标签，并以只包含元数据的形式上传到：

```text
data/qwen_server_agent/agent_training_tags.jsonl
data/qwen_server_agent/agent_training_tags/<agent_name>.jsonl
```

测试时，第二层先提取题目能力标签和风险标签，再与四个智能体的训练标签匹配并按得分选择前 3 个。只有这 3 个智能体生成回答，第二层服务器读取三份完整内容、取并集、消除冲突并生成最终答案。

### Server 历史关键信息缓存

第二层 server agent 还会维护已验证的历史关键信息库：

```text
data/qwen_server_agent/key_information_store.jsonl
```

启用第二层 `server_model` 时，`run_four_agent_voting_on_test()` 会默认创建并搜索该库，也可以显式传入 `server_key_information_store=server_agent.key_information_store` 共享已有 server agent 的存储器。精确题目或高相似的关键信息命中时，直接复用历史 server synthesis，不再调用第三层 client agent。未命中时才调用标签匹配度最高的 3 个 client agent，并把经 GPT-5.6 Sol 语义验证正确的服务器整合结果写回缓存。

普通训练状态重置会保留该历史库。只有显式传入 `reset_key_information_store=True` 时才会删除。推理报告中的 `server_cache_hit` 用于表示本次是否跳过了 client agent。

### 网页搜索

第二层可以通过正式 JSON 搜索 API 获取网页结果，再由受限制的抓取器提取网页正文。配置环境变量后，启用 `server_model` 的推理流程会自动创建网页搜索服务：

```bash
export WEB_SEARCH_PROVIDER="bing"  # generic, bing, brave, serper
export WEB_SEARCH_ENDPOINT="https://your-search-api.example/v1/search"
export WEB_SEARCH_API_KEY="your-api-key"
```

第二层题目分析只有在 `requires_web_search=true` 时才会联网。数学题、题目内已给出完整信息的问题不会触发搜索。价格、时刻表、天气、新闻等实时信息可以通过 `web_cache_ttl_seconds` 设置较短缓存时间。

网页结果缓存位于：

```text
data/qwen_server_agent/web_search_store.jsonl
```

缓存未过期时不再调用搜索 API 和网页抓取器。抓取器只允许公网 `http`/`https` 地址，拒绝 localhost、内网 IP 和重定向到内网的请求，并限制响应体和提取文本大小。推理报告会记录 `web_search_cache_hit` 和带 URL 的 `web_search_results`。

网页搜索还会提取 `og:image`、`twitter:image`、`img src`、`data-src` 和 `srcset` 中的图片地址。图片通过与网页相同的公网地址和重定向安全检查后下载，并使用 Pillow 验证真实图片格式、文件大小和像素数。默认每个页面最多下载 2 张、每道题最多下载 8 张，图片缓存位于：

```text
data/qwen_server_agent/web_images/
```

`WebSearchResult.image_urls` 保存来源图片 URL，`WebSearchResult.local_image_paths` 保存已验证的本地图片。当 server model 提供 `generate_multimodal()` 时，第二层会把这些图片和网页文本一起传入 Qwen2.5-VL；没有图片或模型不支持图文输入时，自动回退到纯文本整合。

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
- `LocalQwenEngine._load()`：懒加载 `AutoProcessor` 和 `Qwen2_5_VLForConditionalGeneration`
- `LocalQwenEngine._load_adapter_model()`：用 PEFT 加载 agent adapter，并根据文件指纹刷新缓存
- `LocalQwenEngine.unload()`：释放模型和 CUDA 缓存
- `LocalQwenEvolutionBackend.solve()`：把任务、私有样本和经验库组织成 prompt，调用本地 Qwen 解题
- `LocalQwenEvolutionBackend.suggest_improvements()`：让本地 Qwen 给出 3 到 5 条评价建议和分数
- `LocalQwenEvolutionBackend.release_model_memory()`：LoRA 训练前释放推理模型显存
- `LocalQwenAgentModel.generate()`：投票评测时每个智能体生成自己的答案，优先使用该智能体的 LoRA adapter

LoRA adapter 选择顺序：

1. 如果 `data/lora_adapters_qwen2_5_vl_7b/<agent>/adapter` 可用，使用该智能体自己的 LoRA adapter
2. 如果不存在，就使用原始 base model

## LoRA 自进化逻辑

LoRA 逻辑在 [b_magent/lora.py](/home/cxh/b_magent/b_magent/lora.py)。

默认输出：

```text
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/current_sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/lora_state.json
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/adapter/
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
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --dataset-dir data/gsm8k \
  --test-limit 100 \
  --lora-output-dir data/lora_adapters_qwen2_5_vl_7b \
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
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/current_sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/lora_state.json
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/adapter/
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
python -m train.four_agent_private_train --mode b-magent --backend local-qwen --model-path models/Qwen2.5-VL-7B-Instruct --dataset-dir data/gsm8k --rounds 200
```

4. 最后用训练后的 4 个智能体投票评测：

```bash
python -m train.four_agent_private_train --mode local-qwen-vote --model-path models/Qwen2.5-VL-7B-Instruct --dataset-dir data/gsm8k --test-limit 100 --lora-output-dir data/lora_adapters_qwen2_5_vl_7b --output train/four_agent_trained_voting_100_report.json
```
