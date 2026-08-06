# 六智能体图片识别与自进化训练

本项目使用本地 Qwen2.5-VL 模型和 InfographicVQA 数据集，完成信息图图片问答、六智能体协作训练、经验库自进化、LoRA 增量训练以及验证集评测。

项目默认使用六个同构视觉智能体：

- `qwen_agent_1`
- `qwen_agent_2`
- `qwen_agent_3`
- `qwen_agent_4`
- `qwen_agent_5`
- `qwen_agent_6`

每轮训练随机选择三个智能体作为参与者，另外三个作为评价者。参与者读取各自的私有训练数据并识别图片，评价者对答案进行互评；参与者随后根据评价进行反思和改进，并将结果写入专业经验库。服务器智能体负责汇总评价经验和维护路由标签。

## 项目目录

```text
第二层-图片识别/
├── main.py                         # 训练后验证入口：选择三个智能体并聚合答案
├── train/six_agent_training.py       # 六智能体训练与兼容评测入口
├── b_magent/                       # 智能体、数据集、模型后端、经验库和 LoRA 逻辑
├── s_server/                       # 服务器智能体、关键信息和检索存储
├── baseline/                       # 单模型与视觉基线
├── scripts/                        # 环境检查和数据准备脚本
├── data/infographicsvqa/           # InfographicVQA 数据和图片
├── data/qwen_agent_*/              # 各智能体私有数据和经验库
├── data/qwen_server_agent/         # 服务器全局经验与路由标签
└── data/lora_adapters_qwen2_5_vl_7b/
                                       # 各智能体 LoRA 数据和 adapter
```

项目运行代码统一使用六个候选智能体。每轮训练选择三个参与者和三个评价者；验证时由服务器从六个候选智能体中选择三个识别图片。

## 环境

推荐创建可迁移的 conda 环境：

```bash
cd /home/cxh/第二层-图片识别
conda env create -f environment-portable.yml
conda activate cxh
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

检查环境和本地资源：

```bash
python scripts/check_setup.py
```

默认模型目录为：

```text
models/Qwen2.5-VL-7B-Instruct
```

项目按离线本地模型运行，不会自动下载模型。可使用 `--model-path` 指定其他本地模型目录。

## InfographicVQA 数据集

程序默认读取：

```text
data/infographicsvqa/train.jsonl
data/infographicsvqa/validation.jsonl
data/infographicsvqa/test.jsonl
data/infographicsvqa/images/
```

当前本地数据量：

| 划分 | 样本数 | 用途 |
| --- | ---: | --- |
| `train` | 23,946 | 六智能体训练和私有数据拆分 |
| `validation` | 2,801 | 本地有标签验证，计算准确率和 ANLS |
| `test` | 3,288 | 官方测试提交；官方答案不在本地公开 |

JSONL 样本的主要字段包括：

```json
{
  "dataset": "infographicsvqa",
  "id": "65718",
  "question_id": "65718",
  "question": "...",
  "answers": ["..."],
  "image": "images/train-20471.jpeg",
  "data_split": "train"
}
```

`image` 相对于 `data/infographicsvqa` 解析。样本还可能包含 `image_url`、`ocr`、`answer_type` 和 `operation_reasoning` 等官方字段。训练和验证必须同时保留 JSONL 文件及 `images/` 目录。

如果需要从官方原始文件重新生成本地数据，可运行：

```bash
python scripts/prepare_infographicvqa_official.py --help
```

## 六智能体训练

主要训练入口是：

```text
train/six_agent_training.py
```

默认参数已经指向 `data/infographicsvqa`，因此可直接运行：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 200 \
  --output train/b_magent_infographicsvqa_training_report.json
```

显式指定数据集的等价命令：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --dataset-dir data/infographicsvqa \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 200
```

当 `--rounds 0`（默认值）时，程序会根据平均拆分后的私有训练数据自动确定轮数。每个参与者每轮默认读取四条私有样本，可通过 `--private-batch-size` 调整。

### 重要：新训练与续训

不带 `--resume` 启动 `b-magent` 模式时，程序会清理上一轮生成的智能体经验库、私有数据、服务器训练状态和 LoRA adapter，然后重新训练。

继续已有训练必须添加：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --dataset-dir data/infographicsvqa \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 400 \
  --resume
```

`--rounds` 在续训时表示目标总轮数，不是额外增加的轮数。

### 离线逻辑测试

不加载本地模型、不执行 LoRA 的快速测试：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend demo \
  --dataset-dir data/infographicsvqa \
  --rounds 1 \
  --disable-lora \
  --answer-validator local \
  --output train/demo_visual_training_report.json
```

### 反思答案正确性判断

InfographicVQA 的反思答案始终与数据集样本中的 `answers` 对比，不调用外部 API。程序支持一个样本包含多个官方答案，并对大小写、首尾标点和多余空白进行归一化后匹配。

训练任务中的 `Gold final answer` 来自对应 JSONL 样本的 `answers` 字段。该标注不会传给智能体解题，只在反思完成后用于判断答案是否正确，并决定结果进入正确经验还是错误反思经验，以及是否可以进入 LoRA 精选数据。

`--answer-validator` 默认值为 `local`。保留的 `gpt-5.6-sol` 选项只用于兼容非视觉、开放式任务；即使显式选择该选项，InfographicVQA 视觉样本仍固定使用数据集答案判断，不会调用 API。

## LoRA 训练

LoRA 默认启用。程序从通过正确性和评价分数门槛的自反思结果构建各智能体独立的精选 SFT 数据集，并在达到刷新条件时训练 adapter。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--lora-output-dir` | `data/lora_adapters_qwen2_5_vl_7b` | LoRA 数据和 adapter 目录 |
| `--lora-threshold` | `50` | 每个智能体触发刷新所需的新接收样本数 |
| `--lora-min-training-examples` | `16` | 创建或刷新 adapter 的最少精选样本数 |
| `--lora-max-seq-length` | `4096` | 最大训练序列长度 |
| `--lora-train-batch-size` | `1` | 单设备 batch size |
| `--lora-gradient-accumulation-steps` | `1` | 梯度累积步数 |
| `--lora-epochs` | `1.0` | 每次刷新训练轮数 |
| `--lora-learning-rate` | `5e-5` | 学习率 |
| `--disable-lora` | - | 只运行经验库自进化，不训练 adapter |

默认情况下，有官方答案的样本只有在反思答案正确时才能进入 LoRA 监督数据。`--allow-uncorrect-lora-labels` 会关闭这一保护，仅建议用于实验。

## 训练后验证

根目录的 `main.py` 是当前推荐的 InfographicVQA 验证入口。它会：

1. 加载六个候选智能体的训练标签和专业经验。
2. 由调度模型为每张图片选择三个智能体。
3. 让三个智能体分别查看图片并回答问题。
4. 由调度模型聚合三个识别结果。
5. 在 `validation` 划分上计算准确率和官方风格 ANLS。

运行前需要确保各智能体 LoRA adapter、专业经验库以及服务器路由标签已经生成。

```bash
python main.py \
  --dataset-dir data/infographicsvqa \
  --split validation \
  --limit 100 \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --lora-output-dir data/lora_adapters_qwen2_5_vl_7b \
  --output train/six_agent_lora_infographicsvqa_validation_report.json
```

`main.py` 只允许本地有标签的 `validation` 划分。InfographicVQA 官方 `test` 答案由评测服务器保管，不能在本地计算准确率。

## 经验库与训练产物

每个智能体的本地数据：

```text
data/qwen_agent_*/private_data.jsonl
data/qwen_agent_*/professional_library.jsonl
data/qwen_agent_*/evaluation_library.jsonl
```

服务器智能体数据：

```text
data/qwen_server_agent/global_evaluation_library.jsonl
data/qwen_server_agent/agent_training_tags.jsonl
data/qwen_server_agent/key_information_store.jsonl
data/qwen_server_agent/web_search_store.jsonl
```

各智能体 LoRA 产物：

```text
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/current_sft_dataset.jsonl
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/lora_state.json
data/lora_adapters_qwen2_5_vl_7b/qwen_agent_*/adapter/
```

## 测试

运行完整单元测试：

```bash
pytest -q
```

仅运行视觉数据和主入口测试：

```bash
pytest -q tests/test_vision_datasets.py tests/test_main.py tests/test_training.py
```

## 推荐运行顺序

1. 使用 `python scripts/check_setup.py` 检查环境、模型和数据。
2. 使用 `demo` 后端运行一轮离线逻辑测试。
3. 使用本地 Qwen2.5-VL 启动六智能体训练。
4. 中断后使用 `--resume` 继续，避免清空已有训练成果。
5. adapter 和经验库准备完成后，使用 `python main.py --split validation` 进行验证。
