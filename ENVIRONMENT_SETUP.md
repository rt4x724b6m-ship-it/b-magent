# 环境配置清单

这份文档把本项目运行所需的环境整理成可直接执行的配置清单。

## 1. 基础前提

- 操作系统：Linux
- Python：3.12.x
- CUDA：12.8
- PyTorch：2.8.0 + cu128
- 主要运行方式：本地离线模型，不自动下载模型

## 2. 推荐环境

优先使用这个 conda 环境文件：

```bash
conda env create -f environment-portable.yml
conda activate cxh
```

备用方案：

```bash
conda env create -f environment.yml
```

如果不走 conda，也可以用 pip：

```bash
pip install -r requirements.txt
```

## 3. 必需 Python 依赖

项目核心运行依赖主要是这些：

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `bitsandbytes`
- `sentencepiece`
- `tiktoken`
- `safetensors`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `pyarrow`
- `pytest`
- `protobuf`
- `huggingface_hub`

`requirements.txt` 里还包含了 Jupyter、可视化和开发调试相关包，属于完整环境的一部分。

## 4. 项目必需资源

### 本地模型

默认模型路径：

```text
models/Qwen2.5-1.5B-Instruct
```

至少应包含：

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

### GSM8K 数据

默认数据目录：

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
```

每行需要是 JSON，至少有：

- `question`
- `answer`

## 5. GPU 运行建议

该项目的本地 Qwen 路径和 LoRA 训练都默认按 GPU 场景设计。

建议显卡满足：

- 可用 CUDA
- 有足够显存加载 `Qwen2.5-1.5B-Instruct`
- 训练 LoRA 时显存更充裕更稳

如果只是跑逻辑 smoke test，可以用：

```bash
python -m train.four_agent_private_train --mode b-magent --backend demo --dataset-dir data/gsm8k --rounds 1 --disable-lora
```

## 6. 环境验证

先检查基础依赖和本地文件：

```bash
python scripts/check_setup.py
```

它会检查：

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- CUDA 是否可用
- 本地模型文件是否存在
- GSM8K 训练/测试集是否存在

## 7. 运行入口

### 普通 demo

```bash
python main.py --task "设计一个多智能体协作解决数学题的流程" --seed 1
```

### baseline

```bash
python -m baseline.qwen_gsm8k --dataset-dir data/gsm8k --split test --output baseline/qwen_gsm8k_report.json
```

### 四智能体训练

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend local-qwen \
  --model-path models/Qwen2.5-1.5B-Instruct \
  --dataset-dir data/gsm8k \
  --rounds 200 \
  --output train/b_magent_training_report.json
```

### 仅跑逻辑，不加载模型

```bash
python -m train.four_agent_private_train \
  --mode b-magent \
  --backend demo \
  --dataset-dir data/gsm8k \
  --rounds 1 \
  --disable-lora
```

## 8. 给后续配置的最小口令

以后你直接让我“配置这个项目环境”时，我会按这套顺序处理：

1. 创建或激活 conda 环境
2. 安装项目依赖
3. 放好本地模型
4. 放好 GSM8K 数据
5. 跑 `scripts/check_setup.py`
6. 再启动对应入口

