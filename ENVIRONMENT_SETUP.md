# 环境配置

## 基础环境

- Linux
- Python 3.12
- CUDA 12.8
- PyTorch 2.8.0 + cu128
- 本地 Qwen2.5-VL 模型

推荐使用项目环境文件：

```bash
cd /home/cxh/第二层-图片识别
conda env create -f environment-portable.yml
conda activate cxh
```

或者安装 pip 依赖：

```bash
pip install -r requirements.txt
```

## 必需资源

默认模型目录：

```text
models/Qwen2.5-VL-7B-Instruct
```

默认数据目录：

```text
data/infographicsvqa/train.jsonl
data/infographicsvqa/validation.jsonl
data/infographicsvqa/test.jsonl
data/infographicsvqa/images/
```

检查环境：

```bash
python scripts/check_setup.py
```

## 六智能体训练

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --dataset-dir data/infographicsvqa \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 200 \
  --output train/six_agent_training_report.json
```

系统固定使用 `qwen_agent_1` 到 `qwen_agent_6`。每轮三个智能体参与训练，另外三个智能体负责评价。

离线逻辑测试：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend demo \
  --dataset-dir data/infographicsvqa \
  --rounds 1 \
  --disable-lora \
  --answer-validator local
```

训练完成后运行验证：

```bash
python main.py \
  --dataset-dir data/infographicsvqa \
  --split validation \
  --limit 100 \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --output train/six_agent_lora_infographicsvqa_validation_report.json
```
