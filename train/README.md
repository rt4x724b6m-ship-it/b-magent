# 六智能体训练入口

正式训练入口是 `train/six_agent_training.py`。系统固定创建六个视觉智能体，每轮选择三个参与解题，另外三个负责互评。

## InfographicVQA 训练

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --dataset-dir data/infographicsvqa \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 200 \
  --output train/six_agent_training_report.json
```

续训必须使用 `--resume`：

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend local-qwen \
  --dataset-dir data/infographicsvqa \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --rounds 400 \
  --resume
```

## 离线逻辑测试

```bash
python -m train.six_agent_training \
  --mode b-magent \
  --backend demo \
  --dataset-dir data/infographicsvqa \
  --rounds 1 \
  --disable-lora \
  --answer-validator local
```

`four_agent_private_train.py` 仅作为旧命令兼容模块保留。它与正式入口加载同一套六智能体实现，不存在四智能体运行模式。
