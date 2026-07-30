# GPT-5.6 Sol 图片元素识别

`gpt56_sol_image_elements.py` 读取 `data/mm-vet` 和 `data/infographicsvqa`
中的非空 JSONL 文件。输出记录完整保留原字段，只新增 `image_elements`，默认写到
`baseline/gpt56_sol_output`，不会覆盖原始数据。

先设置环境变量，不要把 API Key 写进代码或命令历史：

```bash
export OPENAI_API_KEY="替换成新 API Key"
```

运行全部现有数据：

```bash
python -m baseline.gpt56_sol_image_elements
```

先处理每个 split 的两条记录做连通性测试：

```bash
python -m baseline.gpt56_sol_image_elements --limit 2
```

若服务商使用兼容 OpenAI Responses API 的自定义地址：

```bash
export OPENAI_BASE_URL="https://服务商地址/v1"
python -m baseline.gpt56_sol_image_elements
```

脚本默认模型 ID 是 `gpt-5.6-sol`。输出支持断点续跑；同一个 split 中多条问题共用
同一图片时只请求一次。
