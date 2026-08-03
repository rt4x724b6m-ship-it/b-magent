from __future__ import annotations


AGENT_SPECIALTIES: dict[str, str] = {
    "qwen_agent_1": "OCR文字识别专家：优先逐区域读取标题、标签、图例和小字号文本，并核对易混字符",
    "qwen_agent_2": "图表数值专家：优先定位坐标轴、单位、图例和数据标记，执行计数、比较与数值计算",
    "qwen_agent_3": "视觉布局专家：优先分析颜色、形状、空间位置、连接关系和版面层级",
    "qwen_agent_4": "语义验证专家：优先理解问题约束，交叉核验证据、单位和候选答案，排除歧义",
}


def agent_specialty(agent_name: str) -> str:
    return AGENT_SPECIALTIES.get(agent_name, "通用视觉问答专家")

