from __future__ import annotations

import re


ROUTING_TAGS = (
    "visual-content-completeness",
    "ocr-coverage",
    "object-coverage",
    "layout-coverage",
    "ocr",
    "chart-reading",
    "table-reading",
    "object-recognition",
    "attribute-recognition",
    "spatial-relation",
    "visual-counting",
    "fine-grained-detail",
    "scene-understanding",
    "visual-grounding",
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "fraction",
    "percentage",
    "ratio",
    "rate",
    "unit-conversion",
    "money",
    "time",
    "geometry",
    "counting",
    "multi-step",
    "arithmetic",
    "final-answer",
    "verification",
    "boundary",
    "structure",
    "scoring",
)

ROUTING_TAG_IMPORTANCE = {
    "visual-content-completeness": 2.0,
    "ocr-coverage": 1.6,
    "object-coverage": 1.5,
    "layout-coverage": 1.3,
    "ocr": 1.25,
    "visual-grounding": 1.25,
    "fine-grained-detail": 1.15,
    "arithmetic": 0.25,
    "final-answer": 0.10,
    "verification": 0.15,
    "boundary": 0.50,
    "structure": 0.10,
    "scoring": 0.10,
}

_TAG_ALIASES = {
    "visual-content-completeness": ("visual content completeness", "complete image content", "图像内容完整度", "完整识别"),
    "ocr-coverage": ("ocr coverage", "text coverage", "文字覆盖率", "文本完整度"),
    "object-coverage": ("object coverage", "object completeness", "物体覆盖率", "物体完整度"),
    "layout-coverage": ("layout coverage", "layout completeness", "布局覆盖率", "布局完整度"),
    "ocr": ("ocr", "read the text", "scene text", "written", "label", "文字识别", "读取文字", "文本"),
    "chart-reading": ("chart", "graph", "plot", "infographic", "图表", "信息图", "趋势图"),
    "table-reading": ("table", "row", "column", "表格", "行", "列"),
    "object-recognition": ("object", "item", "what is shown", "identify", "物体", "识别", "是什么"),
    "attribute-recognition": ("color", "shape", "size", "brand", "颜色", "形状", "大小", "品牌"),
    "spatial-relation": ("left of", "right of", "above", "below", "next to", "左边", "右边", "上方", "下方", "旁边"),
    "visual-counting": ("how many", "number of", "count", "多少个", "数量", "数一数"),
    "fine-grained-detail": ("detail", "small text", "fine-grained", "细节", "小字", "精细"),
    "scene-understanding": ("scene", "occasion", "happening", "场景", "发生了什么"),
    "visual-grounding": ("image", "picture", "visual", "visible", "图像", "图片", "视觉", "可见"),
    "addition": ("addition", "add", "sum", "total", "altogether", "combined", "in all", "加法", "总共", "合计"),
    "subtraction": ("subtraction", "subtract", "difference", "remaining", "remain", "left", "fewer", "减法", "剩余", "相差"),
    "multiplication": ("multiplication", "multiply", "product", "times as many", "each", "乘法", "乘以", "每个"),
    "division": ("division", "divide", "quotient", "split equally", "shared equally", "average", "除法", "平均"),
    "fraction": ("fraction", "half", "one-third", "one third", "quarter", "分数", "一半", "三分之一", "四分之一"),
    "percentage": ("percentage", "percent", "%", "百分比", "百分之"),
    "ratio": ("ratio", "proportion", "proportional", "as many", "比例", "比率", "倍数"),
    "rate": ("rate", "per hour", "per minute", "per day", "each hour", "each minute", "speed", "速率", "速度", "每小时", "每分钟"),
    "unit-conversion": ("unit conversion", "convert", "conversion", "inches", "feet", "yards", "meters", "kilometers", "grams", "kilograms", "单位换算", "转换为"),
    "money": ("money", "dollar", "cent", "cost", "price", "paid", "earns", "profit", "$", "金额", "美元", "价格", "花费", "收入", "利润"),
    "time": ("minute", "hour", "day", "week", "month", "year", "clock", "时间", "分钟", "小时", "天", "星期", "月份", "年份"),
    "geometry": ("geometry", "area", "perimeter", "rectangle", "square", "triangle", "circle", "length", "width", "几何", "面积", "周长", "长方形", "正方形", "三角形", "圆形"),
    "counting": ("counting", "how many ways", "arrangement", "combination", "permutation", "计数", "多少种", "排列", "组合"),
    "multi-step": ("multi-step", "multiple steps", "several steps", "多步骤", "多步"),
    "arithmetic": ("arithmetic", "numeric", "calculation", "math", "算术", "计算", "数字"),
    "final-answer": ("final-answer", "final answer", "####", "最终答案"),
    "verification": ("verification", "verify", "check", "验证", "检查", "自检"),
    "boundary": ("boundary", "edge case", "condition", "边界", "条件"),
    "structure": ("structure", "step", "checklist", "结构", "清单", "步骤", "编号"),
    "scoring": ("scoring", "score", "correctness", "safety", "efficiency", "评分", "正确性"),
}

_EQUATION_RE = re.compile(r"<<\s*(.+?)\s*=.+?>>")


def extract_math_task_tags(text: str) -> set[str]:
    """Extract stable visual, operation, and answer-quality tags."""
    lower = str(text).lower().replace("_", "-")
    tags = {
        tag
        for tag, aliases in _TAG_ALIASES.items()
        if any(_contains_alias(lower, alias) for alias in aliases)
    }

    equations = _EQUATION_RE.findall(lower)
    for expression in equations:
        if "+" in expression:
            tags.add("addition")
        if re.search(r"\d\s*-\s*\d", expression):
            tags.add("subtraction")
        if "*" in expression or "×" in expression:
            tags.add("multiplication")
        if "/" in expression or "÷" in expression:
            tags.add("division")
    if equations:
        tags.add("arithmetic")
    if len(equations) >= 2:
        tags.add("multi-step")
    if re.search(r"\d\s*\+\s*\d", lower):
        tags.update(("addition", "arithmetic"))
    if re.search(r"\d\s*-\s*\d", lower):
        tags.update(("subtraction", "arithmetic"))
    if re.search(r"\d\s*(?:\*|×)\s*\d", lower):
        tags.update(("multiplication", "arithmetic"))
    if re.search(r"\d\s*(?:/|÷)\s*\d", lower):
        tags.update(("division", "arithmetic"))
    return tags


def routing_tag_importance(tag: str) -> float:
    return ROUTING_TAG_IMPORTANCE.get(tag, 1.0)


def _contains_alias(text: str, alias: str) -> bool:
    if any(ord(character) > 127 for character in alias) or not alias.replace("-", "").replace(" ", "").isalnum():
        return alias in text
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None
