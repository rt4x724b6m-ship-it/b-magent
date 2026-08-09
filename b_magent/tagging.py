from __future__ import annotations

import re


ROUTING_TAGS = (
    # ── visual ──────────────────────────────────────────────────────
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
    # ── math ────────────────────────────────────────────────────────
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
    # ── travel planning ─────────────────────────────────────────────
    "itinerary-planning",      # day-by-day schedule construction
    "transportation",          # flight / train / taxi selection
    "accommodation",           # hotel / Airbnb selection
    "attraction",              # sightseeing / POI selection
    "restaurant",              # dining selection
    "budget-constraint",       # total cost vs budget
    "local-constraint",        # cuisine / room-type / house-rule constraints
    "multi-city",              # visits multiple cities
    "multi-day",               # trip spans several days
    "route-optimization",      # ordering cities / legs efficiently
    "feasibility-check",       # verifying schedule is physically possible
    "commonsense-travel",      # general travel knowledge
)

ROUTING_TAG_IMPORTANCE = {
    # visual
    "ocr": 1.25,
    "visual-grounding": 1.25,
    "fine-grained-detail": 1.15,
    "arithmetic": 0.25,
    "final-answer": 0.10,
    "verification": 0.15,
    "boundary": 0.50,
    "structure": 0.10,
    "scoring": 0.10,
    # travel — core planning steps matter most
    "itinerary-planning": 1.30,
    "budget-constraint": 1.20,
    "local-constraint": 1.20,
    "feasibility-check": 1.15,
    "route-optimization": 1.10,
    "transportation": 1.00,
    "accommodation": 1.00,
    "attraction": 0.90,
    "restaurant": 0.90,
    "multi-city": 1.05,
    "multi-day": 0.80,
    "commonsense-travel": 0.70,
}

_TAG_ALIASES = {
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
    # ── travel aliases ───────────────────────────────────────────────
    "itinerary-planning": ("itinerary", "plan a trip", "travel plan", "day-by-day", "schedule", "行程", "旅行计划", "日程", "规划"),
    "transportation": ("flight", "train", "taxi", "bus", "car", "drive", "departure", "arrival", "交通", "航班", "火车", "出租车", "驾车", "出发", "抵达"),
    "accommodation": ("hotel", "airbnb", "hostel", "accommodation", "stay", "lodging", "住宿", "酒店", "民宿", "入住"),
    "attraction": ("attraction", "museum", "park", "sightseeing", "landmark", "visit", "景点", "博物馆", "公园", "观光", "地标"),
    "restaurant": ("restaurant", "breakfast", "lunch", "dinner", "cuisine", "dining", "eat", "餐厅", "早餐", "午餐", "晚餐", "美食", "用餐"),
    "budget-constraint": ("budget", "cost", "expense", "affordable", "within", "预算", "花费", "费用", "经济"),
    "local-constraint": ("local constraint", "house rule", "room type", "cuisine preference", "transportation mode", "本地限制", "房型", "饮食偏好"),
    "multi-city": ("multiple cities", "visiting city", "cities", "多城市", "途经", "多个城市"),
    "multi-day": ("days", "spanning", "nights", "多天", "天数", "晚上"),
    "route-optimization": ("route", "order", "efficient", "optimize", "路线", "顺序", "优化"),
    "feasibility-check": ("feasible", "possible", "realistic", "connection", "overlap", "time conflict", "可行性", "冲突", "衔接"),
    "commonsense-travel": ("travel", "trip", "journey", "旅行", "旅程", "出行"),
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
