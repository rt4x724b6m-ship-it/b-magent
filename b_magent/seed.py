from __future__ import annotations

import re

from .agent import QwenAgent
from .models import LibraryRecord


def seed_agent_libraries(agents: list[QwenAgent]) -> None:
    for agent in agents:
        _seed_professional(agent)
        _seed_evaluation(agent)


def _is_travel_agent(agents: list[QwenAgent]) -> bool:
    """Heuristic: if any agent's data dir contains TravelPlanner private data, treat as travel mode."""
    return False  # resolved at runtime via _seed_professional


def _seed_professional(agent: QwenAgent) -> None:
    if agent.professional_library.all_records():
        return
    # Detect task domain from private data if it exists
    private_file = agent.data_dir / agent.name / "private_data.jsonl"
    is_travel = False
    if private_file.exists():
        first_line = ""
        for line in private_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                first_line = line.strip()
                break
        import json as _json
        try:
            payload = _json.loads(first_line)
            if isinstance(payload, dict) and "question" in payload and "days" in payload:
                is_travel = True
        except Exception:
            pass

    if is_travel:
        summary = (
            f"{agent.specialty} 旅行规划任务: 先拆解出发地、目的地、天数、人数、预算、本地约束，"
            "再逐天规划交通→景点→餐饮→住宿，最后验证预算和约束满足度"
        )
        detail = "初始化专业能力库，为旅行规划任务提供基础框架：约束识别、逐日行程、可行性校验。"
        tags = [agent.specialty, "seed", "professional", "itinerary-planning",
                "budget-constraint", "feasibility-check", "commonsense-travel"]
    else:
        summary = f"{agent.specialty} 任务先拆目标、约束、步骤、风险"
        detail = "初始化专业能力库，给首次任务提供基础上下文。"
        tags = [agent.specialty, "seed", "professional"]

    agent.professional_library.add_record(
        LibraryRecord(
            agent_name=agent.name,
            library_type="professional",
            source_task="seed",
            summary=summary,
            detail=detail,
            tags=tags,
        )
    )


def _seed_evaluation(agent: QwenAgent) -> None:
    if agent.evaluation_library.all_records():
        return
    # Check domain from professional library tags
    pro_records = agent.professional_library.all_records()
    is_travel = any("itinerary-planning" in r.tags for r in pro_records)

    if is_travel:
        summary = "评价旅行规划时检查：约束覆盖、预算合规、交通衔接、景点合理性、餐饮安排、住宿安排"
        detail = (
            "初始化评价能力库：提醒评价者检查行程是否覆盖所有约束（budget/local_constraint/transportation），"
            "是否每天都有完整的交通→景点→餐饮→住宿安排，以及时间衔接是否可行。"
        )
        tags = [agent.specialty, "seed", "evaluation", "feasibility-check",
                "budget-constraint", "local-constraint", "itinerary-planning"]
    else:
        summary = "评价只提出可修改建议，不做分数判断"
        detail = "初始化评价能力库，提醒评价者基于问题、答案和思考轨迹给出改进建议。"
        tags = [agent.specialty, "seed", "evaluation"]

    agent.evaluation_library.add_record(
        LibraryRecord(
            agent_name=agent.name,
            library_type="evaluation",
            source_task="seed",
            summary=summary,
            detail=detail,
            tags=tags,
        )
    )

