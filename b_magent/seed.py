from __future__ import annotations

from .agent import QwenAgent
from .models import LibraryRecord


def seed_agent_libraries(agents: list[QwenAgent]) -> None:
    for agent in agents:
        if not agent.professional_library.all_records():
            agent.professional_library.add_record(
                LibraryRecord(
                    agent_name=agent.name,
                    library_type="professional",
                    source_task="seed",
                    summary=f"{agent.specialty} 任务先拆目标、约束、步骤、风险",
                    detail="初始化专业能力库，给首次任务提供基础上下文。",
                    tags=[agent.specialty, "seed", "professional"],
                )
            )
        if not agent.evaluation_library.all_records():
            agent.evaluation_library.add_record(
                LibraryRecord(
                    agent_name=agent.name,
                    library_type="evaluation",
                    source_task="seed",
                    summary="评价只提出可修改建议，不做分数判断",
                    detail="初始化评价能力库，提醒评价者基于问题、答案和思考轨迹给出改进建议。",
                    tags=[agent.specialty, "seed", "evaluation"],
                )
            )

