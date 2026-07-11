from __future__ import annotations

from .models import Draft, EvaluationScores, PeerEvaluation


class DemoQwenBackend:
    """Deterministic local backend; replace these methods with real Qwen calls later."""

    def solve(
        self,
        agent_name: str,
        specialty: str,
        task: str,
        private_training: list[str],
        professional_memory: list[str],
        evaluation_alerts: list[str],
    ) -> tuple[str, list[str]]:
        thought_trace = [
            f"理解任务: {task}",
            f"采用专业视角: {specialty}",
            f"吸收私有训练样本: {len(private_training)} 条",
            f"检索专业经验: {len(professional_memory)} 条",
            f"检查评价约束: {len(evaluation_alerts)} 条",
        ]
        answer = (
            f"[{agent_name}] 从“{specialty}”角度处理任务：{task}\n"
            f"1. 结合本地私有数据形成角色策略。\n"
            f"2. 参考专业库经验：{'; '.join(professional_memory) or '暂无'}。\n"
            f"3. 按评价库约束自查：{'; '.join(evaluation_alerts) or '暂无'}。\n"
            f"4. 输出可执行方案，并保留可被评价者修改的推理轨迹。"
        )
        return answer, thought_trace

    def suggest_improvements(
        self,
        evaluator_name: str,
        target_draft: Draft,
        task: str,
        evaluation_memory: list[str],
    ) -> PeerEvaluation:
        suggestions = [
            "补充输入、输出和边界条件",
            "把关键步骤改成可复查的编号清单",
            "明确哪些结论来自私有训练数据，哪些来自专业库经验",
        ]
        if target_draft.thought_trace:
            suggestions.append("将思考轨迹中的假设转成可验证检查项")
        rationale = (
            f"{evaluator_name} 根据任务、联邦抽象答案摘要和脱敏轨迹给出可修改建议；"
            f"参考评价库: {'; '.join(evaluation_memory) or '暂无'}。"
        )
        return PeerEvaluation(
            evaluator=evaluator_name,
            target=target_draft.agent_name,
            suggestions=suggestions,
            rationale=rationale,
            evaluation_memory_used=evaluation_memory,
            scores=EvaluationScores(correctness=0.8, safety=1.0, efficiency=0.8),
        )

    def improve_answer(
        self,
        agent_name: str,
        specialty: str,
        task: str,
        draft: Draft,
        suggestions: list[str],
        professional_memory: list[str],
        evaluation_alerts: list[str],
    ) -> tuple[str, str]:
        applied = "\n".join(f"- {item}" for item in suggestions) if suggestions else "- 保持原答案并补充自检"
        revised_answer = (
            f"[{agent_name}] 改进后的最终答案\n"
            f"任务: {task}\n"
            "1. 重新检查输入、输出和边界条件。\n"
            "2. 将原始答案中的关键判断整理为可复查步骤。\n"
            f"3. 已吸收评价建议:\n{applied}\n"
            f"4. 参考专业经验: {'; '.join(professional_memory) or '暂无'}。\n"
            f"5. 参考评价约束: {'; '.join(evaluation_alerts) or '暂无'}。\n"
            "最终结论:\n"
            f"{draft.answer}"
        )
        reflection = (
            "Reflection: regenerated the answer from the task, original draft, evaluator feedback, "
            "professional memory, and evaluation checks."
        )
        return revised_answer, reflection
