from __future__ import annotations

from .evaluation_format import format_confidence_from_scores, format_structured_evaluation
from .models import Draft, EvaluationEvolution, EvaluationScores, LibraryRecord, PeerEvaluation


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
        scores = EvaluationScores(correctness=0.8, safety=1.0, efficiency=0.8)
        rationale = format_structured_evaluation(
            task=task,
            observed_error=(
                "目标答案需要补充输入、输出、边界条件，并把脱敏轨迹中的假设转成可验证检查项。"
            ),
            evaluation_decision=(
                f"{evaluator_name} 建议修改 {target_draft.agent_name} 的答案结构和可复查性；"
                f"参考评价库: {'; '.join(evaluation_memory) or '暂无'}。"
            ),
            confidence=format_confidence_from_scores(scores),
            improvement_pattern=" ; ".join(suggestions),
        )
        return PeerEvaluation(
            evaluator=evaluator_name,
            target=target_draft.agent_name,
            suggestions=suggestions,
            rationale=rationale,
            evaluation_memory_used=evaluation_memory,
            scores=scores,
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

    def aggregate_global_experience(
        self,
        server_name: str,
        task: str,
        peer_reviews: list[PeerEvaluation],
        evaluation_evolutions: list[EvaluationEvolution],
        consensus_evaluation_records: list[LibraryRecord],
        prior_global_memory: list[str],
    ) -> str:
        suggestions = []
        for review in peer_reviews:
            for suggestion in review.suggestions:
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        top_suggestion = suggestions[0] if suggestions else "检查答案结构、结论一致性和可复查证据"
        confidence = (
            sum((review.scores.correctness + review.scores.safety + review.scores.efficiency) / 3 for review in peer_reviews)
            / len(peer_reviews)
            if peer_reviews
            else 0.0
        )
        return format_structured_evaluation(
            task=task,
            observed_error=(
                f"综合 {len(peer_reviews)} 条互评轨迹和 "
                f"{len(consensus_evaluation_records)} 条共识评价经验后，主要风险是评价检查点不统一。"
            ),
            evaluation_decision=(
                f"[{server_name}] 聚合为全局评价经验；后续评价需交叉比较评价理由、评分和目标智能体改进结果。"
            ),
            confidence=f"{confidence:.2f}; prior_global_memory={len(prior_global_memory)}",
            improvement_pattern=f"后续评价优先执行: {top_suggestion}",
        )
