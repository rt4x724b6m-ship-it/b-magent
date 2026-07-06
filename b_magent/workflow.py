from __future__ import annotations

import json
import random
from pathlib import Path

from .agent import QwenAgent
from .backend import DemoQwenBackend
from .models import Draft, EvolutionReport, PeerReview


def build_default_agents(base_dir: Path | None = None) -> list[QwenAgent]:
    root = base_dir or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    backend = DemoQwenBackend()
    return [
        QwenAgent("qwen_planner", "方案规划", data_dir, backend),
        QwenAgent("qwen_executor", "执行落地", data_dir, backend),
        QwenAgent("qwen_reviewer", "质量评审", data_dir, backend),
    ]


class MultiAgentWorkflow:
    def __init__(self, agents: list[QwenAgent], random_seed: int | None = None) -> None:
        if len(agents) < 2:
            raise ValueError("at least two agents are required")
        self.agents = agents
        self.random = random.Random(random_seed)

    def run(self, task: str) -> EvolutionReport:
        participants, evaluators = self._split_roles()

        drafts: list[Draft] = []
        for agent in participants:
            private_training = agent.train_private_data(task)
            drafts.append(agent.solve_task(task, private_training))

        peer_reviews: list[PeerReview] = []
        for evaluator in evaluators:
            for draft in drafts:
                peer_reviews.append(evaluator.review_peer(task, draft))

        self_improvements = []
        for participant in participants:
            draft = next(item for item in drafts if item.agent_name == participant.name)
            reviews_for_agent = [review for review in peer_reviews if review.target == participant.name]
            self_improvements.append(participant.self_improve(task, draft, reviews_for_agent))

        evaluation_evolutions = [
            evaluator.evolve_evaluation_library(task, peer_reviews)
            for evaluator in evaluators
        ]

        return EvolutionReport(
            task=task,
            participants=[agent.name for agent in participants],
            evaluators=[agent.name for agent in evaluators],
            drafts=drafts,
            peer_reviews=peer_reviews,
            self_improvements=self_improvements,
            evaluation_evolutions=evaluation_evolutions,
        )

    def export_report(self, report: EvolutionReport, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _split_roles(self) -> tuple[list[QwenAgent], list[QwenAgent]]:
        shuffled = list(self.agents)
        self.random.shuffle(shuffled)
        split_at = self.random.randint(1, len(shuffled) - 1)
        participants = shuffled[:split_at]
        evaluators = shuffled[split_at:]
        return participants, evaluators

