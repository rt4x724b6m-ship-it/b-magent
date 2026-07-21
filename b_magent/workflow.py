from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .agent import QwenAgent
from .backend import DemoQwenBackend
from .models import Draft, EvolutionReport, PeerEvaluation
from s_server import ServerAgent
from s_server.server_agent import select_consensus_peer_reviews


def build_default_agents(base_dir: Path | None = None, backend: Any | None = None) -> list[QwenAgent]:
    root = base_dir or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    backend = backend or DemoQwenBackend()
    return [
        QwenAgent("qwen_agent_1", "通用智能体", data_dir, backend),
        QwenAgent("qwen_agent_2", "通用智能体", data_dir, backend),
        QwenAgent("qwen_agent_3", "通用智能体", data_dir, backend),
        QwenAgent("qwen_agent_4", "通用智能体", data_dir, backend),
    ]


def build_default_server_agent(base_dir: Path | None = None, backend: Any | None = None) -> ServerAgent:
    root = base_dir or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    return ServerAgent("qwen_server_agent", data_dir, backend or DemoQwenBackend())


class MultiAgentWorkflow:
    def __init__(
        self,
        agents: list[QwenAgent],
        server_agent: ServerAgent | None = None,
        random_seed: int | None = None,
        private_batch_size: int | None = None,
    ) -> None:
        if len(agents) != 4:
            raise ValueError("b_magent training requires exactly four agents")
        self.agents = agents
        backend = agents[0].backend if agents else DemoQwenBackend()
        data_dir = agents[0].data_dir if agents else Path(__file__).resolve().parent.parent / "data"
        self.server_agent = server_agent or ServerAgent("qwen_server_agent", data_dir, backend)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.private_batch_size = private_batch_size

    def run(self, task: str, participant_names: list[str] | None = None) -> EvolutionReport:
        participants = self._select_participants(participant_names)
        participant_names = {agent.name for agent in participants}
        evaluators = [agent for agent in self.agents if agent.name not in participant_names]

        drafts: list[Draft] = []
        for agent in participants:
            private_training = agent.train_private_data(task, batch_size=self.private_batch_size)
            drafts.append(agent.solve_task(task, private_training))

        peer_reviews: list[PeerEvaluation] = []
        for evaluator in evaluators:
            for draft in drafts:
                if evaluator.name == draft.agent_name:
                    continue
                peer_reviews.append(evaluator.evaluate_peer(task, draft))

        self_improvements = []
        for participant in participants:
            draft = next(item for item in drafts if item.agent_name == participant.name)
            reviews_for_agent = [review for review in peer_reviews if review.target == participant.name]
            self_improvements.append(participant.self_improve(task, draft, reviews_for_agent))

        consensus_reviews = select_consensus_peer_reviews(peer_reviews)
        evaluation_evolutions = [
            evaluator.evolve_evaluation_library(
                task,
                [review for review in consensus_reviews if review.evaluator == evaluator.name],
                consensus_reviews,
                self_improvements,
            )
            for evaluator in evaluators
        ]
        global_experience = self.server_agent.aggregate_evaluation_experience(
            task,
            peer_reviews,
            evaluation_evolutions,
            evaluators,
        )

        return EvolutionReport(
            task=task,
            participants=[agent.name for agent in participants],
            evaluators=[agent.name for agent in evaluators],
            drafts=drafts,
            peer_reviews=peer_reviews,
            self_improvements=self_improvements,
            evaluation_evolutions=evaluation_evolutions,
            global_experience=global_experience,
        )

    def _select_participants(self, participant_names: list[str] | None) -> list[QwenAgent]:
        if participant_names is None:
            return self._rng.sample(self.agents, 2)
        if len(participant_names) != 2:
            raise ValueError("each workflow round requires exactly two participant names")
        agents_by_name = {agent.name: agent for agent in self.agents}
        missing = [name for name in participant_names if name not in agents_by_name]
        if missing:
            raise ValueError(f"unknown participant agents: {', '.join(missing)}")
        if len(set(participant_names)) != 2:
            raise ValueError("participant names must be distinct")
        return [agents_by_name[name] for name in participant_names]

    def export_report(self, report: EvolutionReport, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
