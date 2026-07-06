from __future__ import annotations

import argparse
from pathlib import Path

from b_magent.seed import seed_agent_libraries
from b_magent.workflow import MultiAgentWorkflow, build_default_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three-agent self-evolution demo.")
    parser.add_argument("--task", default="设计一个三智能体自我进化流程")
    parser.add_argument("--output", type=Path, default=Path("data/latest_report.json"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = build_default_agents(Path(__file__).resolve().parent)
    seed_agent_libraries(agents)
    workflow = MultiAgentWorkflow(agents, random_seed=args.seed)
    report = workflow.run(args.task)
    workflow.export_report(report, args.output)
    print(f"participants: {', '.join(report.participants)}")
    print(f"evaluators: {', '.join(report.evaluators)}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
