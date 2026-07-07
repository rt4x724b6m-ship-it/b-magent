from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from .models import Draft, PeerEvaluation, SelfImprovement


DEFAULT_LORA_THRESHOLD = 100


@dataclass(frozen=True)
class LoraTrainingConfig:
    base_model_path: str
    output_dir: Path
    threshold: int = DEFAULT_LORA_THRESHOLD
    require_correct_answer: bool = True
    min_evaluation_score: float = 0.6
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("LoRA threshold must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("LoRA max_seq_length must be positive")


@dataclass(frozen=True)
class LoraSFTExample:
    agent_name: str
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LoraUpdate:
    agent_name: str
    dataset_path: str
    adapter_path: str
    examples: int
    trained: bool
    reason: str = ""
    version: int = 0
    pending_examples: int = 0


@dataclass
class AgentLoraState:
    agent_name: str
    dataset_path: Path
    adapter_path: Path
    examples: int = 0
    pending_examples: int = 0
    trained_examples: int = 0
    version: int = 0
    example_hashes: list[str] = field(default_factory=list)
    updates: list[LoraUpdate] = field(default_factory=list)


class LoraTrainer(Protocol):
    def train(self, agent_name: str, dataset_path: Path, adapter_path: Path, config: LoraTrainingConfig) -> None:
        """Train or refresh one agent LoRA adapter from its SFT dataset."""


class PeftSFTLoraTrainer:
    """PEFT/Transformers SFT trainer that freezes the backbone and trains only LoRA."""

    def train(self, agent_name: str, dataset_path: Path, adapter_path: Path, config: LoraTrainingConfig) -> None:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "LoRA training requires peft plus torch, datasets, and transformers. "
                "Install peft to enable --enable-lora."
            ) from exc

        rows = _read_jsonl(dataset_path)
        if not rows:
            raise ValueError(f"LoRA dataset is empty: {dataset_path}")

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=True,
        )
        model.config.use_cache = False
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
        )
        model = get_peft_model(model, peft_config)

        dataset = Dataset.from_list(rows)

        def tokenize(row: dict[str, str]) -> dict[str, list[int]]:
            prompt = format_lora_prompt(
                instruction=row["instruction"],
                input_text=row["input"],
                output=row["output"],
            )
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=config.max_seq_length,
                padding=False,
            )
            tokenized["labels"] = list(tokenized["input_ids"])
            return tokenized

        tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
        training_args = TrainingArguments(
            output_dir=str(adapter_path / "trainer_state"),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train()
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)


class LoraEvolutionManager:
    def __init__(self, config: LoraTrainingConfig, trainer: LoraTrainer | None = None) -> None:
        self.config = config
        self.trainer = trainer or PeftSFTLoraTrainer()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def update_from_round(
        self,
        task: str,
        drafts: list[Draft],
        peer_reviews: list[PeerEvaluation],
        self_improvements: list[SelfImprovement],
    ) -> list[LoraUpdate]:
        updates: list[LoraUpdate] = []
        improvements_by_agent = {item.agent_name: item for item in self_improvements}
        reviews_by_target: dict[str, list[PeerEvaluation]] = {}
        for review in peer_reviews:
            reviews_by_target.setdefault(review.target, []).append(review)

        for draft in drafts:
            improvement = improvements_by_agent.get(draft.agent_name)
            if improvement is None:
                continue
            example = build_lora_example(task, draft, reviews_by_target.get(draft.agent_name, []), improvement)
            accepted, reason = self.add_example_if_usable(
                task,
                draft,
                reviews_by_target.get(draft.agent_name, []),
                improvement,
                example,
            )
            if not accepted:
                updates.append(self.skipped_update(draft.agent_name, reason))
                continue
            updates.append(self.maybe_train_agent(draft.agent_name))
        return updates

    def add_example_if_usable(
        self,
        task: str,
        draft: Draft,
        evaluations: list[PeerEvaluation],
        improvement: SelfImprovement,
        example: LoraSFTExample,
    ) -> tuple[bool, str]:
        state = self.load_state(draft.agent_name)
        if not evaluations:
            return False, "no evaluator report"
        low_scores = [evaluation for evaluation in evaluations if not evaluation.scores.is_usable_for_lora(self.config.min_evaluation_score)]
        if low_scores:
            return False, "evaluation scores below LoRA quality threshold"
        correct = is_improved_answer_correct(task, improvement.revised_answer)
        improvement.is_correct = correct
        if self.config.require_correct_answer and correct is False:
            return False, "improved answer failed gold-answer correctness gate"
        digest = hash_lora_example(example)
        if digest in state.example_hashes:
            return False, "duplicate SFT example"
        append_lora_example(self.dataset_path(draft.agent_name), example)
        state.examples += 1
        state.pending_examples += 1
        state.example_hashes.append(digest)
        self.save_state(state)
        return True, "accepted"

    def maybe_train_agent(self, agent_name: str) -> LoraUpdate:
        dataset_path = self.dataset_path(agent_name)
        adapter_path = self.adapter_path(agent_name)
        state = self.load_state(agent_name)
        examples = count_jsonl_rows(dataset_path)
        state.examples = examples
        if state.pending_examples < self.config.threshold:
            self.save_state(state)
            return LoraUpdate(
                agent_name=agent_name,
                dataset_path=str(dataset_path),
                adapter_path=str(adapter_path),
                examples=examples,
                trained=False,
                reason=f"pending dataset below threshold: {state.pending_examples}/{self.config.threshold}",
                version=state.version,
                pending_examples=state.pending_examples,
            )

        self.trainer.train(agent_name, dataset_path, adapter_path, self.config)
        state.version += 1
        state.trained_examples = examples
        state.pending_examples = 0
        self.save_state(state)
        update = LoraUpdate(
            agent_name=agent_name,
            dataset_path=str(dataset_path),
            adapter_path=str(adapter_path),
            examples=examples,
            trained=True,
            version=state.version,
            pending_examples=state.pending_examples,
        )
        write_lora_metadata(adapter_path, update, self.config)
        return update

    def skipped_update(self, agent_name: str, reason: str) -> LoraUpdate:
        state = self.load_state(agent_name)
        return LoraUpdate(
            agent_name=agent_name,
            dataset_path=str(self.dataset_path(agent_name)),
            adapter_path=str(self.adapter_path(agent_name)),
            examples=state.examples,
            trained=False,
            reason=reason,
            version=state.version,
            pending_examples=state.pending_examples,
        )

    def dataset_path(self, agent_name: str) -> Path:
        return self.config.output_dir / agent_name / "sft_dataset.jsonl"

    def adapter_path(self, agent_name: str) -> Path:
        return self.config.output_dir / agent_name / "adapter"

    def state_path(self, agent_name: str) -> Path:
        return self.config.output_dir / agent_name / "lora_state.json"

    def load_state(self, agent_name: str) -> AgentLoraState:
        dataset_path = self.dataset_path(agent_name)
        adapter_path = self.adapter_path(agent_name)
        state_path = self.state_path(agent_name)
        if not state_path.exists():
            return AgentLoraState(
                agent_name=agent_name,
                dataset_path=dataset_path,
                adapter_path=adapter_path,
                examples=count_jsonl_rows(dataset_path),
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return AgentLoraState(
            agent_name=agent_name,
            dataset_path=dataset_path,
            adapter_path=adapter_path,
            examples=int(payload.get("examples", 0)),
            pending_examples=int(payload.get("pending_examples", 0)),
            trained_examples=int(payload.get("trained_examples", 0)),
            version=int(payload.get("version", 0)),
            example_hashes=[str(item) for item in payload.get("example_hashes", [])],
        )

    def save_state(self, state: AgentLoraState) -> None:
        path = self.state_path(state.agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_name": state.agent_name,
            "dataset_path": str(state.dataset_path),
            "adapter_path": str(state.adapter_path),
            "examples": state.examples,
            "pending_examples": state.pending_examples,
            "trained_examples": state.trained_examples,
            "version": state.version,
            "example_hashes": state.example_hashes,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_lora_example(
    task: str,
    draft: Draft,
    evaluations: list[PeerEvaluation],
    improvement: SelfImprovement,
) -> LoraSFTExample:
    evaluation_report = format_evaluation_report(evaluations)
    trajectory = format_trajectory(draft)
    return LoraSFTExample(
        agent_name=draft.agent_name,
        instruction="Solve the task, reflect on evaluator feedback, and produce the improved final answer.",
        input=(
            f"Task:\n{task}\n\n"
            f"Trajectory:\n{trajectory}\n\n"
            f"Evaluation Report:\n{evaluation_report}"
        ),
        output=improvement.revised_answer,
    )


def format_trajectory(draft: Draft) -> str:
    return (
        f"Agent: {draft.agent_name}\n"
        f"Specialty: {draft.specialty}\n"
        f"Thought Trace:\n{_format_list(draft.thought_trace)}\n"
        f"Tool Calls:\n{_format_list(draft.tool_calls)}\n"
        f"Answer:\n{draft.answer}"
    )


def format_evaluation_report(evaluations: list[PeerEvaluation]) -> str:
    if not evaluations:
        return "No evaluator feedback."
    chunks = []
    for evaluation in evaluations:
        chunks.append(
            f"Evaluator: {evaluation.evaluator}\n"
            f"Scores: correctness={evaluation.scores.correctness:.2f}, "
            f"safety={evaluation.scores.safety:.2f}, efficiency={evaluation.scores.efficiency:.2f}\n"
            f"Suggestions:\n{_format_list(evaluation.suggestions)}\n"
            f"Rationale:\n{evaluation.rationale}"
        )
    return "\n\n".join(chunks)


def format_lora_prompt(instruction: str, input_text: str, output: str) -> str:
    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Input\n"
        f"{input_text}\n\n"
        "### Output\n"
        f"{output}"
    )


def append_lora_example(dataset_path: Path, example: LoraSFTExample) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_lora_metadata(adapter_path: Path, update: LoraUpdate, config: LoraTrainingConfig) -> None:
    adapter_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": asdict(update),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "target_modules": list(config.target_modules),
        },
        "objective": "SFT cross-entropy on reflection-improved answers; frozen backbone plus trainable LoRA adapter.",
        "formula": {
            "adapter": "W = W0 + BA",
            "loss": "L_lora = -sum_t log P_{theta + delta_theta}(y*_t | x, y*_<t)",
        },
    }
    (adapter_path / "b_magent_lora_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def hash_lora_example(example: LoraSFTExample) -> str:
    payload = json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def is_improved_answer_correct(task: str, improved_answer: str) -> bool | None:
    gold = extract_gold_final_answer(task)
    if gold is None:
        return None
    predicted = extract_final_answer(improved_answer)
    return predicted == gold


def extract_gold_final_answer(task: str) -> str | None:
    match = re.search(r"Gold final answer:\s*([^\n]+)", task)
    if not match:
        return None
    return normalize_answer(match.group(1))


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return normalize_answer(matches[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def normalize_answer(text: str) -> str:
    cleaned = str(text).strip().replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def _format_list(items: list[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)
