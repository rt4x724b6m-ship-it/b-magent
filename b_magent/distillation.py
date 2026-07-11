from __future__ import annotations

import json
import gc
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from .lora import LoraTrainingConfig, _read_jsonl, count_jsonl_rows, format_lora_prompt


DEFAULT_DISTILLATION_THRESHOLD = 2


@dataclass(frozen=True)
class DistillationConfig:
    base_model_path: str
    lora_output_dir: Path
    threshold: int = DEFAULT_DISTILLATION_THRESHOLD
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    temperature: float = 2.0
    kd_weight: float = 0.5
    sft_weight: float = 1.0
    min_teacher_score: float = 0.0
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
            raise ValueError("distillation threshold must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("distillation max_seq_length must be positive")
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        if self.kd_weight < 0 or self.sft_weight < 0:
            raise ValueError("distillation loss weights must be non-negative")

    @classmethod
    def from_lora_config(
        cls,
        lora_config: LoraTrainingConfig,
        threshold: int = DEFAULT_DISTILLATION_THRESHOLD,
        temperature: float = 2.0,
        kd_weight: float = 0.5,
        sft_weight: float = 1.0,
    ) -> "DistillationConfig":
        return cls(
            base_model_path=lora_config.base_model_path,
            lora_output_dir=lora_config.output_dir,
            threshold=threshold,
            max_seq_length=lora_config.max_seq_length,
            per_device_train_batch_size=lora_config.per_device_train_batch_size,
            gradient_accumulation_steps=lora_config.gradient_accumulation_steps,
            learning_rate=lora_config.learning_rate,
            num_train_epochs=lora_config.num_train_epochs,
            temperature=temperature,
            kd_weight=kd_weight,
            sft_weight=sft_weight,
            lora_r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
        )


@dataclass(frozen=True)
class TeacherAdapter:
    agent_name: str
    adapter_path: str
    weight: float
    examples: int
    version: int


@dataclass(frozen=True)
class DistillationUpdate:
    agent_name: str
    dataset_path: str
    adapter_path: str
    examples: int
    teachers: list[TeacherAdapter]
    trained: bool
    reason: str = ""
    version: int = 0
    pending_examples: int = 0


@dataclass
class DistillationState:
    dataset_path: Path
    adapter_path: Path
    examples: int = 0
    pending_examples: int = 0
    trained_examples: int = 0
    version: int = 0
    source_hashes: list[str] = field(default_factory=list)


class DistillationTrainer(Protocol):
    def train(
        self,
        agent_name: str,
        dataset_path: Path,
        adapter_path: Path,
        teachers: list[TeacherAdapter],
        config: DistillationConfig,
    ) -> None:
        """Train one agent's private distilled adapter from many-teacher KL targets."""


class PeftManyToManyDistillationTrainer:
    """PEFT trainer for private per-agent LoRA distillation with many-teacher KD loss."""

    def train(
        self,
        agent_name: str,
        dataset_path: Path,
        adapter_path: Path,
        teachers: list[TeacherAdapter],
        config: DistillationConfig,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from datasets import Dataset
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Distillation requires peft plus torch, datasets, and transformers. "
                "Install the project requirements to enable distillation training."
            ) from exc

        rows = _read_jsonl(dataset_path)
        if not rows:
            raise ValueError(f"distillation dataset is empty: {dataset_path}")
        if not teachers:
            raise ValueError("distillation requires at least one teacher adapter")

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        student = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
        )
        student.config.use_cache = False
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
        )
        student = get_peft_model(student, peft_config)

        def load_teacher_model(teacher: TeacherAdapter):  # type: ignore[no-untyped-def]
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(model, teacher.adapter_path)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            return model

        def release_teacher_model(model) -> None:  # type: ignore[no-untyped-def]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        def teacher_input_device(teacher_model) -> torch.device:  # type: ignore[no-untyped-def]
            embeddings = teacher_model.get_input_embeddings()
            if embeddings is not None:
                return embeddings.weight.device
            return next(teacher_model.parameters()).device

        class DistillationTrainerImpl(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[no-untyped-def]
                labels = inputs.get("labels")
                outputs = model(**inputs)
                sft_loss = outputs.loss
                with torch.no_grad():
                    teacher_probs = None
                    total_weight = 0.0
                    teacher_inputs = {
                        key: value
                        for key, value in inputs.items()
                        if key in {"input_ids", "attention_mask", "position_ids"}
                    }
                    for teacher in teachers:
                        teacher_model = load_teacher_model(teacher)
                        try:
                            device = teacher_input_device(teacher_model)
                            model_inputs = {key: value.to(device) for key, value in teacher_inputs.items()}
                            teacher_outputs = teacher_model(**model_inputs)
                            teacher_logits = teacher_outputs.logits.to(outputs.logits.device)
                            probs = F.softmax(teacher_logits / config.temperature, dim=-1) * float(teacher.weight)
                            teacher_probs = probs if teacher_probs is None else teacher_probs + probs
                            total_weight += float(teacher.weight)
                        finally:
                            release_teacher_model(teacher_model)
                            del teacher_model
                    teacher_probs = teacher_probs / max(total_weight, 1e-8)
                student_log_probs = F.log_softmax(outputs.logits / config.temperature, dim=-1)
                kd_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
                if labels is not None:
                    mask = labels.ne(-100).to(kd_per_token.dtype)
                    kd_loss = (kd_per_token * mask).sum() / mask.sum().clamp_min(1.0)
                else:
                    kd_loss = kd_per_token.mean()
                loss = (config.sft_weight * sft_loss) + (config.kd_weight * (config.temperature**2) * kd_loss)
                return (loss, outputs) if return_outputs else loss

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
            label_names=["labels"],
        )

        class TrainerAdamW(torch.optim.AdamW):
            def train(self) -> None:
                return None

            def eval(self) -> None:
                return None

        optimizer = TrainerAdamW(student.parameters(), lr=config.learning_rate)
        trainer = DistillationTrainerImpl(
            model=student,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )
        trainer.train()
        adapter_path.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        del trainer
        del optimizer
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DistillationManager:
    def __init__(self, config: DistillationConfig, trainer: DistillationTrainer | None = None) -> None:
        self.config = config
        self.trainer = trainer or PeftManyToManyDistillationTrainer()
        self.config.lora_output_dir.mkdir(parents=True, exist_ok=True)

    def update_from_lora_updates(self, lora_updates: object) -> list[DistillationUpdate]:
        teachers = self.collect_teachers()
        if len(teachers) < 2:
            return [self.skipped_update(agent_name, "need at least two trained agent LoRA teachers") for agent_name in self.agent_names()]

        agent_names = self.agent_names_with_new_lora_examples(lora_updates)
        if not agent_names:
            return []

        updates: list[DistillationUpdate] = []
        for agent_name in agent_names:
            added = self.refresh_agent_dataset(agent_name)
            if added == 0:
                updates.append(self.skipped_update(agent_name, "no new private SFT examples for distillation"))
                continue
            updates.append(self.maybe_train_agent_distillation(agent_name, teachers))
        return updates

    def agent_names_with_new_lora_examples(self, lora_updates: object) -> list[str]:
        if not isinstance(lora_updates, list):
            return self.agent_names()

        agent_names: set[str] = set()
        for update in lora_updates:
            agent_name = getattr(update, "agent_name", None)
            if not isinstance(agent_name, str) or not agent_name:
                continue
            trained = bool(getattr(update, "trained", False))
            reason = str(getattr(update, "reason", ""))
            if (
                trained
                or reason == "accepted into curated SFT dataset"
                or reason.startswith("pending dataset below threshold:")
                or "waiting for LoRA threshold" in reason
            ):
                agent_names.add(agent_name)
        return sorted(agent_names)

    def refresh_agent_dataset(self, agent_name: str) -> int:
        state = self.load_state(agent_name)
        added = 0
        source_dataset_path = self.config.lora_output_dir / agent_name / "sft_dataset.jsonl"
        if not source_dataset_path.exists():
            self.save_state(agent_name, state)
            return 0
        for row in _read_jsonl(source_dataset_path):
            digest = hash_distillation_source(source_dataset_path, row)
            if digest in state.source_hashes:
                continue
            append_distillation_row(self.dataset_path(agent_name), row)
            state.examples += 1
            state.pending_examples += 1
            state.source_hashes.append(digest)
            added += 1
        self.save_state(agent_name, state)
        return added

    def collect_teachers(self) -> list[TeacherAdapter]:
        teachers = []
        for state_path in sorted(self.config.lora_output_dir.glob("qwen_agent_*/lora_state.json")):
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            adapter_path = Path(str(payload.get("adapter_path", "")))
            if not adapter_path.exists():
                continue
            version = int(payload.get("version", 0))
            if version <= 0:
                continue
            agent_name = str(payload.get("agent_name") or state_path.parent.name)
            examples = int(payload.get("trained_examples") or payload.get("examples") or 0)
            teachers.append(
                TeacherAdapter(
                    agent_name=agent_name,
                    adapter_path=str(adapter_path),
                    weight=float(max(examples, 1)),
                    examples=examples,
                    version=version,
                )
            )
        return normalize_teacher_weights(teachers)

    def maybe_train_agent_distillation(self, agent_name: str, teachers: list[TeacherAdapter]) -> DistillationUpdate:
        state = self.load_state(agent_name)
        state.examples = count_jsonl_rows(self.dataset_path(agent_name))
        if state.pending_examples < self.config.threshold:
            self.save_state(agent_name, state)
            return DistillationUpdate(
                agent_name=agent_name,
                dataset_path=str(self.dataset_path(agent_name)),
                adapter_path=str(self.adapter_path(agent_name)),
                examples=state.examples,
                teachers=teachers,
                trained=False,
                reason=f"pending distillation dataset below threshold: {state.pending_examples}/{self.config.threshold}",
                version=state.version,
                pending_examples=state.pending_examples,
            )

        self.trainer.train(agent_name, self.dataset_path(agent_name), self.adapter_path(agent_name), teachers, self.config)
        state.version += 1
        state.trained_examples = state.examples
        state.pending_examples = 0
        self.save_state(agent_name, state)
        update = DistillationUpdate(
            agent_name=agent_name,
            dataset_path=str(self.dataset_path(agent_name)),
            adapter_path=str(self.adapter_path(agent_name)),
            examples=state.examples,
            teachers=teachers,
            trained=True,
            version=state.version,
            pending_examples=state.pending_examples,
        )
        write_distillation_metadata(self.adapter_path(agent_name), update, self.config)
        return update

    def skipped_update(self, agent_name: str, reason: str) -> DistillationUpdate:
        state = self.load_state(agent_name)
        return DistillationUpdate(
            agent_name=agent_name,
            dataset_path=str(self.dataset_path(agent_name)),
            adapter_path=str(self.adapter_path(agent_name)),
            examples=state.examples,
            teachers=self.collect_teachers(),
            trained=False,
            reason=reason,
            version=state.version,
            pending_examples=state.pending_examples,
        )

    def agent_names(self) -> list[str]:
        return sorted(path.name for path in self.config.lora_output_dir.glob("qwen_agent_*") if path.is_dir())

    def dataset_path(self, agent_name: str) -> Path:
        return self.config.lora_output_dir / agent_name / "distillation_dataset.jsonl"

    def adapter_path(self, agent_name: str) -> Path:
        return self.config.lora_output_dir / agent_name / "distilled_adapter"

    def state_path(self, agent_name: str) -> Path:
        return self.config.lora_output_dir / agent_name / "distillation_state.json"

    def load_state(self, agent_name: str) -> DistillationState:
        state_path = self.state_path(agent_name)
        if not state_path.exists():
            return DistillationState(
                dataset_path=self.dataset_path(agent_name),
                adapter_path=self.adapter_path(agent_name),
                examples=count_jsonl_rows(self.dataset_path(agent_name)),
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return DistillationState(
            dataset_path=self.dataset_path(agent_name),
            adapter_path=self.adapter_path(agent_name),
            examples=int(payload.get("examples", 0)),
            pending_examples=int(payload.get("pending_examples", 0)),
            trained_examples=int(payload.get("trained_examples", 0)),
            version=int(payload.get("version", 0)),
            source_hashes=[str(item) for item in payload.get("source_hashes", [])],
        )

    def save_state(self, agent_name: str, state: DistillationState) -> None:
        path = self.state_path(agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_path": str(state.dataset_path),
            "adapter_path": str(state.adapter_path),
            "examples": state.examples,
            "pending_examples": state.pending_examples,
            "trained_examples": state.trained_examples,
            "version": state.version,
            "source_hashes": state.source_hashes,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_teacher_weights(teachers: list[TeacherAdapter]) -> list[TeacherAdapter]:
    total = sum(max(teacher.weight, 0.0) for teacher in teachers)
    if total <= 0:
        total = float(len(teachers) or 1)
        return [
            TeacherAdapter(teacher.agent_name, teacher.adapter_path, 1.0 / total, teacher.examples, teacher.version)
            for teacher in teachers
        ]
    return [
        TeacherAdapter(
            teacher.agent_name,
            teacher.adapter_path,
            max(teacher.weight, 0.0) / total,
            teacher.examples,
            teacher.version,
        )
        for teacher in teachers
    ]


def append_distillation_row(dataset_path: Path, row: dict[str, object]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_name": str(row.get("agent_name", "")),
        "instruction": str(row.get("instruction", "")),
        "input": str(row.get("input", "")),
        "output": str(row.get("output", "")),
    }
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def hash_distillation_source(dataset_path: Path, row: dict[str, object]) -> str:
    payload = {
        "source": str(dataset_path),
        "row": row,
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_distillation_metadata(
    adapter_path: Path,
    update: DistillationUpdate,
    config: DistillationConfig,
) -> None:
    adapter_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": asdict(update),
        "config": {
            **asdict(config),
            "lora_output_dir": str(config.lora_output_dir),
            "target_modules": list(config.target_modules),
        },
        "objective": (
            "Many-to-many distillation into this agent's private distilled LoRA adapter. "
            "Other agent adapters are teachers only; no common adapter is stored."
        ),
        "formula": {
            "teacher_distribution": "p_T = sum_i alpha_i * p_i, sum_i alpha_i = 1",
            "kd_loss": "L_KD = T^2 * D_KL(p_T || p_S)",
            "total_loss": "L = sft_weight * L_SFT + kd_weight * L_KD",
        },
    }
    (adapter_path / "b_magent_distillation_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
