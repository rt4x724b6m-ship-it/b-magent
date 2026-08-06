from __future__ import annotations

import json
import gc
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol

from .models import Draft, LibraryRecord, PeerEvaluation, SelfImprovement
from .retrieval_training import (
    build_verified_retrieval_output,
    is_retrieval_summary_grounded,
    strip_hidden_retrieval_labels,
)


DEFAULT_LORA_THRESHOLD = 10


@dataclass(frozen=True)
class LoraTrainingConfig:
    base_model_path: str
    output_dir: Path
    threshold: int = DEFAULT_LORA_THRESHOLD
    require_correct_answer: bool = True
    min_evaluation_score: float = 0.6
    min_training_examples: int = 1
    professional_library_dir: Path | None = None
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
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
        if self.per_device_train_batch_size <= 0:
            raise ValueError("LoRA train batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("LoRA gradient accumulation steps must be positive")
        if not 0.0 <= self.min_evaluation_score <= 1.0:
            raise ValueError("LoRA evaluation threshold must be between 0 and 1")
        if self.min_training_examples <= 0:
            raise ValueError("LoRA minimum training examples must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be between 0 (inclusive) and 1")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("LoRA warmup ratio must be between 0 (inclusive) and 1")
        if self.weight_decay < 0.0:
            raise ValueError("LoRA weight decay must not be negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("LoRA max gradient norm must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("LoRA learning rate must be positive")
        if self.num_train_epochs <= 0.0:
            raise ValueError("LoRA training epochs must be positive")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")


@dataclass(frozen=True)
class LoraSFTExample:
    agent_name: str
    instruction: str
    input: str
    output: str
    image: str = ""

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
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LoRA training requires peft plus torch, datasets, and transformers. "
                "Install peft to enable --enable-lora."
            ) from exc

        rows = _read_jsonl(dataset_path)
        if not rows:
            raise ValueError(f"LoRA dataset is empty: {dataset_path}")

        processor = AutoProcessor.from_pretrained(config.base_model_path, local_files_only=True)
        tokenizer = configure_processor_tokenizer(processor)

        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
        if callable(set_matmul_precision):
            set_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model_path,
            torch_dtype=(
                torch.bfloat16
                if use_bf16
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            ),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.config.use_cache = False
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # PEFT freezes the input embeddings. Gradient checkpointing still
            # needs their outputs to require gradients so LoRA layers receive
            # gradients during the recomputed forward pass.
            model.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
        )
        model = get_peft_model(model, peft_config)

        visual_rows = [row for row in rows if str(row.get("image", "")).strip()]
        text_rows = [row for row in rows if not str(row.get("image", "")).strip()]
        if visual_rows and text_rows:
            raise ValueError("Do not mix visual and text-only rows in one LoRA dataset")
        is_visual = bool(visual_rows)
        if is_visual:
            tokenized_dataset = Dataset.from_list(visual_rows)
            data_collator = MultimodalSFTCollator(processor, config.max_seq_length)
        else:
            tokenized_dataset = Dataset.from_list(
                [tokenize_lora_row(tokenizer, row, config.max_seq_length) for row in text_rows]
            )
            from transformers import DataCollatorForSeq2Seq

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=-100,
            )
        training_args = TrainingArguments(
            output_dir=str(adapter_path / "trainer_state"),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            num_train_epochs=config.num_train_epochs,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=torch.cuda.is_available() and not use_bf16,
            bf16=use_bf16,
            tf32=torch.cuda.is_available(),
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            label_names=["labels"],
            remove_unused_columns=not is_visual,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        try:
            trainer.train()
            adapter_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_path)
            processor.save_pretrained(adapter_path)
        finally:
            del trainer
            del data_collator
            del tokenized_dataset
            del model
            del tokenizer
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LoraEvolutionManager:
    def __init__(
        self,
        config: LoraTrainingConfig,
        trainer: LoraTrainer | None = None,
        before_train: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.trainer = trainer or PeftSFTLoraTrainer()
        self.before_train = before_train
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
                update = self.skipped_update(draft.agent_name, reason)
                updates.append(update)
                self.record_lora_experience(
                    draft.agent_name,
                    reason,
                    update,
                    accepted=False,
                    source_task=task,
                    example_hash=hash_lora_example(example),
                )
                continue
            state = self.load_state(draft.agent_name)
            if state.pending_examples < self.config.threshold:
                update = self.skipped_update(
                        draft.agent_name,
                        f"pending examples below LoRA threshold: {state.pending_examples}/{self.config.threshold}",
                    )
                updates.append(update)
                self.record_lora_experience(
                    draft.agent_name,
                    reason,
                    update,
                    accepted=True,
                    source_task=task,
                    example_hash=hash_lora_example(example),
                )
                continue
            updates.append(self.train_agent_on_curated_dataset(draft.agent_name))
        return updates

    def flush_pending(self, agent_names: list[str] | tuple[str, ...]) -> list[LoraUpdate]:
        """Train final partial batches so accepted samples are not left unused."""
        updates = []
        for agent_name in agent_names:
            state = self.load_state(agent_name)
            if state.pending_examples > 0:
                updates.append(self.train_agent_on_curated_dataset(agent_name))
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
        has_verified_visual_label = bool(extract_task_image_path(task) and extract_gold_final_answer(task))
        low_scores = [
            evaluation
            for evaluation in evaluations
            if not evaluation.scores.is_usable_for_lora(self.config.min_evaluation_score)
        ]
        # For visual SFT the target is the dataset gold label, not the draft or
        # evaluator-authored answer. Subjective safety/efficiency scores must not
        # discard a verified image-answer pair.
        if low_scores and not has_verified_visual_label:
            return False, "evaluation scores below LoRA quality threshold"
        correct = improvement.is_correct
        if correct is None:
            correct = is_improved_answer_correct(task, improvement.revised_answer)
        improvement.is_correct = correct
        has_verified_retrieval_label = build_verified_retrieval_output(task) is not None
        # Visual SFT rows are built from the dataset gold answer below, not from
        # an unverified draft. This retains strict output correctness while still
        # allowing a specialist to learn from a task it initially answered wrong.
        if (
            self.config.require_correct_answer
            and correct is not True
            and not has_verified_retrieval_label
            and not has_verified_visual_label
        ):
            return False, "improved answer did not pass a verifiable correctness or grounding gate"
        digest = hash_lora_example(example)
        if digest in state.example_hashes:
            return False, "duplicate SFT example"
        append_lora_example(self.dataset_path(draft.agent_name), example)
        state.examples += 1
        state.pending_examples += 1
        state.example_hashes.append(digest)
        self.save_state(state)
        return True, "accepted"

    def train_agent_on_curated_dataset(self, agent_name: str) -> LoraUpdate:
        selected_dataset_path = self.dataset_path(agent_name)
        dataset_path = self.current_dataset_path(agent_name)
        adapter_path = self.adapter_path(agent_name)
        copy_lora_dataset(selected_dataset_path, dataset_path)
        state = self.load_state(agent_name)
        examples = count_jsonl_rows(dataset_path)

        if examples < self.config.min_training_examples:
            update = self.skipped_update(
                agent_name,
                f"curated examples below LoRA safety minimum: "
                f"{examples}/{self.config.min_training_examples}",
            )
            self.record_lora_experience(agent_name, update.reason, update, accepted=True)
            return update
        training_config = replace(
            self.config,
            num_train_epochs=safe_lora_epochs(examples, self.config.num_train_epochs),
        )
        if self.before_train is not None:
            self.before_train()
        self.trainer.train(agent_name, dataset_path, adapter_path, training_config)
        state.version += 1
        state.trained_examples = state.examples
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
        write_lora_metadata(adapter_path, update, training_config)
        self.record_lora_experience(agent_name, update.reason or "adapter trained", update, accepted=True)
        return update

    def record_lora_experience(
        self,
        agent_name: str,
        reason: str,
        update: LoraUpdate,
        *,
        accepted: bool,
        source_task: str = "",
        example_hash: str = "",
    ) -> None:
        if self.config.professional_library_dir is None:
            return
        from .library import EvolutionLibrary

        library = EvolutionLibrary(
            self.config.professional_library_dir / agent_name / "professional_library.jsonl",
            "professional",
        )
        status = "trained" if update.trained else ("accepted" if accepted else "rejected")
        library.add_record(
            LibraryRecord(
                agent_name=agent_name,
                library_type="professional",
                source_task=source_task or f"LoRA dataset: {update.dataset_path}",
                summary=f"LoRA lifecycle status={status}; {reason}",
                detail=(
                    f"status={status} | examples={update.examples} | "
                    f"pending_examples={update.pending_examples} | version={update.version} | "
                    f"adapter_path={update.adapter_path} | example_hash={example_hash or '(batch)'} | "
                    f"reason={reason}"
                ),
                tags=["lora-training-metadata", f"lora-{status}", "professional-storage-audit"],
            )
        )

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

    def current_dataset_path(self, agent_name: str) -> Path:
        return self.config.output_dir / agent_name / "current_sft_dataset.jsonl"

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
    verified_output = build_verified_retrieval_output(task)
    image_path = extract_task_image_path(task)
    if image_path:
        gold = extract_gold_final_answer(task) or ""
        answer = gold.split("|", 1)[0].strip()
        return LoraSFTExample(
            agent_name=draft.agent_name,
            instruction=(
                f"Act as {draft.specialty}. Apply that specialist evidence-inspection workflow to "
                "the supplied image. First identify the requested answer type, then inspect the complete "
                "image and localize the relevant region. Cross-check labels, legends, axes, units, and "
                "nearby text before answering. Do not infer values that are not visible. Preserve exact "
                "visible spelling and numeric formatting. Return only the shortest direct answer, without "
                "JSON, evidence, reasoning, or an answer label."
            ),
            input=strip_gold_annotations(task),
            output=answer,
            image=image_path,
        )
    return LoraSFTExample(
        agent_name=draft.agent_name,
        instruction=(
            f"Act as {draft.specialty}. Solve the task from the information supplied in the task. "
            "Preserve the user's target and hard constraints, use only stated evidence, verify the final "
            "result, and follow the requested answer format exactly."
        ),
        # Draft trajectories and evaluator reports are unavailable at inference.
        # Excluding them prevents the adapter from learning to depend on noisy,
        # training-only context rather than the user task and supplied evidence.
        input=f"Task:\n{strip_gold_annotations(task)}",
        output=verified_output or improvement.revised_answer,
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


def tokenize_lora_row(tokenizer: object, row: dict[str, str], max_length: int) -> dict[str, list[int]]:
    """Tokenize an SFT row while computing loss only on the desired response."""
    if max_length < 2:
        raise ValueError("LoRA max sequence length must be at least 2")

    prefix = format_lora_prompt(row["instruction"], row["input"], output="")
    prefix_ids = list(tokenizer(prefix, add_special_tokens=True, padding=False)["input_ids"])
    output_ids = list(tokenizer(row["output"], add_special_tokens=False, padding=False)["input_ids"])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not output_ids or output_ids[-1] != eos_token_id):
        output_ids.append(eos_token_id)

    # Long reflection trajectories must not push all target tokens past the
    # sequence boundary. Reserve half of the window for response supervision.
    max_output_length = max(1, max_length // 2)
    if len(output_ids) > max_output_length:
        output_ids = output_ids[:max_output_length]
    prefix_ids = prefix_ids[: max_length - len(output_ids)]
    input_ids = prefix_ids + output_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + list(output_ids),
    }


class MultimodalSFTCollator:
    """Build Qwen-VL image tensors while masking every token except the answer."""

    def __init__(self, processor: object, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        configure_multimodal_image_budget(processor, max_length)

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, object]:
        from PIL import Image

        full_texts: list[str] = []
        prefix_texts: list[str] = []
        images = []
        for row in rows:
            image_path = Path(row["image"])
            if not image_path.is_file():
                raise FileNotFoundError(f"LoRA training image not found: {image_path}")
            images.append(Image.open(image_path).convert("RGB"))
            user_content = [
                {"type": "image", "image": str(image_path.resolve())},
                {
                    "type": "text",
                    "text": format_lora_prompt(row["instruction"], row["input"], output=""),
                },
            ]
            prefix_messages = [{"role": "user", "content": user_content}]
            full_messages = [
                *prefix_messages,
                {"role": "assistant", "content": str(row["output"])},
            ]
            prefix_texts.append(
                self.processor.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )
        try:
            batch = self.processor(
                text=full_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prefixes = self.processor(
                text=prefix_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, prefix_length in enumerate(prefixes["attention_mask"].sum(dim=1).tolist()):
            labels[index, : int(prefix_length)] = -100
        batch["labels"] = labels
        return dict(batch)


def configure_multimodal_image_budget(processor: object, max_length: int) -> int:
    """Cap Qwen-VL image tokens so sequence truncation cannot split an image block."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return 0

    # Keep one quarter of the context (at least 256 tokens) for the chat
    # template, question, and supervised answer. Qwen-VL produces one visual
    # token per merge_size**2 image patches.
    text_budget = min(max(256, max_length // 4), max_length - 1)
    visual_token_budget = max(1, max_length - text_budget)
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    max_pixels = visual_token_budget * (patch_size * merge_size) ** 2

    current_max_pixels = getattr(image_processor, "max_pixels", None)
    if current_max_pixels is None or int(current_max_pixels) > max_pixels:
        image_processor.max_pixels = max_pixels
    current_min_pixels = getattr(image_processor, "min_pixels", None)
    if current_min_pixels is not None and int(current_min_pixels) > max_pixels:
        image_processor.min_pixels = max_pixels
    return max_pixels


def configure_processor_tokenizer(processor: object) -> object:
    """Return and configure the text tokenizer wrapped by a VL processor."""
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("LoRA tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def safe_lora_epochs(example_count: int, requested_epochs: float) -> float:
    """Cap epochs for tiny datasets, where repeated SFT quickly collapses generation."""
    if example_count <= 0:
        raise ValueError("LoRA example count must be positive")
    if example_count < 8:
        return min(requested_epochs, 3.0)
    if example_count < 32:
        return min(requested_epochs, 5.0)
    return min(requested_epochs, 10.0)


def extract_task_image_path(task: str) -> str:
    match = re.search(r"^Image:\s*(.+)$", task, re.MULTILINE)
    return match.group(1).strip() if match else ""


def append_lora_example(dataset_path: Path, example: LoraSFTExample) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def write_lora_example(dataset_path: Path, example: LoraSFTExample) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(example.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")


def copy_lora_dataset(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


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
            "professional_library_dir": (
                str(config.professional_library_dir)
                if config.professional_library_dir is not None
                else None
            ),
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
    retrieval_correct = is_retrieval_summary_grounded(task, improved_answer)
    if retrieval_correct is not None:
        return retrieval_correct
    gold = extract_gold_final_answer(task)
    if gold is None:
        return None
    if extract_task_image_path(task):
        predicted_text = _extract_visual_text_answer(improved_answer)
        if not predicted_text:
            return False
        return _normalize_text_answer(predicted_text) in {
            _normalize_text_answer(candidate)
            for candidate in gold.split("|")
        }
    predicted = extract_final_answer(improved_answer)
    if predicted and _numeric_equal(predicted, gold):
        return True
    text_match = re.search(r"(?:final\s+answer|answer)\s*:\s*([^\n]+)", improved_answer, re.I)
    if not text_match:
        return False
    predicted_text = _normalize_text_answer(text_match.group(1))
    return predicted_text in {
        _normalize_text_answer(candidate)
        for candidate in gold.split("|")
    }


def _extract_visual_text_answer(answer: str) -> str:
    text = str(answer).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        return str(payload.get("final_answer", "")).strip()

    plain_text = text.replace("**", "").replace("__", "").strip()
    labelled = re.search(r"(?:final\s+answer|answer)\s*:\s*([^\n]+)", plain_text, re.I)
    if labelled:
        return labelled.group(1).strip()
    if len(plain_text) <= 200 and len(plain_text.split()) <= 30 and "\n" not in plain_text:
        return plain_text
    return ""


def _normalize_text_answer(text: str) -> str:
    value = str(text).strip().casefold().strip(". ,;:\"'")
    return " ".join(value.split())


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


def _numeric_equal(left: str, right: str) -> bool:
    try:
        return abs(Decimal(left) - Decimal(right)) <= Decimal("1e-9")
    except (InvalidOperation, ValueError):
        return left == right


def strip_gold_annotations(task: str) -> str:
    task = strip_hidden_retrieval_labels(task)
    lines = []
    in_gold_reasoning = False
    for line in task.splitlines():
        if re.match(r"\s*Gold image elements:", line):
            continue
        if re.match(r"\s*Gold reasoning:", line):
            in_gold_reasoning = True
            continue
        if re.match(r"\s*Gold final answer:", line):
            in_gold_reasoning = False
            continue
        if in_gold_reasoning:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _format_list(items: list[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)
