from __future__ import annotations

import gc
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation_format import format_confidence_from_scores, format_structured_evaluation
from .models import Draft, EvaluationEvolution, EvaluationScores, LibraryRecord, PeerEvaluation
from .self_evolution import normalize_experience_tags


DEFAULT_QWEN_MODEL = "models/Qwen2.5-1.5B-Instruct"

NUMERIC_ANSWER_INSTRUCTION = (
    "You are a careful math reasoning assistant. Solve the problem step by step. "
    "End your response with a final line exactly in this format: #### <numeric_answer>. "
    "Do not include units, explanations, or full sentences after ####."
)


@dataclass(frozen=True)
class QwenGenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


class LocalQwenEngine:
    """Lazy Transformers loader for a local Qwen2.5 model."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_QWEN_MODEL,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        generation_config: QwenGenerationConfig | None = None,
        local_files_only: bool = True,
        system_prompt: str | None = NUMERIC_ANSWER_INSTRUCTION,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.generation_config = generation_config or QwenGenerationConfig()
        self.local_files_only = local_files_only
        self.system_prompt = system_prompt
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._adapter_models: dict[tuple[str, int], Any] = {}
        self._load_lock = threading.Lock()
        self._adapter_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self._load()
        return self._model

    def generate(self, prompt: str, adapter_path: str | Path | None = None) -> str:
        with self._generation_lock:
            return self._generate_unlocked(prompt, adapter_path=adapter_path)

    def _generate_unlocked(self, prompt: str, adapter_path: str | Path | None = None) -> str:
        self._load()
        model = self._model
        if adapter_path is not None and _is_lora_adapter_ready(Path(adapter_path)):
            model = self._load_adapter_model(Path(adapter_path))
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": self.generation_config.max_new_tokens,
            "do_sample": self.generation_config.do_sample,
        }
        if self.generation_config.do_sample:
            generation_kwargs["temperature"] = self.generation_config.temperature
            generation_kwargs["top_p"] = self.generation_config.top_p
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
        completion_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()

    def _load_adapter_model(self, adapter_path: Path) -> Any:
        resolved_path = str(adapter_path.resolve())
        key = (resolved_path, _adapter_fingerprint(adapter_path))
        with self._adapter_lock:
            if key in self._adapter_models:
                return self._adapter_models[key]
            self._adapter_models = {
                cached_key: cached_model
                for cached_key, cached_model in self._adapter_models.items()
                if cached_key[0] != resolved_path
            }
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading LoRA adapters requires peft.") from exc
            adapter_model = PeftModel.from_pretrained(self._model, resolved_path)
            model_device = getattr(self._model, "device", None)
            if model_device is not None:
                adapter_model = adapter_model.to(model_device)
            self._adapter_models[key] = adapter_model
            return adapter_model

    def unload(self) -> None:
        self._adapter_models.clear()
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_torch_dtype(self) -> Any:
        if self.torch_dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Local Qwen requires torch.") from exc
        return getattr(torch, self.torch_dtype)

    def _resolve_device_map(self) -> Any:
        if self.device_map != "auto":
            return self.device_map
        return None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        model_path = Path(self.model_name_or_path)
        if self.local_files_only and not model_path.exists():
            raise RuntimeError(
                "Local Qwen model directory was not found. "
                f"Expected: {model_path.resolve()}. "
                "Pass --model-path /path/to/Qwen2.5-1.5B-Instruct or place the model under models/Qwen2.5-1.5B-Instruct."
            )
        try:
            import transformers

            AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM", None)
            AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
            if AutoModelForCausalLM is None:
                from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            if AutoTokenizer is None:
                from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen requires transformers. Install transformers and torch, "
                "then pass a local Qwen2.5-1.5B model path or use the default Hugging Face id."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._resolve_torch_dtype(),
            device_map=self._resolve_device_map(),
            low_cpu_mem_usage=self.device_map != "auto",
            local_files_only=self.local_files_only,
        )
        if self.device_map == "auto":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Local Qwen requires torch.") from exc
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        tie_weights = getattr(self._model, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()
        if not self.generation_config.do_sample:
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None


class LocalQwenAgentModel:
    """Qwen2.5-backed agent model implementing the train/vote interfaces."""

    def __init__(
        self,
        agent_name: str,
        engine: LocalQwenEngine,
        lora_output_dir: str | Path | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.engine = engine
        self.lora_output_dir = Path(lora_output_dir) if lora_output_dir is not None else None
        self.training_examples: list[str] = []

    def train_batch(self, batch: object) -> None:
        if not batch:
            return
        for sample in batch:
            to_training_text = getattr(sample, "to_training_text", None)
            if callable(to_training_text):
                self.training_examples.append(to_training_text())

    def generate(self, question: str) -> str:
        return self.generate_with_server_guidance(question, "")

    def generate_with_server_guidance(self, question: str, server_guidance: str) -> str:
        context = "\n".join(self.training_examples[-4:])
        prompt = (
            f"Agent: {self.agent_name}\n"
            f"Private examples:\n{context or '(none)'}\n\n"
            f"Question:\n{question}\n\n"
            "Server evaluation guidance:\n"
            f"{server_guidance or '(none)'}\n\n"
            "Solve independently while applying the server's risk checks. "
            "Do not treat the guidance as a proposed numeric answer.\n\n"
            f"Output constraint:\n{NUMERIC_ANSWER_INSTRUCTION}"
        )
        adapter_path = self._adapter_path()
        if adapter_path is None:
            return self.engine.generate(prompt)
        return self.engine.generate(prompt, adapter_path=adapter_path)

    def _adapter_path(self) -> Path | None:
        if self.lora_output_dir is None:
            return None
        adapter_path = self.lora_output_dir / self.agent_name / "adapter"
        return adapter_path if _is_lora_adapter_ready(adapter_path) else None


class LocalQwenEvolutionBackend:
    """Local-Qwen backend for the b_magent self-evolution workflow."""

    def __init__(
        self,
        engine: LocalQwenEngine,
        lora_output_dir: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.lora_output_dir = Path(lora_output_dir) if lora_output_dir is not None else None

    def solve(
        self,
        agent_name: str,
        specialty: str,
        task: str,
        private_training: list[str],
        professional_memory: list[str],
        evaluation_alerts: list[str],
    ) -> tuple[str, list[str]]:
        prompt = (
            f"Agent: {agent_name}\n"
            f"Agent type: {specialty}\n"
            "Task:\n"
            f"{task}\n\n"
            "Private training examples:\n"
            f"{_format_context(private_training)}\n\n"
            "Professional evolution library memories:\n"
            f"{_format_context(professional_memory)}\n\n"
            "Evaluation evolution library checks:\n"
            f"{_format_context(evaluation_alerts)}\n\n"
            "Produce a complete answer and include reusable lessons that this agent can absorb."
        )
        answer = self.engine.generate(prompt, adapter_path=self._adapter_path(agent_name))
        thought_trace = [
            f"local_qwen_agent={agent_name}",
            f"private_training={len(private_training)}",
            f"professional_memory={len(professional_memory)}",
            f"evaluation_alerts={len(evaluation_alerts)}",
        ]
        return answer, thought_trace

    def suggest_improvements(
        self,
        evaluator_name: str,
        target_draft: Draft,
        task: str,
        evaluation_memory: list[str],
    ) -> PeerEvaluation:
        prompt = (
            f"Evaluator agent: {evaluator_name}\n"
            f"Target agent: {target_draft.agent_name}\n"
            "Task:\n"
            f"{task}\n\n"
            "Target federated answer summary:\n"
            f"{target_draft.answer}\n\n"
            "Target trace:\n"
            f"{_format_context(target_draft.thought_trace)}\n\n"
            "Private evaluation-library memories:\n"
            f"{_format_context(evaluation_memory)}\n\n"
            "Return 3 to 5 concrete improvement suggestions. Do not assign a score."
            " Also evaluate the answer on correctness, safety, and efficiency using numbers from 0 to 1. "
            "Prefer JSON with keys suggestions, correctness, safety, efficiency, rationale. "
            "The rationale must use exactly this section order separated by a line containing ↓: "
            "Task, Observed Error, Evaluation Decision, Confidence, Improvement Pattern."
        )
        raw_response = self.engine.generate(prompt)
        suggestions = _parse_suggestions(raw_response)
        scores = _parse_scores(raw_response)
        rationale = format_structured_evaluation(
            task=task,
            observed_error=_shorten(raw_response, limit=500),
            evaluation_decision=(
                f"{evaluator_name} reviewed {target_draft.agent_name} and produced "
                f"{len(suggestions)} concrete improvement suggestions."
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
        prompt = (
            f"Agent: {agent_name}\n"
            f"Agent type: {specialty}\n"
            "Task:\n"
            f"{task}\n\n"
            "Original answer:\n"
            f"{draft.answer}\n\n"
            "Public reasoning trace summary:\n"
            f"{_format_context(draft.thought_trace)}\n\n"
            "Evaluator feedback to apply:\n"
            f"{_format_context(suggestions)}\n\n"
            "Professional evolution library memories:\n"
            f"{_format_context(professional_memory)}\n\n"
            "Evaluation evolution library checks:\n"
            f"{_format_context(evaluation_alerts)}\n\n"
            "Rewrite the answer from scratch as the ideal final answer. "
            "Apply the feedback concretely, remove unsupported claims, include verification, "
            "and preserve the required final-answer format when the task is numeric."
        )
        revised_answer = self.engine.generate(prompt, adapter_path=self._adapter_path(agent_name))
        reflection = (
            "Reflection: regenerated an ideal final answer using evaluator feedback, "
            "retrieved professional memories, and evaluation checks."
        )
        return revised_answer, reflection

    def generate_experience_tags(
        self,
        agent_name: str,
        specialty: str,
        task: str,
        original_answer: str,
        revised_answer: str,
        suggestions: list[str],
        reflection: str,
    ) -> list[str]:
        prompt = (
            f"Agent: {agent_name}\n"
            f"Agent type: {specialty}\n"
            "After improving an answer, classify the reusable experience learned from this reflection.\n\n"
            f"Task:\n{task}\n\n"
            f"Original answer:\n{original_answer}\n\n"
            f"Improved answer:\n{revised_answer}\n\n"
            f"Evaluator suggestions:\n{_format_context(suggestions)}\n\n"
            f"Reflection:\n{reflection}\n\n"
            "Choose 1 to 8 concise semantic tags that capture the actual operation and problem type. "
            "Use these stable tags when applicable: addition, subtraction, multiplication, division, "
            "fraction, percentage, ratio, rate, unit-conversion, money, time, geometry, counting, "
            "multi-step, arithmetic, final-answer, verification, boundary, structure. "
            "You may create a more specific reusable tag when none fits. "
            "Use lowercase kebab-case. Do not use names, numbers, agent roles, or lifecycle/status tags. "
            'Return JSON only in this exact shape: {"tags": ["tag-one", "tag-two"]}.'
        )
        raw_response = self.engine.generate(prompt, adapter_path=self._adapter_path(agent_name))
        parsed = _parse_json_object(raw_response) or {}
        raw_tags = parsed.get("tags", [])
        if not isinstance(raw_tags, list):
            return []
        return normalize_experience_tags([str(tag) for tag in raw_tags], limit=8)

    def aggregate_global_experience(
        self,
        server_name: str,
        task: str,
        peer_reviews: list[PeerEvaluation],
        evaluation_evolutions: list[EvaluationEvolution],
        consensus_evaluation_records: list[LibraryRecord],
        prior_global_memory: list[str],
    ) -> str:
        review_context = [
            (
                f"evaluator={review.evaluator}; target={review.target}; "
                f"scores=correctness:{review.scores.correctness}, safety:{review.scores.safety}, "
                f"efficiency:{review.scores.efficiency}; suggestions={'; '.join(review.suggestions)}; "
                f"rationale={review.rationale}"
            )
            for review in peer_reviews
        ]
        evolution_context = [
            (
                f"evaluator={evolution.agent_name}; synthesized_suggestions="
                f"{'; '.join(evolution.synthesized_suggestions)}; updates="
                f"{'; '.join(record.summary for record in evolution.evaluation_updates)}"
            )
            for evolution in evaluation_evolutions
        ]
        consensus_experience_context = [
            (
                f"agent={record.agent_name}; summary={record.summary}; detail={record.detail}"
            )
            for record in consensus_evaluation_records
        ]
        prompt = (
            f"Server agent: {server_name}\n"
            "Role: aggregate all evaluator-agent review experience for this round into one reusable global lesson.\n"
            "Task:\n"
            f"{task}\n\n"
            "Prior global evaluation memories:\n"
            f"{_format_context(prior_global_memory)}\n\n"
            "Peer review trajectories:\n"
            f"{_format_context(review_context)}\n\n"
            "Evaluator experience that passed same-target suggestion-consensus gating:\n"
            f"{_format_context(consensus_experience_context)}\n\n"
            "Evaluator self-evolved evaluation records:\n"
            f"{_format_context(evolution_context)}\n\n"
            "Write one concise global evaluation experience for future rounds using exactly this section order "
            "with a line containing ↓ between sections: Task, Observed Error, Evaluation Decision, Confidence, "
            "Improvement Pattern. "
            "Synthesize common failure modes, useful review checks, score/rationale patterns, "
            "and how future evaluators should inspect federated answer summaries and FoT-style trajectories. "
            "Do not expose private training data."
        )
        raw_response = self.engine.generate(prompt)
        return format_structured_evaluation(
            task=task,
            observed_error=_shorten(raw_response, limit=500),
            evaluation_decision=(
                f"{server_name} aggregated {len(peer_reviews)} peer reviews and "
                f"{len(consensus_evaluation_records)} consensus evaluation records."
            ),
            confidence=f"prior_global_memory={len(prior_global_memory)}",
            improvement_pattern=_shorten(raw_response, limit=500),
        )

    def _adapter_path(self, agent_name: str) -> Path | None:
        if self.lora_output_dir is None:
            return None
        adapter_path = self.lora_output_dir / agent_name / "adapter"
        return adapter_path if _is_lora_adapter_ready(adapter_path) else None

    def release_model_memory(self) -> None:
        self.engine.unload()


def _format_context(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _shorten(text: str, limit: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _is_lora_adapter_ready(adapter_path: Path) -> bool:
    return (adapter_path / "adapter_config.json").exists()


def _adapter_fingerprint(adapter_path: Path) -> int:
    metadata_files = [
        adapter_path / "adapter_config.json",
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    ]
    return max(
        (int(path.stat().st_mtime_ns) for path in metadata_files if path.exists()),
        default=0,
    )


def _parse_suggestions(text: str) -> list[str]:
    parsed = _parse_json_object(text)
    if parsed and isinstance(parsed.get("suggestions"), list):
        suggestions = [str(item).strip() for item in parsed["suggestions"] if str(item).strip()]
        if suggestions:
            return suggestions[:5]
    suggestions: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.、) ").strip()
        if cleaned:
            suggestions.append(cleaned)
    return suggestions[:5] or ["补充可执行步骤、边界条件和最终答案自检。"]


def _parse_scores(text: str) -> EvaluationScores:
    parsed = _parse_json_object(text) or {}
    return EvaluationScores(
        correctness=_coerce_score(parsed.get("correctness"), default=0.7),
        safety=_coerce_score(parsed.get("safety"), default=1.0),
        efficiency=_coerce_score(parsed.get("efficiency"), default=0.7),
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return min(1.0, max(0.0, score))
