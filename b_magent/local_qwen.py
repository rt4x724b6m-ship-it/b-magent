from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Draft, EvaluationScores, PeerEvaluation


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
        self._adapter_models: dict[str, Any] = {}

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self._load()
        return self._model

    def generate(self, prompt: str, adapter_path: str | Path | None = None) -> str:
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
        key = str(adapter_path.resolve())
        if key in self._adapter_models:
            return self._adapter_models[key]
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading LoRA adapters requires peft.") from exc
        adapter_model = PeftModel.from_pretrained(self._model, key)
        self._adapter_models[key] = adapter_model
        return adapter_model

    def _resolve_torch_dtype(self) -> Any:
        if self.torch_dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Local Qwen requires torch.") from exc
        return getattr(torch, self.torch_dtype)

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        model_path = Path(self.model_name_or_path)
        if self.local_files_only and not model_path.exists():
            raise RuntimeError(
                "Local Qwen model directory was not found. "
                f"Expected: {model_path.resolve()}. "
                "Pass --model-path /path/to/Qwen2.5-1.5B-Instruct or place the model under models/Qwen2.5-1.5B-Instruct."
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
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
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
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
        prefer_distilled_adapter: bool = True,
    ) -> None:
        self.agent_name = agent_name
        self.engine = engine
        self.lora_output_dir = Path(lora_output_dir) if lora_output_dir is not None else None
        self.prefer_distilled_adapter = prefer_distilled_adapter
        self.training_examples: list[str] = []

    def train_batch(self, batch: object) -> None:
        if not batch:
            return
        for sample in batch:
            to_training_text = getattr(sample, "to_training_text", None)
            if callable(to_training_text):
                self.training_examples.append(to_training_text())

    def generate(self, question: str) -> str:
        context = "\n".join(self.training_examples[-4:])
        prompt = (
            f"Agent: {self.agent_name}\n"
            f"Private examples:\n{context or '(none)'}\n\n"
            f"Question:\n{question}\n\n"
            f"Output constraint:\n{NUMERIC_ANSWER_INSTRUCTION}"
        )
        adapter_path = self._adapter_path()
        if adapter_path is None:
            return self.engine.generate(prompt)
        return self.engine.generate(prompt, adapter_path=adapter_path)

    def _adapter_path(self) -> Path | None:
        if self.lora_output_dir is None:
            return None
        distilled_adapter_path = self.lora_output_dir / self.agent_name / "distilled_adapter"
        if self.prefer_distilled_adapter and _is_lora_adapter_ready(distilled_adapter_path):
            return distilled_adapter_path
        adapter_path = self.lora_output_dir / self.agent_name / "adapter"
        return adapter_path if _is_lora_adapter_ready(adapter_path) else None


class LocalQwenEvolutionBackend:
    """Local-Qwen backend for the b_magent self-evolution workflow."""

    def __init__(
        self,
        engine: LocalQwenEngine,
        lora_output_dir: str | Path | None = None,
        prefer_distilled_adapter: bool = False,
    ) -> None:
        self.engine = engine
        self.lora_output_dir = Path(lora_output_dir) if lora_output_dir is not None else None
        self.prefer_distilled_adapter = prefer_distilled_adapter

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
            "Target answer:\n"
            f"{target_draft.answer}\n\n"
            "Target trace:\n"
            f"{_format_context(target_draft.thought_trace)}\n\n"
            "Private evaluation-library memories:\n"
            f"{_format_context(evaluation_memory)}\n\n"
            "Return 3 to 5 concrete improvement suggestions. Do not assign a score."
            " Also evaluate the answer on correctness, safety, and efficiency using numbers from 0 to 1. "
            "Prefer JSON with keys suggestions, correctness, safety, efficiency, rationale."
        )
        raw_response = self.engine.generate(prompt)
        suggestions = _parse_suggestions(raw_response)
        scores = _parse_scores(raw_response)
        return PeerEvaluation(
            evaluator=evaluator_name,
            target=target_draft.agent_name,
            suggestions=suggestions,
            rationale=raw_response,
            evaluation_memory_used=evaluation_memory,
            scores=scores,
        )

    def _adapter_path(self, agent_name: str) -> Path | None:
        if self.lora_output_dir is None:
            return None
        distilled_adapter_path = self.lora_output_dir / agent_name / "distilled_adapter"
        if self.prefer_distilled_adapter and _is_lora_adapter_ready(distilled_adapter_path):
            return distilled_adapter_path
        adapter_path = self.lora_output_dir / agent_name / "adapter"
        return adapter_path if _is_lora_adapter_ready(adapter_path) else None


def _format_context(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _is_lora_adapter_ready(adapter_path: Path) -> bool:
    return (adapter_path / "adapter_config.json").exists()


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
