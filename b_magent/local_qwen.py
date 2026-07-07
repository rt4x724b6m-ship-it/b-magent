from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.generation_config = generation_config or QwenGenerationConfig()
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self._load()
        return self._model

    def generate(self, prompt: str) -> str:
        self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful math reasoning assistant. "
                    "Solve the problem step by step and end with '#### <answer>'."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=self.generation_config.do_sample,
        )
        completion_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()

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
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen requires transformers. Install transformers and torch, "
                "then pass a local Qwen2.5-1.5B model path or use the default Hugging Face id."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._resolve_torch_dtype(),
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )


class LocalQwenAgentModel:
    """Qwen2.5-backed agent model implementing the train/vote interfaces."""

    def __init__(self, agent_name: str, engine: LocalQwenEngine) -> None:
        self.agent_name = agent_name
        self.engine = engine
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
            f"Question:\n{question}"
        )
        return self.engine.generate(prompt)
