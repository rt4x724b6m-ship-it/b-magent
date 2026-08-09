from __future__ import annotations

import gc
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation_format import format_confidence_from_scores, format_structured_evaluation
from .models import Draft, EvaluationEvolution, EvaluationScores, LibraryRecord, PeerEvaluation
from .self_evolution import normalize_experience_tags


DEFAULT_QWEN_MODEL = "models/gemma-3-4b-it"

NUMERIC_ANSWER_INSTRUCTION = (
    "You are a careful math reasoning assistant. Solve the problem step by step. "
    "End your response with a final line exactly in this format: #### <numeric_answer>. "
    "Do not include units, explanations, or full sentences after ####."
)

TRAVEL_PLANNING_INSTRUCTION = (
    "You are a professional travel planning agent. Strictly follow every rule below "
    "when completing TravelPlanner itinerary tasks.\n\n"
    "INPUT: The user provides target cities, total trip days, budget ceiling, travel dates, "
    "preferences, transport restrictions, and traveller details.\n\n"
    "HARD RULES — never violate:\n"
    "1. No fabrication. Opening hours, ticket prices, travel durations, and fares must come "
    "from the query or retrieval results only — never guess or invent figures.\n"
    "2. Budget cap: total trip spend (tickets + meals + transport) must not exceed the given budget.\n"
    "3. Time validity: never schedule visits outside attraction opening hours; reserve realistic "
    "travel time for inter-city moves; never place unreachable destinations on the same day.\n"
    "4. Geographic coherence: keep same-day attractions close together to minimise commuting; "
    "schedule inter-city travel in the morning or evening slot.\n"
    "5. Structured output: produce a day-by-day itinerary. Each day must include: "
    "date · city · morning · afternoon · evening · daily cost · transport notes.\n"
    "6. Cost transparency: itemise every expense (tickets, meals, fares) with amounts; "
    "end with total trip cost ≤ given budget.\n\n"
    "FORBIDDEN: Do not invent prices, hours, or durations. Do not exceed budget. "
    "Do not schedule closed attractions. Do not use vague phrases such as 'explore the area' "
    "— name specific venues. Do not omit the total cost summary.\n\n"
    "Output: Return strictly the following JSON array, no extra text:\n"
    '[{"days":1,"current_city":"...","transportation":"...","breakfast":"...","attraction":"...","lunch":"...","dinner":"...","accommodation":"..."},...]'
)

_TRAVEL_TASK_RE = re.compile(r"(?m)^(?:Origin:|Query:|Gold plan:|Destination:)")


def _is_travel_prompt(text: str) -> bool:
    """Return True when the prompt content comes from TravelPlanner."""
    return bool(_TRAVEL_TASK_RE.search(text))


@dataclass(frozen=True)
class QwenGenerationConfig:
    # Travel plans produce long JSON arrays; 2048 tokens avoids mid-plan truncation.
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


class LocalQwenEngine:
    """Lazy Transformers loader for the local text-only Qwen model."""

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

    def unload(self) -> None:
        """Release inference and adapter models before an in-process LoRA refresh."""
        with self._generation_lock:
            self._adapter_models.clear()
            self._model = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _generate_unlocked(self, prompt: str, adapter_path: str | Path | None = None) -> str:
        self._load()
        model = self._model
        if adapter_path is not None and _is_lora_adapter_ready(Path(adapter_path)):
            model = self._load_adapter_model(Path(adapter_path))
        messages: list[dict[str, Any]] = []
        # Auto-select system prompt: travel planning tasks get a dedicated instruction
        # so the model does not append a numeric #### answer line.
        if _is_travel_prompt(prompt):
            active_system_prompt = TRAVEL_PLANNING_INSTRUCTION
        else:
            active_system_prompt = self.system_prompt
        if active_system_prompt:
            messages.append({"role": "system", "content": active_system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text=[text], padding=True, return_tensors="pt").to(model.device)
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
                "Local model directory was not found. "
                f"Expected: {model_path.resolve()}. "
                "Pass --model-path /path/to/gemma-3-4b-it or place the model under "
                "models/gemma-3-4b-it."
            )
        try:
            import transformers

            AutoConfig = transformers.AutoConfig
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen requires transformers. Install transformers and torch, "
                "then pass a local gemma-3-4b-it model path."
            ) from exc

        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        if getattr(config, "model_type", "") == "qwen2_5_vl":
            raise RuntimeError(
                "This project uses the text-only gemma-3-4b-it model; a Qwen-VL model path is not supported."
            )
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


def _extract_image_path(prompt: str) -> Path | None:
    """Return the first existing local image referenced by a normalized task."""
    for match in re.finditer(r"(?m)^Image:\s*(.+?)\s*$", prompt):
        path = Path(match.group(1).strip())
        if path.is_file():
            return path
    return None


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
        if _is_travel_prompt(question):
            prompt = (
                f"{TRAVEL_PLANNING_INSTRUCTION}\n\n"
                f"Role: {self.agent_name} (professional travel planning agent)\n"
                "Tasks: ① Generate a day-by-day itinerary from the query. "
                "② Each day must fill in all six fields: transportation / breakfast / attraction / lunch / dinner / accommodation.\n"
                "Constraints: Only use information given in the query. If missing, fill with \"-\". "
                "Budget and local constraints must all be satisfied. "
                "Server guidance is for verification only — do not use it as a direct answer.\n\n"
                "Private examples (for reference only, do not copy verbatim):\n"
                f"{context or '(none)'}\n\n"
                "Query:\n"
                f"{question}\n\n"
                "Server evaluation guidance:\n"
                f"{server_guidance or '(none)'}\n\n"
                "Output: Return strictly the JSON array, no extra text."
            )
        else:
            prompt = (
                f"Agent: {self.agent_name}\n"
                f"Private examples:\n{context or '(none)'}\n\n"
                f"Question:\n{question}\n\n"
                "Server evaluation guidance:\n"
                f"{server_guidance or '(none)'}\n\n"
                "Answer independently while applying the server's observable risk checks. "
                "Treat guidance as verification advice, not as a proposed answer.\n\n"
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
        if _is_travel_prompt(task):
            prompt = (
                f"{TRAVEL_PLANNING_INSTRUCTION}\n\n"
                f"Role: {agent_name} ({specialty})\n"
                "Tasks: ① Generate a day-by-day itinerary from the query. "
                "② Each day must fill in all six fields: transportation / breakfast / attraction / lunch / dinner / accommodation.\n"
                "Constraints: Only use information given in the query. If missing, fill with \"-\". "
                "Budget and local constraints must all be satisfied.\n\n"
                "Query:\n"
                f"{task}\n\n"
                "Private training examples (for reference only, do not copy verbatim):\n"
                f"{_format_context(private_training)}\n\n"
                "Professional evolution library (follow with priority):\n"
                f"{_format_context(professional_memory)}\n\n"
                "Evaluation alerts (must avoid):\n"
                f"{_format_context(evaluation_alerts)}\n\n"
                "Output: Return strictly the JSON array, no extra text."
            )
        else:
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
                "Solve the text problem carefully, show the necessary reasoning, and end with a line in the "
                "exact format #### <numeric_answer>."
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
        if _is_travel_prompt(task):
            # Extract budget/people/constraint summary from the task for the evaluator
            budget_match = re.search(r"Budget:\s*\$?([\d,]+)", task)
            people_match = re.search(r"Travelers:\s*(\d+)", task)
            days_match = re.search(r"Days:\s*(\d+)", task)
            budget_hint = f"  budget=${budget_match.group(1)}" if budget_match else ""
            people_hint = f"  travelers={people_match.group(1)}" if people_match else ""
            days_hint = f"  days={days_match.group(1)}" if days_match else ""
            prompt = (
                f"Evaluator agent: {evaluator_name}\n"
                f"Target agent: {target_draft.agent_name}\n"
                f"Task constraints:{budget_hint}{people_hint}{days_hint}\n"
                "Task:\n"
                f"{task}\n\n"
                "Target itinerary:\n"
                f"{target_draft.answer}\n\n"
                "Private evaluation-library memories:\n"
                f"{_format_context(evaluation_memory)}\n\n"
                "Evaluate the itinerary against ALL of the following travel-plan quality checks:\n"
                "① COMPLETENESS — each day must have: transportation, breakfast, attraction, lunch, dinner, accommodation.\n"
                "② BUDGET — sum up all daily costs; total must not exceed the stated budget.\n"
                "③ CONSTRAINT SATISFACTION — check house rules, cuisine preferences, room type, transportation mode.\n"
                "④ FABRICATION — verify no invented flight numbers, hotels, or restaurants outside query scope.\n"
                "⑤ GEOGRAPHIC COHERENCE — same-day attractions must be in realistic proximity; inter-city moves need enough transit time.\n"
                "⑥ TRAVELER COUNT — meal/accommodation costs must account for the stated number of travelers.\n"
                "Return 1–4 concrete, actionable suggestions addressing whichever checks fail. "
                "If the itinerary is sound, return only one short verification note.\n"
                "Score from 0 to 1: correctness=plan correctness vs constraints, "
                "safety=no fabrication or policy violation, efficiency=coverage and cost optimisation.\n"
                "Return exactly one JSON object with keys: suggestions, correctness, safety, efficiency, rationale.\n"
                "Do not wrap in markdown. Rationale sections (separated by ↓): "
                "Task, Observed Error, Evaluation Decision, Confidence, Improvement Pattern."
            )
        else:
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
                "Check whether the reasoning, calculations, and final numeric answer are consistent with the task. "
                "Return 1 to 3 concrete suggestions; if the answer is already correct, suggest only concise verification. "
                "Score correctness, safety, and efficiency from 0 to 1, where 1 means fully correct/safe/concise, "
                "0.5 means materially incomplete, and 0 means wrong or unusable. Harmless math answers should "
                "normally receive safety=1. Return only one valid JSON object with keys suggestions, correctness, "
                "safety, efficiency, rationale. Do not wrap it in markdown. "
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
        if _is_travel_prompt(task):
            prompt = (
                f"{TRAVEL_PLANNING_INSTRUCTION}\n\n"
                f"Role: {agent_name} ({specialty}, itinerary revision mode)\n"
                "Tasks: ① Rewrite the full itinerary based on evaluator feedback. "
                "② Each day must fill in all six fields: transportation / breakfast / attraction / lunch / dinner / accommodation.\n"
                "Constraints: Only use information given in the query. If missing, fill with \"-\". "
                "Budget and local constraints must all be satisfied. Apply every piece of evaluator feedback.\n\n"
                "Query:\n"
                f"{task}\n\n"
                "Original itinerary (for reference only, improve thoroughly):\n"
                f"{draft.answer}\n\n"
                "Evaluator feedback (apply every item):\n"
                f"{_format_context(suggestions)}\n\n"
                "Professional evolution library (follow with priority):\n"
                f"{_format_context(professional_memory)}\n\n"
                "Evaluation alerts (must avoid):\n"
                f"{_format_context(evaluation_alerts)}\n\n"
                "Output: Return strictly the JSON array, no extra text."
            )
        else:
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
                "Recalculate the problem, apply the feedback concretely, and preserve the required numeric "
                "final-answer format."
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
            "Use mathematical operation tags when applicable, such as addition, subtraction, multiplication, "
            "division, percentage, ratio, money, or arithmetic. "
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
        is_travel = _is_travel_prompt(task)
        if is_travel:
            domain_instruction = (
                "This is a travel planning task. In addition to general quality patterns, "
                "synthesize the following travel-specific lessons:\n"
                "• Budget violations: which itineraries exceeded the budget and how?\n"
                "• Constraint gaps: which local constraints (cuisine, house rule, room type, transport) were missed?\n"
                "• Fabrication patterns: what kinds of invented data appeared most?\n"
                "• Day-completeness failures: which of the six daily fields (transportation/breakfast/attraction/"
                "lunch/dinner/accommodation) were most often missing or vague?\n"
                "• Geographic errors: unrealistic same-day distances or insufficient transit time?\n"
                "Encode these as actionable review checks future evaluators must apply."
            )
        else:
            domain_instruction = (
                "Synthesize common failure modes, useful review checks, score/rationale patterns, "
                "and how future evaluators should inspect federated answer summaries. "
                "Preserve reusable checks for arithmetic accuracy, reasoning completeness, and final-answer consistency."
            )
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
            f"{domain_instruction}\n\n"
            "Write one concise global evaluation experience for future rounds using exactly this section order "
            "with a line containing ↓ between sections: Task, Observed Error, Evaluation Decision, Confidence, "
            "Improvement Pattern. Do not expose private training data."
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
    parsed = _parse_json_object(text)
    required = {"correctness", "safety", "efficiency"}
    if parsed is None or not required.issubset(parsed):
        return EvaluationScores(
            correctness=0.0,
            safety=0.0,
            efficiency=0.0,
            parsed_successfully=False,
        )
    return EvaluationScores(
        correctness=_coerce_score(parsed.get("correctness"), default=0.0),
        safety=_coerce_score(parsed.get("safety"), default=0.0),
        efficiency=_coerce_score(parsed.get("efficiency"), default=0.0),
        parsed_successfully=True,
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
