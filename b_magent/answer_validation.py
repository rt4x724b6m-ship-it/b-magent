from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

import httpx

from .retrieval_training import strip_hidden_retrieval_labels


@dataclass(frozen=True)
class AnswerValidationResult:
    correct: bool
    requirements_met: list[str]
    requirements_missed: list[str]
    unsupported_claims: list[str]
    rationale: str


class AnswerValidator(Protocol):
    def validate(self, task: str, answer: str) -> AnswerValidationResult:
        """Judge whether an answer satisfies the task requirements."""


class GPT56SolRequirementValidator:
    """Quality-first requirement judge backed by GPT-5.6 Sol Responses API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GPT-5.6 Sol validation requires an OpenAI API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = client or httpx.Client(timeout=timeout)
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must not be negative")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def validate(self, task: str, answer: str) -> AnswerValidationResult:
        visible_task = strip_hidden_retrieval_labels(task)
        request_payload = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
            "input": [
                    {
                        "role": "developer",
                        "content": (
                            "You are a strict but semantics-based answer validator. Mark an answer correct "
                            "when it satisfies the user's requested outcome and every explicit hard constraint. "
                            "Do not require wording, ordering, formatting, or choices to match a reference answer "
                            "unless the task explicitly requires them. Treat harmless extra detail as acceptable. "
                            "Mark it incorrect for a missed hard constraint, an internally inconsistent plan, or "
                            "a material factual claim unsupported by the candidate evidence. Evaluate only the "
                            "observable task, evidence, and answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"TASK AND AVAILABLE EVIDENCE:\n{visible_task}\n\nANSWER TO VALIDATE:\n{answer}",
                    },
            ],
            "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "answer_requirement_validation",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "correct": {"type": "boolean"},
                                "requirements_met": {"type": "array", "items": {"type": "string"}},
                                "requirements_missed": {"type": "array", "items": {"type": "string"}},
                                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                                "rationale": {"type": "string"},
                            },
                            "required": [
                                "correct",
                                "requirements_met",
                                "requirements_missed",
                                "unsupported_claims",
                                "rationale",
                            ],
                            "additionalProperties": False,
                        },
                    }
            },
        }
        last_error: Exception | None = None
        last_response: httpx.Response | None = None
        attempts_used = 0
        for attempt in range(self.max_retries + 1):
            attempts_used = attempt + 1
            last_response = None
            try:
                response = self.client.post(
                    f"{self.base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
                last_response = response
                response.raise_for_status()
                payload = response.json()
                validation_payload = json.loads(_response_output_text(payload))
                return _parse_validation_result(validation_payload)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_validation_failure(exc, last_response):
                    break
                time.sleep(self.retry_backoff * (2**attempt))

        detail = _validation_failure_detail(last_response)
        raise RuntimeError(
            f"GPT-5.6 Sol answer validation failed after {attempts_used} attempt(s); "
            f"training was stopped. {detail}"
        ) from last_error


def build_answer_validator_from_env() -> GPT56SolRequirementValidator:
    load_local_openai_environment()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when --answer-validator gpt-5.6-sol is selected"
        )
    return GPT56SolRequirementValidator(
        api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        reasoning_effort=os.environ.get("OPENAI_VALIDATOR_REASONING_EFFORT", "medium"),
        max_retries=int(os.environ.get("OPENAI_VALIDATOR_MAX_RETRIES", "3")),
        retry_backoff=float(os.environ.get("OPENAI_VALIDATOR_RETRY_BACKOFF", "1")),
    )


def load_local_openai_environment(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    allowed = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_VALIDATOR_REASONING_EFFORT",
        "OPENAI_VALIDATOR_MAX_RETRIES",
        "OPENAI_VALIDATOR_RETRY_BACKOFF",
    }
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _response_output_text(payload: dict[str, object]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses payload did not contain output")
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for content in item["content"]:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("Responses payload did not contain output text")


def _is_retryable_validation_failure(
    error: Exception,
    response: httpx.Response | None,
) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if response is None:
        return False
    if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
        return True
    # Successful responses with empty, HTML, or malformed JSON bodies are commonly
    # produced by transient compatibility-proxy failures.
    return response.is_success and isinstance(
        error, (json.JSONDecodeError, KeyError, TypeError, ValueError)
    )


def _validation_failure_detail(response: httpx.Response | None) -> str:
    if response is None:
        return "No HTTP response was received."
    content_type = response.headers.get("content-type", "(missing)")
    body = " ".join(response.text.strip().split())
    if len(body) > 500:
        body = body[:500] + "..."
    return (
        f"Last response: HTTP {response.status_code}, content-type={content_type}, "
        f"body={body or '(empty)'!r}."
    )


def _parse_validation_result(payload: object) -> AnswerValidationResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("correct"), bool):
        raise ValueError("invalid answer validation payload")
    return AnswerValidationResult(
        correct=payload["correct"],
        requirements_met=_string_list(payload.get("requirements_met")),
        requirements_missed=_string_list(payload.get("requirements_missed")),
        unsupported_claims=_string_list(payload.get("unsupported_claims")),
        rationale=str(payload.get("rationale", "")).strip(),
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("answer validation list field is invalid")
    return [str(item).strip() for item in value if str(item).strip()]
