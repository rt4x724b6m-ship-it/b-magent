from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "gpt56_sol_output"
DEFAULT_DATASETS = ("mm-vet", "infographicsvqa")
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
IMAGE_ELEMENTS_FIELD = "image_elements"

ELEMENT_PROMPT = """Analyze only the supplied image. Do not answer any associated question.
Return valid JSON and nothing else, using exactly this shape:
{
  "summary": "one concise overall description",
  "objects": ["all salient people, objects, icons, charts, and illustrations"],
  "visible_text": ["important text visible in the image, preserving wording"],
  "layout": "spatial arrangement and relationships among elements",
  "colors": ["dominant or semantically important colors"]
}
Use empty arrays or an empty string when an element is absent. Do not infer facts that are not visible."""


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("Responses API result contains no output text")


def parse_elements(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        candidate = candidate.rsplit("```", 1)[0].strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("image element result must be a JSON object")
    return value


class GPT56SolVisionClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def describe(self, image_path: Path) -> dict[str, Any]:
        request = {
            "model": self.model,
            "reasoning": {"effort": "none"},
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ELEMENT_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_path),
                        "detail": "original",
                    },
                ],
            }],
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("responses", json=request)
                response.raise_for_status()
                return parse_elements(response_text(response.json()))
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


def enrich_jsonl(
    input_file: Path,
    output_file: Path,
    describe: Callable[[Path], dict[str, Any]],
    limit: int | None = None,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    cache: dict[Path, dict[str, Any]] = {}
    if output_file.exists():
        with output_file.open(encoding="utf-8") as handle:
            completed = [json.loads(line) for line in handle if line.strip()]
        for row in completed:
            image = row.get("image") or row.get("image_path")
            if image and IMAGE_ELEMENTS_FIELD in row:
                cache[(input_file.parent / image).resolve()] = row[IMAGE_ELEMENTS_FIELD]

    processed = len(completed)
    mode = "a" if completed else "w"
    with input_file.open(encoding="utf-8") as source, output_file.open(mode, encoding="utf-8") as target:
        for index, line in enumerate(source):
            if limit is not None and index >= limit:
                break
            if index < processed or not line.strip():
                continue
            row = json.loads(line)
            if IMAGE_ELEMENTS_FIELD not in row:
                image_value = row.get("image") or row.get("image_path")
                if not image_value:
                    raise ValueError(f"{input_file}:{index + 1} has no image field")
                image_path = (input_file.parent / str(image_value)).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"image not found: {image_path}")
                if image_path not in cache:
                    cache[image_path] = describe(image_path)
                row[IMAGE_ELEMENTS_FIELD] = cache[image_path]
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            processed += 1
            print(f"[{input_file.parent.name}/{input_file.stem}] {processed}", flush=True)
    return processed


def discover_inputs(data_root: Path, datasets: list[str]) -> list[Path]:
    inputs: list[Path] = []
    for dataset in datasets:
        dataset_dir = data_root / dataset
        if not dataset_dir.exists():
            print(f"skip missing dataset: {dataset_dir}")
            continue
        inputs.extend(path for path in sorted(dataset_dir.glob("*.jsonl")) if path.stat().st_size)
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add GPT-5.6 Sol image element descriptions to visual JSONL datasets."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required; do not put API keys in source code")
    inputs = discover_inputs(args.data_root, args.datasets)
    if not inputs:
        raise SystemExit("no non-empty JSONL dataset splits found")

    client = GPT56SolVisionClient(api_key, args.base_url, args.model)
    try:
        for input_file in inputs:
            relative = input_file.relative_to(args.data_root)
            count = enrich_jsonl(input_file, args.output_root / relative, client.describe, args.limit)
            print(f"wrote {count} rows: {args.output_root / relative}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
