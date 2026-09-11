"""Zero-shot routing with a frontier model, used as the accuracy ceiling to beat.

Every prediction is cached on disk keyed by (model, effort, prompt version,
example id), so re-running an eval after a crash or a code change that does
not touch the prompt costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from router.data import Example

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "low"  # classification does not benefit from deep thinking
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You route customer-support messages for a retail bank to the team that handles them.\n\n"
    "You will be given one customer message and the full list of routing labels. Pick the "
    "single label that best describes what the customer needs. Every message fits exactly one "
    "label. If two labels seem close, prefer the more specific one.\n\n"
    "Respond only with the JSON object described by the output schema."
)


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    label: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached: bool = False


def _cache_key(model: str, effort: str, example_id: str) -> str:
    raw = f"{model}|{effort}|{PROMPT_VERSION}|{example_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class FrontierRouter:
    def __init__(
        self,
        labels: list[str],
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        cache_dir: Path | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        self.labels = sorted(labels)
        self.model = model
        self.effort = effort
        self.cache_dir = cache_dir
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        # Zero-arg constructor: picks up ANTHROPIC_API_KEY or an `ant auth login` profile.
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": self.labels}},
            "required": ["label"],
            "additionalProperties": False,
        }

    def _user_message(self, text: str) -> str:
        label_list = "\n".join(f"- {label}" for label in self.labels)
        return f"Routing labels:\n{label_list}\n\nCustomer message:\n{text}"

    def _cache_path(self, example_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_cache_key(self.model, self.effort, example_id)}.json"

    def predict(self, example: Example) -> Prediction:
        cache_path = self._cache_path(example.id)
        if cache_path is not None and cache_path.exists():
            data = json.loads(cache_path.read_text())
            return Prediction(**{**data, "cached": True})

        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # The system prompt and label list never change across calls,
                    # so cache them; only the customer message varies.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": self._user_message(example.text)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self.output_schema},
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused example {example.id}: {response.stop_details}")

        text = next(block.text for block in response.content if block.type == "text")
        label = json.loads(text)["label"]
        prediction = Prediction(
            id=example.id,
            label=label,
            latency_ms=round(latency_ms, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(asdict(prediction)))
        return prediction
