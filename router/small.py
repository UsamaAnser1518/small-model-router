"""Local inference with a fine-tuned small model, with a confidence score per answer.

Confidence is the probability the model assigns to the exact label it produced:
the product of the per-token probabilities under greedy decoding. It is what
the router will threshold on to decide when to escalate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from pathlib import Path

from router.data import Example
from router.finetune import SYSTEM_PROMPT
from router.predictions import Prediction

INVALID_LABEL = "<invalid>"


class SmallRouter:
    def __init__(
        self,
        labels: list[str],
        model: str,
        adapter_path: Path | None = None,
        max_tokens: int = 12,
    ) -> None:
        self.labels = set(labels)
        self.model_name = model
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> SmallRouter:
        from mlx_lm import load  # heavy import, deferred

        self._model, self._tokenizer = load(
            self.model_name,
            adapter_path=str(self.adapter_path) if self.adapter_path else None,
        )
        return self

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def prompt_for(self, text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str) -> Iterable[tuple[int, float]]:
        """Yield (token_id, logprob) under greedy decoding. Overridden in tests."""
        from mlx_lm import stream_generate

        if self._model is None:
            self.load()
        for response in stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=self.max_tokens
        ):
            yield response.token, float(response.logprobs[response.token])

    def predict(self, example: Example) -> Prediction:
        started = time.perf_counter()
        prompt = self.prompt_for(example.text)
        tokens: list[int] = []
        total_logprob = 0.0
        for token, logprob in self.generate(prompt):
            if token in self.tokenizer.eos_token_ids:
                break
            tokens.append(token)
            total_logprob += logprob
        latency_ms = (time.perf_counter() - started) * 1000

        text = self.tokenizer.decode(tokens).strip()
        return Prediction(
            id=example.id,
            label=text if text in self.labels else INVALID_LABEL,
            latency_ms=round(latency_ms, 1),
            input_tokens=0,
            output_tokens=len(tokens),
            confidence=math.exp(total_logprob),
        )
