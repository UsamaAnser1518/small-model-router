"""The one record every system produces, so every system is scored the same way."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    label: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached: bool = False
    # Probability the model assigns to its own answer, when the system exposes one.
    # The frontier API does not; the small model does, and the router relies on it.
    confidence: float | None = None
