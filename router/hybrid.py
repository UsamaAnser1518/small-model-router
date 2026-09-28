"""The router itself: answer locally when the small model is sure, escalate when it is not.

The small model always runs first. If its answer is a valid label and its
confidence is at or above the threshold, that answer is returned. Otherwise
the frontier model is asked and its answer is returned instead. An invalid
output from the small model is treated as zero confidence, so it always
escalates.

`predict` is the whole decision in one call, for evaluation. The serving layer
uses the two halves, `predict_local` and `escalate`, so it can run the small
model on its own thread and the frontier call on another.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from router.data import Example
from router.frontier import FrontierRouter
from router.predictions import Prediction
from router.small import INVALID_LABEL, SmallRouter

SMALL = "small"
FRONTIER = "frontier"


@dataclass
class HybridRouter:
    small: SmallRouter
    frontier: FrontierRouter
    threshold: float = 0.95
    # How many requests each system answered, for the /stats endpoint and the report.
    counts: dict[str, int] = field(default_factory=lambda: {SMALL: 0, FRONTIER: 0})

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")

    def should_escalate(self, prediction: Prediction) -> bool:
        if prediction.label == INVALID_LABEL or prediction.confidence is None:
            return True
        return prediction.confidence < self.threshold

    def predict_local(self, example: Example) -> Prediction:
        """The small model's answer, tagged as local. Check `should_escalate` on it."""
        return replace(self.small.predict(example), source=SMALL)

    def escalate(self, example: Example, local: Prediction) -> Prediction:
        """Ask the frontier model; keep the small model's confidence for the record.

        The caller waited for both models, so the latency is the sum. Token counts
        and therefore cost come from the frontier call alone; the small model is free.
        """
        remote = self.frontier.predict(example)
        return Prediction(
            id=example.id,
            label=remote.label,
            latency_ms=round(local.latency_ms + remote.latency_ms, 1),
            input_tokens=remote.input_tokens,
            output_tokens=remote.output_tokens,
            cached=remote.cached,
            confidence=local.confidence,
            source=FRONTIER,
        )

    def record(self, prediction: Prediction) -> Prediction:
        self.counts[prediction.source or SMALL] += 1
        return prediction

    def predict(self, example: Example) -> Prediction:
        local = self.predict_local(example)
        if not self.should_escalate(local):
            return self.record(local)
        return self.record(self.escalate(example, local))

    @property
    def escalation_rate(self) -> float:
        total = self.counts[SMALL] + self.counts[FRONTIER]
        return self.counts[FRONTIER] / total if total else 0.0
