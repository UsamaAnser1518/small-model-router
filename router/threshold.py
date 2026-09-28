"""Pick the escalation threshold from saved small-model predictions, without re-running anything.

For each candidate threshold: how many rows the small model would keep, how accurate
it is on those rows, and, given the frontier model's accuracy on the same split,
what the hybrid would score overall. The frontier accuracy is a projection input,
not a measurement of the hybrid; the benchmark harness on day 4 measures it for real.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.predictions import Prediction
from router.small import INVALID_LABEL

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    threshold: float
    n: int
    kept: int
    kept_accuracy: float
    escalated: int
    escalation_rate: float
    # None when the frontier accuracy is unknown.
    projected_accuracy: float | None

    def as_row(self) -> dict:
        return {
            "threshold": self.threshold,
            "kept": self.kept,
            "kept_accuracy": round(self.kept_accuracy, 4),
            "escalated": self.escalated,
            "escalation_rate": round(self.escalation_rate, 4),
            "projected_accuracy": (
                None if self.projected_accuracy is None else round(self.projected_accuracy, 4)
            ),
        }


def would_escalate(prediction: Prediction, threshold: float) -> bool:
    if prediction.label == INVALID_LABEL or prediction.confidence is None:
        return True
    return prediction.confidence < threshold


def sweep(
    gold: dict[str, str],
    predictions: list[Prediction],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    frontier_accuracy: float | None = None,
) -> list[ThresholdRow]:
    rows = []
    n = len(predictions)
    for t in thresholds:
        kept = [p for p in predictions if not would_escalate(p, t)]
        kept_correct = sum(p.label == gold[p.id] for p in kept)
        escalated = n - len(kept)
        kept_accuracy = kept_correct / len(kept) if kept else 0.0
        projected = None
        if frontier_accuracy is not None and n:
            projected = (kept_correct + escalated * frontier_accuracy) / n
        rows.append(
            ThresholdRow(
                threshold=t,
                n=n,
                kept=len(kept),
                kept_accuracy=kept_accuracy,
                escalated=escalated,
                escalation_rate=escalated / n if n else 0.0,
                projected_accuracy=projected,
            )
        )
    return rows
