"""Score every system on the same rows and produce one comparison table.

The hybrid rows are composed, not re-run: the small model's saved predictions
decide which rows escalate, and the frontier model's saved predictions supply the
answers for those rows. Both sets of predictions are real measurements on the
same examples, so the hybrid row is a measurement too, just assembled from parts.
Latency for an escalated row is the sum of both calls; cost is the frontier's.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.data import Example
from router.evaluate import Report, evaluate
from router.hybrid import FRONTIER, SMALL
from router.predictions import Prediction
from router.threshold import would_escalate


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    system: str
    report: Report
    escalation_rate: float | None = None

    def as_row(self) -> dict:
        row = self.report.as_row(self.system)
        row["escalation_rate"] = (
            None if self.escalation_rate is None else round(self.escalation_rate, 4)
        )
        return row


def compose_hybrid(
    small: list[Prediction], frontier: list[Prediction], threshold: float
) -> list[Prediction]:
    """Assemble hybrid predictions from the two systems' predictions on the same ids."""
    remote = {p.id: p for p in frontier}
    out = []
    for local in small:
        if not would_escalate(local, threshold):
            out.append(
                Prediction(
                    id=local.id,
                    label=local.label,
                    latency_ms=local.latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    confidence=local.confidence,
                    source=SMALL,
                )
            )
            continue
        far = remote[local.id]
        out.append(
            Prediction(
                id=local.id,
                label=far.label,
                latency_ms=round(local.latency_ms + far.latency_ms, 1),
                input_tokens=far.input_tokens,
                output_tokens=far.output_tokens,
                cached=far.cached,
                confidence=local.confidence,
                source=FRONTIER,
            )
        )
    return out


def hybrid_row(
    name: str,
    examples: list[Example],
    small: list[Prediction],
    frontier: list[Prediction],
    threshold: float,
    frontier_model: str,
) -> BenchmarkRow:
    composed = compose_hybrid(small, frontier, threshold)
    escalated = sum(p.source == FRONTIER for p in composed)
    return BenchmarkRow(
        system=name,
        report=evaluate(examples, composed, model=frontier_model),
        escalation_rate=escalated / len(composed) if composed else 0.0,
    )


def markdown_table(rows: list[dict], pending: list[str] = ()) -> str:
    """The comparison table as GitHub-flavoured Markdown, ready to paste into the README."""
    header = "| System | Accuracy | Macro-F1 | p50 | p95 | Cost / 1k msgs | Escalated |"
    rule = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, rule]
    for r in rows:
        esc = "" if r["escalation_rate"] is None else f"{r['escalation_rate']:.1%}"
        lines.append(
            f"| {r['system']} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | "
            f"{r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms | ${r['cost_per_1k_usd']:.2f} | {esc} |"
        )
    for name in pending:
        lines.append(f"| {name} | pending | | | | | |")
    return "\n".join(lines)
