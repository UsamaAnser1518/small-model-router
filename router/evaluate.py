"""Accuracy, macro-F1, latency and cost for a set of predictions."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from router.data import Example
from router.predictions import Prediction

# USD per million tokens, Anthropic first-party API rates.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class Report:
    n: int
    accuracy: float
    macro_f1: float
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    confusions: list[tuple[str, str, int]] = field(default_factory=list)

    def as_row(self, name: str) -> dict:
        return {
            "system": name,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "p50_ms": self.latency_p50_ms,
            "p95_ms": self.latency_p95_ms,
            "cost_per_1k_usd": round(self.cost_usd / max(self.n, 1) * 1000, 4),
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = set(gold) | set(pred)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(f1s) if f1s else 0.0


def evaluate(examples: list[Example], predictions: list[Prediction], model: str) -> Report:
    by_id = {p.id: p for p in predictions}
    gold, pred, latencies = [], [], []
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for e in examples:
        p = by_id[e.id]
        gold.append(e.label)
        pred.append(p.label)
        if not p.cached:
            latencies.append(p.latency_ms)
        if e.label != p.label:
            confusion[(e.label, p.label)] += 1
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    in_tok = sum(p.input_tokens for p in predictions)
    out_tok = sum(p.output_tokens for p in predictions)
    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    return Report(
        n=len(examples),
        accuracy=correct / len(examples) if examples else 0.0,
        macro_f1=macro_f1(gold, pred),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=(in_tok * in_price + out_tok * out_price) / 1_000_000,
        confusions=sorted(((g, p, n) for (g, p), n in confusion.items()), key=lambda t: -t[2])[:10],
    )


def confidence_buckets(
    examples: list[Example],
    predictions: list[Prediction],
    edges: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.01),
) -> list[dict]:
    """Accuracy within confidence bands, to see whether confidence predicts correctness."""
    gold = {e.id: e.label for e in examples}
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = [p for p in predictions if p.confidence is not None and lo <= p.confidence < hi]
        if not band:
            continue
        correct = sum(p.label == gold[p.id] for p in band)
        rows.append(
            {
                "band": f"[{lo:.2f}, {min(hi, 1.0):.2f})",
                "n": len(band),
                "accuracy": round(correct / len(band), 3),
            }
        )
    return rows
