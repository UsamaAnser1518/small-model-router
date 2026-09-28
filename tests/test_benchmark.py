from router.benchmark import compose_hybrid, hybrid_row, markdown_table
from router.data import Example
from router.hybrid import FRONTIER, SMALL
from router.predictions import Prediction
from router.small import INVALID_LABEL


def _small(id_, label, confidence, latency=100.0):
    return Prediction(
        id=id_, label=label, latency_ms=latency, input_tokens=0, output_tokens=2,
        confidence=confidence,
    )  # fmt: skip


def _far(id_, label, latency=1000.0, cached=False):
    return Prediction(
        id=id_, label=label, latency_ms=latency, input_tokens=100, output_tokens=5, cached=cached
    )


def test_compose_hybrid_routes_by_threshold_and_sums_latency():
    small = [_small("1", "a", 0.99), _small("2", "b", 0.5), _small("3", INVALID_LABEL, 0.99)]
    far = [_far("1", "a"), _far("2", "a"), _far("3", "b", cached=True)]
    composed = {p.id: p for p in compose_hybrid(small, far, threshold=0.95)}
    assert composed["1"].source == SMALL and composed["1"].input_tokens == 0
    assert composed["2"].source == FRONTIER and composed["2"].label == "a"
    assert composed["2"].latency_ms == 1100.0
    assert composed["2"].confidence == 0.5  # the small model's, kept for the record
    assert composed["3"].source == FRONTIER and composed["3"].cached is True


def test_hybrid_row_reports_escalation_rate_and_frontier_cost_only():
    examples = [Example(id=i, text="t", label="a") for i in "12"]
    small = [_small("1", "a", 0.99), _small("2", "b", 0.5)]
    far = [_far("1", "a"), _far("2", "a")]
    row = hybrid_row("hybrid", examples, small, far, 0.95, frontier_model="claude-opus-5")
    assert row.escalation_rate == 0.5
    assert row.report.accuracy == 1.0
    assert row.report.input_tokens == 100  # only the escalated row's tokens
    assert row.as_row()["escalation_rate"] == 0.5


def test_markdown_table_has_one_line_per_system_plus_pending():
    rows = [
        {
            "system": "s",
            "accuracy": 0.841,
            "macro_f1": 0.8335,
            "p50_ms": 206,
            "p95_ms": 274,
            "cost_per_1k_usd": 0.0,
            "escalation_rate": None,
        }  # fmt: skip
    ]
    md = markdown_table(rows, pending=["claude-opus-5 (zero-shot)"])
    lines = md.splitlines()
    assert lines[0].startswith("| System |")
    assert "| s | 84.1% | 0.834 | 206 ms | 274 ms | $0.00 |  |" in lines
    assert lines[-1].startswith("| claude-opus-5 (zero-shot) | pending")
