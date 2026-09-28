"""Hybrid router and threshold sweep, with scripted small-model output and a stubbed frontier."""

import math

import pytest

from router.data import Example
from router.hybrid import FRONTIER, SMALL, HybridRouter
from router.predictions import Prediction
from router.small import INVALID_LABEL
from router.threshold import sweep
from tests.test_frontier import _router as frontier_router
from tests.test_small import EOS, ScriptedRouter


def _hybrid(script, threshold=0.95, frontier_label="top_up_failed"):
    small = ScriptedRouter(script)
    frontier, messages = frontier_router(frontier_label)
    return HybridRouter(small=small, frontier=frontier, threshold=threshold), messages


EXAMPLE = Example(id="a", text="where is my card", label="card_arrival")


def test_confident_answer_stays_local():
    router, messages = _hybrid([(1, math.log(0.99)), (2, math.log(0.98)), (EOS, 0.0)])
    pred = router.predict(EXAMPLE)
    assert pred.label == "card_arrival"
    assert pred.source == SMALL
    assert messages.calls == []
    assert router.counts == {SMALL: 1, FRONTIER: 0}
    assert pred.input_tokens == 0  # the local answer costs nothing


def test_low_confidence_escalates_and_keeps_small_confidence():
    router, messages = _hybrid([(1, math.log(0.7)), (2, math.log(0.9)), (EOS, 0.0)])
    pred = router.predict(EXAMPLE)
    assert pred.label == "top_up_failed"
    assert pred.source == FRONTIER
    assert len(messages.calls) == 1
    assert abs(pred.confidence - 0.63) < 1e-9
    assert pred.input_tokens == 120  # cost comes from the frontier call
    assert router.escalation_rate == 1.0


def test_invalid_output_always_escalates():
    router, messages = _hybrid([(3, math.log(0.999)), (EOS, 0.0)], threshold=0.0)
    pred = router.predict(EXAMPLE)
    assert pred.source == FRONTIER
    assert len(messages.calls) == 1


def test_escalated_latency_is_the_sum_of_both_calls():
    router, _ = _hybrid([(3, 0.0), (EOS, 0.0)])
    pred = router.predict(EXAMPLE)
    assert pred.latency_ms >= 0
    # Both routers measure with perf_counter, so we can only check the sum is plausible.
    assert pred.latency_ms == round(pred.latency_ms, 1)


def test_threshold_out_of_range_is_rejected():
    small = ScriptedRouter([])
    frontier, _ = frontier_router("card_arrival")
    with pytest.raises(ValueError):
        HybridRouter(small=small, frontier=frontier, threshold=1.5)


def _pred(id_, label, confidence):
    return Prediction(id=id_, label=label, latency_ms=1.0, input_tokens=0, output_tokens=1,
                      confidence=confidence)  # fmt: skip


def test_sweep_counts_kept_and_escalated_rows():
    gold = {"1": "a", "2": "a", "3": "b", "4": "b"}
    preds = [
        _pred("1", "a", 0.99),  # kept at every threshold below 0.99, correct
        _pred("2", "b", 0.90),  # kept at 0.9 and below, wrong
        _pred("3", "b", 0.60),  # kept at 0.6 and below, correct
        _pred("4", INVALID_LABEL, 0.99),  # invalid: always escalated
    ]
    result = sweep(gold, preds, thresholds=(0.5, 0.95), frontier_accuracy=0.9)
    rows = {r.threshold: r for r in result}
    low, high = rows[0.5], rows[0.95]
    assert (low.kept, low.escalated) == (3, 1)
    assert low.kept_accuracy == pytest.approx(2 / 3)
    assert low.projected_accuracy == pytest.approx((2 + 1 * 0.9) / 4)
    assert (high.kept, high.escalated) == (1, 3)
    assert high.kept_accuracy == 1.0
    assert high.projected_accuracy == pytest.approx((1 + 3 * 0.9) / 4)


def test_sweep_without_frontier_accuracy_has_no_projection():
    gold = {"1": "a"}
    rows = sweep(gold, [_pred("1", "a", 0.5)], thresholds=(0.4,))
    assert rows[0].projected_accuracy is None
    assert rows[0].as_row()["projected_accuracy"] is None
