from router.data import Example
from router.evaluate import evaluate, macro_f1
from router.predictions import Prediction


def test_macro_f1_perfect():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_macro_f1_all_wrong():
    assert macro_f1(["a", "b"], ["b", "a"]) == 0.0


def test_evaluate_report():
    examples = [Example(id=str(i), text="t", label="a" if i < 3 else "b") for i in range(4)]
    predictions = [
        Prediction(id="0", label="a", latency_ms=100, input_tokens=10, output_tokens=5),
        Prediction(id="1", label="a", latency_ms=200, input_tokens=10, output_tokens=5),
        Prediction(id="2", label="b", latency_ms=300, input_tokens=10, output_tokens=5),
        Prediction(
            id="3", label="b", latency_ms=400, input_tokens=10, output_tokens=5, cached=True
        ),
    ]
    report = evaluate(examples, predictions, "claude-opus-5")
    assert report.n == 4
    assert report.accuracy == 0.75
    assert report.latency_p50_ms == 200  # cached rows excluded from latency
    assert report.input_tokens == 40 and report.output_tokens == 20
    assert abs(report.cost_usd - (40 * 5 + 20 * 25) / 1e6) < 1e-9
    assert report.confusions == [("a", "b", 1)]


def test_confidence_buckets_group_by_band():
    from router.evaluate import confidence_buckets

    examples = [Example(id=str(i), text="t", label="a") for i in range(4)]
    predictions = [
        Prediction(
            id="0", label="a", latency_ms=1, input_tokens=0, output_tokens=1, confidence=0.99
        ),
        Prediction(
            id="1", label="b", latency_ms=1, input_tokens=0, output_tokens=1, confidence=0.97
        ),
        Prediction(
            id="2", label="a", latency_ms=1, input_tokens=0, output_tokens=1, confidence=0.3
        ),
        Prediction(
            id="3", label="a", latency_ms=1, input_tokens=0, output_tokens=1, confidence=None
        ),
    ]
    rows = confidence_buckets(examples, predictions)
    assert rows == [
        {"band": "[0.00, 0.50)", "n": 1, "accuracy": 1.0},
        {"band": "[0.95, 1.00)", "n": 2, "accuracy": 0.5},
    ]
