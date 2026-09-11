"""Frontier router tests run without an API key by stubbing the Anthropic client."""

import json
from pathlib import Path
from types import SimpleNamespace

from router.data import Example
from router.frontier import FrontierRouter

LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed"]


class FakeMessages:
    def __init__(self, label: str):
        self.label = label
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            content=[SimpleNamespace(type="text", text=json.dumps({"label": self.label}))],
            usage=SimpleNamespace(input_tokens=120, output_tokens=8),
        )


def _router(label: str, cache_dir: Path | None = None):
    messages = FakeMessages(label)
    client = SimpleNamespace(messages=messages)
    return FrontierRouter(LABELS, cache_dir=cache_dir, client=client), messages


def test_predict_uses_structured_output_with_label_enum():
    router, messages = _router("card_arrival")
    pred = router.predict(Example(id="x", text="where is my card", label="card_arrival"))
    assert pred.label == "card_arrival"
    assert pred.input_tokens == 120
    call = messages.calls[0]
    assert call["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == LABELS
    assert call["output_config"]["effort"] == "low"
    assert "card_arrival" in call["messages"][0]["content"]


def test_predict_is_cached_on_disk(tmp_path: Path):
    router, messages = _router("top_up_failed", cache_dir=tmp_path)
    example = Example(id="y", text="top up did not work", label="top_up_failed")
    first = router.predict(example)
    second = router.predict(example)
    assert len(messages.calls) == 1
    assert first.cached is False and second.cached is True
    assert second.label == "top_up_failed"
