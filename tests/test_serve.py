"""The HTTP layer, exercised with a scripted small model and a stubbed frontier client."""

import math
from types import SimpleNamespace

import anthropic
import pytest
from fastapi.testclient import TestClient

from router.frontier import FrontierRouter
from router.hybrid import HybridRouter
from router.serve import create_app
from tests.test_frontier import LABELS, FakeMessages
from tests.test_small import EOS, ScriptedRouter

CONFIDENT = [(1, math.log(0.99)), (2, math.log(0.99)), (EOS, 0.0)]
UNSURE = [(1, math.log(0.5)), (2, math.log(0.5)), (EOS, 0.0)]


class BrokenMessages:
    def __init__(self, exc: Exception):
        self.exc = exc

    def create(self, **kwargs):
        raise self.exc


def _client(script, messages=None, threshold=0.95):
    small = ScriptedRouter(script)
    small.load = lambda: small  # no weights to load
    client = SimpleNamespace(messages=messages or FakeMessages("top_up_failed"))
    frontier = FrontierRouter(LABELS, client=client)
    hybrid = HybridRouter(small=small, frontier=frontier, threshold=threshold)
    return TestClient(create_app(hybrid))


def test_health_names_both_models():
    with _client(CONFIDENT) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["small_model"] == "fake"
    assert body["frontier"] == "claude-opus-5"


def test_confident_request_is_answered_locally():
    with _client(CONFIDENT) as c:
        res = c.post("/route", json={"text": "where is my card"})
        assert res.status_code == 200
        body = res.json()
        assert body["label"] == "card_arrival"
        assert body["source"] == "small"
        assert body["confidence"] == pytest.approx(0.9801)
        assert body["threshold"] == 0.95
        stats = c.get("/stats").json()
    assert stats == {"small": 1, "frontier": 0, "escalation_rate": 0.0, "threshold": 0.95}


def test_unsure_request_is_escalated():
    messages = FakeMessages("top_up_failed")
    with _client(UNSURE, messages) as c:
        body = c.post("/route", json={"text": "my top up failed"}).json()
        stats = c.get("/stats").json()
    assert body["label"] == "top_up_failed"
    assert body["source"] == "frontier"
    assert body["confidence"] == pytest.approx(0.25)
    assert len(messages.calls) == 1
    assert stats["escalation_rate"] == 1.0


@pytest.mark.parametrize(
    "exc",
    [
        anthropic.AuthenticationError.__new__(anthropic.AuthenticationError),
        TypeError("Could not resolve authentication method"),  # what the SDK raises with no key
    ],
)
def test_failed_escalation_is_a_502_not_a_guess(exc):
    with _client(UNSURE, BrokenMessages(exc)) as c:
        res = c.post("/route", json={"text": "my top up failed"})
    assert res.status_code == 502
    assert "escalation failed" in res.json()["detail"]


def test_empty_text_is_rejected():
    with _client(CONFIDENT) as c:
        assert c.post("/route", json={"text": ""}).status_code == 422
