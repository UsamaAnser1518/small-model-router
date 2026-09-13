"""Small-model tests use a fake tokenizer and a scripted token stream: no weights, no GPU."""

import math

from router.data import Example
from router.finetune import chat_record
from router.small import INVALID_LABEL, SmallRouter

LABELS = ["card_arrival", "top_up_failed"]
EOS = 999


class FakeTokenizer:
    eos_token_ids = {EOS}
    vocab = {1: "card", 2: "_arrival", 3: "banana"}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "|".join(m["content"] for m in messages) + "|assistant:"

    def decode(self, tokens):
        return "".join(self.vocab[t] for t in tokens)


class ScriptedRouter(SmallRouter):
    def __init__(self, script):
        super().__init__(LABELS, model="fake")
        self._tokenizer = FakeTokenizer()
        self._model = object()
        self.script = script
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        yield from self.script


def test_valid_label_with_confidence():
    router = ScriptedRouter([(1, math.log(0.9)), (2, math.log(0.8)), (EOS, 0.0)])
    pred = router.predict(Example(id="a", text="where is my card", label="card_arrival"))
    assert pred.label == "card_arrival"
    assert pred.output_tokens == 2
    assert abs(pred.confidence - 0.72) < 1e-9
    assert router.prompts[0].endswith("where is my card|assistant:")


def test_invalid_output_is_flagged_not_fuzzy_matched():
    router = ScriptedRouter([(3, math.log(0.5)), (EOS, 0.0)])
    pred = router.predict(Example(id="b", text="x", label="card_arrival"))
    assert pred.label == INVALID_LABEL
    assert abs(pred.confidence - 0.5) < 1e-9


def test_chat_record_puts_label_in_assistant_turn():
    rec = chat_record(Example(id="c", text="hello", label="top_up_failed"))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][1]["content"] == "hello"
    assert rec["messages"][2]["content"] == "top_up_failed"
