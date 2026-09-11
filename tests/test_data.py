from pathlib import Path

from router.data import Example, labels_from, read_jsonl, split, write_jsonl


def _examples(n: int, prefix: str = "train") -> list[Example]:
    return [
        Example(id=f"{prefix}-{i}", text=f"message {i}", label=f"label_{i % 3}") for i in range(n)
    ]


def test_split_is_deterministic_and_disjoint():
    train = _examples(100)
    test = _examples(10, "test")
    a = split(train, test)
    b = split(train, test)
    assert a == b
    assert len(a["validation"]) == 10
    assert len(a["train"]) == 90
    assert not {e.id for e in a["train"]} & {e.id for e in a["validation"]}
    assert a["test"] == test


def test_jsonl_round_trip(tmp_path: Path):
    examples = _examples(5)
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, examples) == 5
    assert read_jsonl(path) == examples


def test_labels_sorted_unique():
    assert labels_from(_examples(10)) == ["label_0", "label_1", "label_2"]
