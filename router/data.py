"""Banking77 loading, deterministic splits, and JSONL export.

Banking77 is 13,083 real customer-support queries labelled with one of 77
intents. We use the `mteb/banking77` mirror because the original PolyAI repo
ships a loading script, which recent versions of `datasets` refuse to run.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DATASET_ID = "mteb/banking77"
SEED = 1518

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "label": self.label}


def load_raw() -> tuple[list[Example], list[Example]]:
    """Return (train, test) as loaded from the hub, in hub order."""
    from datasets import load_dataset  # imported lazily: slow and heavy

    ds = load_dataset(DATASET_ID)
    train = [
        Example(id=f"train-{i}", text=row["text"], label=row["label_text"])
        for i, row in enumerate(ds["train"])
    ]
    test = [
        Example(id=f"test-{i}", text=row["text"], label=row["label_text"])
        for i, row in enumerate(ds["test"])
    ]
    return train, test


def split(
    train: list[Example], test: list[Example], validation_fraction: float = 0.1
) -> dict[Split, list[Example]]:
    """Carve a validation set out of train with a fixed seed. Test is never touched."""
    rng = random.Random(SEED)
    shuffled = list(train)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * validation_fraction)
    return {
        "validation": sorted(shuffled[:n_val], key=lambda e: e.id),
        "train": sorted(shuffled[n_val:], key=lambda e: e.id),
        "test": list(test),
    }


def labels_from(examples: Iterable[Example]) -> list[str]:
    return sorted({e.label for e in examples})


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]


def prepare(data_dir: Path) -> dict[Split, int]:
    """Download, split, and write train/validation/test JSONL plus labels.json."""
    train, test = load_raw()
    parts = split(train, test)
    counts: dict[Split, int] = {}
    for name, examples in parts.items():
        counts[name] = write_jsonl(data_dir / f"{name}.jsonl", examples)
    labels = labels_from(train)
    (data_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    return counts
