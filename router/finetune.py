"""Export chat-format training data and run LoRA fine-tuning with mlx-lm.

The small model is trained as a plain instruction-following model: the user
turn is the customer message, the assistant turn is the label. With
`mask_prompt` the loss is computed only on the label tokens, so the model
learns to answer, not to predict customer text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from router.data import Example, read_jsonl

SYSTEM_PROMPT = (
    "You route customer-support messages for a retail bank. Reply with the routing label only."
)

MODELS = {
    "1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "3b": "mlx-community/Qwen2.5-3B-Instruct-bf16",
}


def chat_record(example: Example) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.text},
            {"role": "assistant", "content": example.label},
        ]
    }


def export_chat(data_dir: Path, out_dir: Path) -> dict[str, int]:
    """Write train.jsonl and valid.jsonl in the {"messages": [...]} format mlx-lm expects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for src, dst in (("train", "train"), ("validation", "valid")):
        examples = read_jsonl(data_dir / f"{src}.jsonl")
        with (out_dir / f"{dst}.jsonl").open("w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(chat_record(e), ensure_ascii=False) + "\n")
        counts[dst] = len(examples)
    return counts


def lora_command(
    model: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_layers: int = 16,
    max_seq_length: int = 256,
    steps_per_eval: int = 200,
    seed: int = 1518,
) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_path),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--num-layers", str(num_layers),
        "--max-seq-length", str(max_seq_length),
        "--steps-per-eval", str(steps_per_eval),
        "--steps-per-report", "50",
        "--save-every", str(steps_per_eval),
        "--mask-prompt",
        "--seed", str(seed),
    ]  # fmt: skip


def train(model: str, data_dir: Path, adapter_path: Path, iters: int, **kwargs) -> int:
    cmd = lora_command(model, data_dir, adapter_path, iters, **kwargs)
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "train_command.txt").write_text(" ".join(cmd) + "\n")
    return subprocess.call(cmd)
