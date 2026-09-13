"""Command line entry point: `uv run router --help`."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from router import data as data_mod
from router import finetune
from router.evaluate import confidence_buckets, evaluate
from router.frontier import DEFAULT_EFFORT, DEFAULT_MODEL, FrontierRouter
from router.small import SmallRouter

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
ADAPTERS_DIR = Path("adapters")


@app.command()
def prepare(data_dir: Path = DATA_DIR) -> None:
    """Download Banking77 and write train/validation/test JSONL under DATA_DIR."""
    counts = data_mod.prepare(data_dir)
    for name, n in counts.items():
        console.print(f"[green]{name:11}[/] {n:>6} examples")
    console.print(f"labels written to {data_dir / 'labels.json'}")


@app.command()
def frontier(
    split: str = "test",
    limit: int = typer.Option(200, help="Number of examples to score; 0 means the whole split."),
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Score the frontier model zero-shot on a split and write results/frontier-<split>.json."""
    examples = data_mod.read_jsonl(data_dir / f"{split}.jsonl")
    labels = json.loads((data_dir / "labels.json").read_text())
    examples = _sample(examples, limit)

    router = FrontierRouter(labels, model=model, effort=effort, cache_dir=results_dir / "cache")
    predictions = [router.predict(e) for e in track(examples, description=f"{model} on {split}")]
    report = evaluate(examples, predictions, model)

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"frontier-{split}.json"
    out.write_text(
        json.dumps(
            {
                "model": model,
                "effort": effort,
                "report": report.as_row(f"{model} (zero-shot, effort={effort})"),
                "confusions": report.confusions,
                "predictions": [asdict(p) for p in predictions],
            },
            indent=2,
        )
    )
    _print_report(report.as_row(model), report.confusions)
    console.print(f"written to {out}")


@app.command()
def export_chat(data_dir: Path = DATA_DIR) -> None:
    """Write data/mlx/{train,valid}.jsonl in the chat format mlx-lm trains on."""
    counts = finetune.export_chat(data_dir, data_dir / "mlx")
    for name, n in counts.items():
        console.print(f"[green]{name:6}[/] {n:>6} chat records")


@app.command()
def train(
    size: str = typer.Option("1.5b", help="One of: " + ", ".join(finetune.MODELS)),
    iters: int = typer.Option(1100, help="Optimizer steps. ~1 epoch at batch 8 is 1124."),
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_layers: int = 16,
    data_dir: Path = DATA_DIR,
    adapters_dir: Path = ADAPTERS_DIR,
) -> None:
    """LoRA fine-tune a small model with mlx-lm. Apple silicon only."""
    model = finetune.MODELS[size]
    adapter_path = adapters_dir / size
    console.print(f"training [bold]{model}[/] -> {adapter_path}")
    code = finetune.train(
        model,
        data_dir / "mlx",
        adapter_path,
        iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_layers=num_layers,
    )
    raise typer.Exit(code)


@app.command()
def small(
    size: str = "1.5b",
    split: str = "validation",
    limit: int = typer.Option(200, help="Number of examples to score; 0 means the whole split."),
    adapter: bool = typer.Option(
        True, help="Load the LoRA adapter; --no-adapter scores the base model."
    ),
    data_dir: Path = DATA_DIR,
    adapters_dir: Path = ADAPTERS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Score a small model on a split and write results/small-<size>-<split>.json."""
    examples = _sample(data_mod.read_jsonl(data_dir / f"{split}.jsonl"), limit)
    labels = json.loads((data_dir / "labels.json").read_text())
    adapter_path = adapters_dir / size if adapter else None
    router = SmallRouter(labels, finetune.MODELS[size], adapter_path=adapter_path).load()
    name = f"{finetune.MODELS[size]}" + (" + LoRA" if adapter else " (base)")
    predictions = [router.predict(e) for e in track(examples, description=f"{size} on {split}")]
    report = evaluate(examples, predictions, model="local")

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"small-{size}{'' if adapter else '-base'}-{split}.json"
    out.write_text(
        json.dumps(
            {
                "model": finetune.MODELS[size],
                "adapter": str(adapter_path) if adapter_path else None,
                "report": report.as_row(name),
                "confusions": report.confusions,
                "confidence_bands": confidence_buckets(examples, predictions),
                "predictions": [asdict(p) for p in predictions],
            },
            indent=2,
        )
    )
    _print_report(report.as_row(name), report.confusions)
    bt = Table(title="Accuracy by confidence band")
    for key in ("band", "n", "accuracy"):
        bt.add_column(key, justify="right")
    for row in confidence_buckets(examples, predictions):
        bt.add_row(*(str(v) for v in row.values()))
    console.print(bt)
    console.print(f"written to {out}")


@app.command()
def checkpoints(
    size: str = "1.5b",
    limit: int = 200,
    data_dir: Path = DATA_DIR,
    adapters_dir: Path = ADAPTERS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Score every saved checkpoint of an adapter on the same validation rows."""
    adapter_path = adapters_dir / size
    examples = _sample(data_mod.read_jsonl(data_dir / "validation.jsonl"), limit)
    labels = json.loads((data_dir / "labels.json").read_text())
    files = sorted(adapter_path.glob("*_adapters.safetensors"))
    if not files:
        raise typer.BadParameter(f"no checkpoints under {adapter_path}")

    rows = []
    for ckpt in files:
        step = int(ckpt.name.split("_")[0])
        # mlx-lm loads <dir>/adapters.safetensors, so stage each checkpoint under that name.
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            shutil.copy(adapter_path / "adapter_config.json", staged / "adapter_config.json")
            shutil.copy(ckpt, staged / "adapters.safetensors")
            router = SmallRouter(labels, finetune.MODELS[size], adapter_path=staged).load()
            preds = [router.predict(e) for e in track(examples, description=f"step {step}")]
        report = evaluate(examples, preds, model="local")
        invalid = sum(p.label == "<invalid>" for p in preds)
        rows.append(
            {
                "step": step,
                "accuracy": round(report.accuracy, 4),
                "macro_f1": round(report.macro_f1, 4),
                "invalid": invalid,
                "p50_ms": report.latency_p50_ms,
            }
        )
        console.print(rows[-1])

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"checkpoints-{size}.json"
    summary = {"model": finetune.MODELS[size], "n": len(examples), "rows": rows}
    out.write_text(json.dumps(summary, indent=2))
    table = Table(title=f"{finetune.MODELS[size]}: validation accuracy per checkpoint")
    for key in rows[0]:
        table.add_column(key, justify="right")
    for row in rows:
        table.add_row(*(str(v) for v in row.values()))
    console.print(table)
    console.print(f"written to {out}")


def _sample(examples: list[data_mod.Example], limit: int) -> list[data_mod.Example]:
    """Fixed-seed sample so every system is scored on the same rows."""
    if not limit or limit >= len(examples):
        return examples
    chosen = random.Random(data_mod.SEED).sample(examples, limit)
    return sorted(chosen, key=lambda e: e.id)


def _print_report(row: dict, confusions: list[tuple[str, str, int]]) -> None:
    table = Table(title="Results")
    for key in row:
        table.add_column(key)
    table.add_row(*(str(v) for v in row.values()))
    console.print(table)
    if confusions:
        ct = Table(title="Top confusions (gold -> predicted)")
        ct.add_column("gold")
        ct.add_column("predicted")
        ct.add_column("count", justify="right")
        for gold, pred, n in confusions:
            ct.add_row(gold, pred, str(n))
        console.print(ct)


if __name__ == "__main__":
    app()
