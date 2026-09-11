"""Command line entry point: `uv run router --help`."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from router import data as data_mod
from router.evaluate import evaluate
from router.frontier import DEFAULT_EFFORT, DEFAULT_MODEL, FrontierRouter

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


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
    if limit:
        # Fixed-seed sample so repeated runs and other systems score the same rows.
        examples = random.Random(data_mod.SEED).sample(examples, min(limit, len(examples)))
        examples.sort(key=lambda e: e.id)

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
