# small-model-router

Route customer-support requests with a fine-tuned small model running locally, and escalate to a frontier model only when the small model is unsure. Every claim in this README comes from a script in this repo that you can rerun.

**Status: week 1, day 3.** Dataset pipeline, frontier baseline, evaluation harness, LoRA fine-tuning, the escalation threshold sweep, and an HTTP server that routes locally and escalates on low confidence are in place. The benchmark table that scores every system, including the hybrid, on the untouched test split lands on day 4.

## Why

Support routing is a high-volume, low-ambiguity task. Sending every message to a frontier model buys accuracy you may not need at a latency and cost you definitely notice. The question this repo answers with numbers: how small can the model get before accuracy drops, and what does the escalation threshold have to be to keep the frontier model's accuracy at a fraction of its cost?

## Dataset

[Banking77](https://huggingface.co/datasets/mteb/banking77): 13,083 real online-banking support queries labelled with one of 77 fine-grained intents. It is hard enough to be interesting (many intents differ by one detail) and small enough to fine-tune on a laptop.

| Split      | Rows  | Source                                   |
|------------|------:|------------------------------------------|
| train      | 8,994 | hub train, minus validation              |
| validation |   999 | 10% of hub train, fixed seed             |
| test       | 3,076 | hub test, untouched until the final table |

## Results so far

Both small models were fine-tuned for one epoch (1,124 steps, batch 8, LoRA on 16 layers) and scored on the full 999-row validation split. Latency is per message on an idle M4 Pro. The frontier row is pending an API key.

| System | Accuracy | Macro-F1 | p50 latency | p95 latency | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct + LoRA | 84.1% | 0.834 | 206 ms | 274 ms | 12 / 999 |
| Qwen2.5-3B-Instruct + LoRA   | 82.0% | 0.818 | 321 ms | 446 ms | 28 / 999 |
| Qwen2.5-1.5B-Instruct, untrained | 0% | 0 | 272 ms | 306 ms | 30 / 30 |
| Claude Opus 5, zero-shot | pending | | | | |

Two things worth noticing. The untrained base model scores zero because it cannot produce any of the 77 label strings, which is exactly why fine-tuning is needed for this task. And the 3B model is *worse* than the 1.5B after the same single epoch: its validation loss was still falling, so it is undertrained rather than incapable. [Issue #1](https://github.com/UsamaAnser1518/small-model-router/issues/1) tracks the second epoch and a learning-rate sweep.

### Accuracy climbs with training

Validation accuracy on 200 fixed rows at each saved checkpoint:

| Step | 1.5B | 3B |
|---:|---:|---:|
| 200 | 48.0% | 47.0% |
| 400 | 71.0% | 68.0% |
| 600 | 79.0% | 77.0% |
| 800 | 83.0% | 81.0% |
| 1000 | 80.5% | 81.5% |

### Confidence predicts correctness

The small model's confidence is the probability it assigns to the label it produced. On the 999 validation rows, that number tracks accuracy closely, which is what makes an escalation threshold viable:

| Confidence band | 1.5B rows | 1.5B accuracy | 3B rows | 3B accuracy |
|---|---:|---:|---:|---:|
| below 0.50 | 66 | 22.7% | 77 | 20.8% |
| 0.50 to 0.80 | 174 | 63.8% | 199 | 60.8% |
| 0.80 to 0.95 | 123 | 82.1% | 135 | 80.7% |
| 0.95 and above | 636 | 96.4% | 588 | 97.4% |

Read the last row as: when the 1.5B model is at least 95 percent sure, which happens on 64 percent of messages, it is right 96 percent of the time. The router sends the rest to the frontier model.

### Picking the escalation threshold

`router threshold` sweeps thresholds over the saved validation predictions, so choosing one costs no model runs. For the 1.5B model:

| Threshold | Kept locally | Accuracy on kept rows | Escalated |
|---:|---:|---:|---:|
| 0.50 | 930 (93.1%) | 88.7% | 6.9% |
| 0.60 | 901 (90.2%) | 89.6% | 9.8% |
| 0.70 | 827 (82.8%) | 92.4% | 17.2% |
| 0.80 | 759 (76.0%) | 94.1% | 24.0% |
| 0.90 | 636 (63.7%) | 96.4% | 36.3% |
| 0.95 | 636 (63.7%) | 96.4% | 36.3% |

The rows for 0.90 and 0.95 are identical, and so are 0.80 and 0.85, because the confidence is far coarser than it looks. The model runs in bf16, so every token log-probability comes out as a multiple of 0.125 nats, and the 999 confidences take only 16 distinct values: 1.0, 0.88, 0.78, 0.69, and so on down. In practice a threshold of 0.95 means "escalate unless the model is as sure as bf16 can express", which is 64 percent of messages at 96.4 percent accuracy. [Issue #2](https://github.com/UsamaAnser1518/small-model-router/issues/2) covers computing the log-probabilities in float32 and scoring all 77 labels, which would make the threshold a real dial instead of a switch.

The sweep also writes a projected hybrid accuracy when it knows the frontier model's accuracy on the same split (`--frontier-accuracy`, or `results/frontier-validation.json` if present). That column is a projection, not a measurement; day 4 measures the real thing.

## Quick start

```bash
uv sync
uv run router prepare              # downloads Banking77, writes data/*.jsonl and data/labels.json
uv run router frontier --limit 200 # zero-shot baseline with Claude on 200 fixed test rows
```

`frontier` needs Anthropic credentials: set `ANTHROPIC_API_KEY`, or log in with `ant auth login`. Predictions are cached under `results/cache/`, so interrupted runs resume for free and re-running a report costs nothing.

### Serve the router

```bash
uv run router threshold --size 1.5b        # sweep thresholds over saved predictions, no model runs
uv run router serve --size 1.5b --threshold 0.95
```

```console
$ curl -s localhost:8000/route -H 'content-type: application/json' \
    -d '{"text": "I still have not received my new card, when will it arrive?"}'
{"label":"card_arrival","source":"small","confidence":1.0,"threshold":0.95,"latency_ms":195.4}

$ curl -s localhost:8000/route -H 'content-type: application/json' \
    -d '{"text": "can you tell me about the weather"}'
{"label":"<whatever the frontier model answers>","source":"frontier","confidence":0.4724,"threshold":0.95,"latency_ms":1843.2}

$ curl -s localhost:8000/stats
{"small":1,"frontier":1,"escalation_rate":0.5,"threshold":0.95}
```

The second response is illustrative: the frontier answer and its latency depend on your API credentials. `source` says which model answered. `confidence` is always the small model's, even when the frontier model answered, so you can see how unsure the local model was. If the small model is unsure and the frontier call fails (no credentials, network down), the server returns 502 rather than serving the low-confidence guess as an answer. The same app is available as `router.serve:app` for uvicorn, configured through `ROUTER_SIZE`, `ROUTER_THRESHOLD`, `ROUTER_FRONTIER_MODEL`, and the `ROUTER_*_DIR` variables.

### Fine-tune the small models (Apple silicon)

```bash
uv run router export-chat                 # data/mlx/{train,valid}.jsonl in chat format
uv run router train --size 1.5b           # LoRA adapter -> adapters/1.5b, about one epoch
uv run router train --size 3b
uv run router small --size 1.5b           # score on 200 fixed validation rows
uv run router small --size 1.5b --no-adapter   # the untrained base model, for contrast
```

Training uses [mlx-lm](https://github.com/ml-explore/mlx-lm) and runs on the GPU of any Apple silicon Mac. The small model is trained as a chat model whose assistant turn is the bare label, with the loss masked to the label tokens. At inference the model's confidence is the probability it assigns to the label it produced, which is what the router thresholds on.

## Plan

| Day | Deliverable |
|-----|-------------|
| 1   | Data pipeline, frontier baseline with structured outputs, eval harness with accuracy, macro-F1, latency and cost. Done. |
| 2   | LoRA fine-tune of a 1.5B and a 3B model with MLX on Apple silicon; validation accuracy per checkpoint. Done. |
| 3   | FastAPI serving with a confidence threshold that escalates low-confidence requests to the frontier model. Done. |
| 4   | Benchmark harness: one command produces the accuracy, latency, and cost table for every system, including the hybrid router. |
| 5   | README leads with the table; CI regenerates the small-model numbers from committed adapters. |

## Layout

```
router/
  data.py       Banking77 download, deterministic split, JSONL export
  frontier.py   zero-shot classifier with a JSON-schema-constrained label enum, disk cache
  evaluate.py   accuracy, macro-F1, p50/p95 latency, cost from token usage
  finetune.py   chat-format export and the mlx-lm LoRA training command
  small.py      local inference with a confidence score, invalid outputs flagged
  hybrid.py     the router: small model first, frontier model when confidence is below threshold
  threshold.py  threshold sweep over saved predictions
  serve.py      FastAPI app: POST /route, GET /stats, GET /health
  predictions.py the one Prediction record every system produces
  cli.py        `router prepare|frontier|export-chat|train|small|checkpoints|threshold|serve`
adapters/       LoRA adapters written by `router train`
tests/          run without network or API keys
data/           generated by `router prepare` (JSONL is git-ignored, labels.json is committed)
results/        eval outputs; results/cache is git-ignored
```

## Design decisions

See [DECISIONS.md](DECISIONS.md).
