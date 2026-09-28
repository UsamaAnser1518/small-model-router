"""HTTP serving for the hybrid router.

    uv run router serve --size 1.5b --threshold 0.95
    curl -s localhost:8000/route -H 'content-type: application/json' \\
        -d '{"text": "where is my card"}'

The small model is loaded once at startup, on a dedicated thread, and every local
prediction runs on that same thread: MLX keeps its stream state per thread, and one
worker also serialises access to the GPU. Frontier calls run on the default threadpool
so a slow escalation never blocks local answers. The frontier client is built lazily
on the first escalation, so the server starts without Anthropic credentials and only
fails on the request that actually needs them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router import finetune
from router.data import Example
from router.frontier import DEFAULT_EFFORT, DEFAULT_MODEL, FrontierRouter
from router.hybrid import FRONTIER, SMALL, HybridRouter
from router.small import SmallRouter


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000, description="The customer message.")


class RouteResponse(BaseModel):
    label: str
    source: Literal["small", "frontier"]
    # The small model's confidence in its own answer, even when the frontier answered.
    confidence: float | None
    threshold: float
    latency_ms: float


class StatsResponse(BaseModel):
    small: int
    frontier: int
    escalation_rate: float
    threshold: float


def build_router(
    size: str = "1.5b",
    threshold: float = 0.95,
    data_dir: Path = Path("data"),
    adapters_dir: Path = Path("adapters"),
    frontier_model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    results_dir: Path = Path("results"),
) -> HybridRouter:
    labels = json.loads((data_dir / "labels.json").read_text())
    small = SmallRouter(labels, finetune.MODELS[size], adapter_path=adapters_dir / size)
    frontier = FrontierRouter(
        labels, model=frontier_model, effort=effort, cache_dir=results_dir / "cache"
    )
    return HybridRouter(small=small, frontier=frontier, threshold=threshold)


def router_from_env() -> HybridRouter:
    """Configuration for `uvicorn router.serve:app`; the CLI passes the same values."""
    return build_router(
        size=os.environ.get("ROUTER_SIZE", "1.5b"),
        threshold=float(os.environ.get("ROUTER_THRESHOLD", "0.95")),
        data_dir=Path(os.environ.get("ROUTER_DATA_DIR", "data")),
        adapters_dir=Path(os.environ.get("ROUTER_ADAPTERS_DIR", "adapters")),
        frontier_model=os.environ.get("ROUTER_FRONTIER_MODEL", DEFAULT_MODEL),
        effort=os.environ.get("ROUTER_FRONTIER_EFFORT", DEFAULT_EFFORT),
    )


def create_app(router: HybridRouter | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = router or router_from_env()
        app.state.inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        await asyncio.get_running_loop().run_in_executor(
            app.state.inference, app.state.router.small.load
        )
        try:
            yield
        finally:
            app.state.inference.shutdown(wait=True)

    app = FastAPI(
        title="small-model-router",
        summary="Route support requests locally; escalate to a frontier model when unsure.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        r: HybridRouter = app.state.router
        return {"status": "ok", "small_model": r.small.model_name, "frontier": r.frontier.model}

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        r: HybridRouter = app.state.router
        return StatsResponse(
            small=r.counts[SMALL],
            frontier=r.counts[FRONTIER],
            escalation_rate=round(r.escalation_rate, 4),
            threshold=r.threshold,
        )

    @app.post("/route", response_model=RouteResponse)
    async def route(request: RouteRequest) -> RouteResponse:
        r: HybridRouter = app.state.router
        loop = asyncio.get_running_loop()
        example = _example(request.text)
        prediction = await loop.run_in_executor(app.state.inference, r.predict_local, example)
        if r.should_escalate(prediction):
            try:
                prediction = await loop.run_in_executor(None, r.escalate, example, prediction)
            except (anthropic.AnthropicError, TypeError) as exc:
                # The small model was unsure and the frontier call failed: say so rather
                # than return a low-confidence guess as if it were an answer. The SDK
                # raises TypeError, not its own error class, when no credentials are set.
                raise HTTPException(status_code=502, detail=f"escalation failed: {exc}") from exc
        r.record(prediction)
        return RouteResponse(
            label=prediction.label,
            source=prediction.source or SMALL,
            confidence=prediction.confidence,
            threshold=r.threshold,
            latency_ms=prediction.latency_ms,
        )

    return app


def _example(text: str) -> Example:
    # Live requests have no gold label. The id only matters for the frontier cache,
    # and hashing the text means a repeated message is served from cache.
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Example(id=f"live-{digest}", text=text, label="")


app = create_app()
