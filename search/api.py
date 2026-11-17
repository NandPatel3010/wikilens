import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest
from time import perf_counter
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from psycopg_pool import PoolTimeout
from pydantic import BaseModel

from .engine import Engine, Mode
from .models import Models
from .settings import CANDIDATE_DEPTH, MAX_QUERY_LENGTH
from .store import Store, open_pool

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    pool = open_pool()
    try:
        pool.open(wait=True, timeout=30)
        store = Store(pool)
        app.state.counts = store.counts()
        app.state.fingerprint = store.fingerprint()
        if app.state.counts["documents"] == 0 or not app.state.fingerprint:
            raise RuntimeError("Index is empty. Run python -m search.ingest first.")
        app.state.engine = Engine(store, Models())
        yield
    finally:
        pool.close()


app = FastAPI(title="WikiLens", version="1.0.0", lifespan=lifespan)


def authorize(authorization: Annotated[str | None, Header()] = None):
    token = os.getenv("SEARCH_API_TOKEN", "")
    if token and not compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="Search API authentication required.")


class Result(BaseModel):
    id: str
    title: str
    text: str
    url: str
    score: float


class SearchPage(BaseModel):
    query: str
    mode: Mode
    page: int
    page_size: int
    total: int
    has_next: bool
    cache_hit: bool
    latency_ms: float
    results: list[Result]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", **request.app.state.counts}


@app.get("/benchmark", dependencies=[Depends(authorize)])
def benchmark(request: Request):
    path = Path(os.getenv("SEARCH_BENCHMARK_FILE", "benchmark/results.json"))
    if not path.is_file():
        return {"available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("corpus_sha256") != request.app.state.fingerprint:
        return {"available": False}
    return {**report, "available": True}


@app.get("/search", response_model=SearchPage, dependencies=[Depends(authorize)])
def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH)],
    mode: Mode = "reranked",
    page: Annotated[int, Query(ge=1, le=CANDIDATE_DEPTH)] = 1,
    page_size: Annotated[int, Query(ge=1, le=20)] = 8,
):
    query = " ".join(q.split())
    if not query:
        raise HTTPException(status_code=422, detail="Enter a search query.")
    start = (page - 1) * page_size
    if start >= CANDIDATE_DEPTH:
        raise HTTPException(status_code=422, detail="Page exceeds the 50-result ranking window.")
    began = perf_counter()
    try:
        hits, cache_hit = request.app.state.engine.search(query, mode)
    except (psycopg.Error, PoolTimeout) as exc:
        log.exception("Search database request failed")
        raise HTTPException(status_code=503, detail="Search database is temporarily unavailable.") from exc
    return {
        "query": query, "mode": mode, "page": page, "page_size": page_size,
        "total": len(hits), "has_next": start + page_size < len(hits),
        "cache_hit": cache_hit, "latency_ms": round((perf_counter() - began) * 1000, 2),
        "results": hits[start:start + page_size],
    }
