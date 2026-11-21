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
        state = store.index_state()
        if state is None or not state["documents"]:
            raise RuntimeError("Index is empty. Run python -m search.ingest first.")
        app.state.store = store
        app.state.engine = Engine(store, Models())
        app.state.shed_reranking = os.getenv("SEARCH_SHED_RERANKING", "1") == "1"
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
    effective_mode: Mode
    index_profile: str
    page: int
    page_size: int
    total: int
    has_next: bool
    cache_hit: bool
    latency_ms: float
    results: list[Result]


@app.get("/health")
def health(request: Request):
    state = active_index(request)
    return {"status": "ok", "documents": state["documents"], "passages": state["passages"],
            "index_profile": state["profile"]}


def active_index(request):
    try:
        state = request.app.state.store.index_state()
    except (psycopg.Error, PoolTimeout) as exc:
        log.exception("Reading the active search index failed")
        raise HTTPException(status_code=503, detail="Search database is temporarily unavailable.") from exc
    if state is None:
        raise HTTPException(status_code=503, detail="No search index is active.")
    return state


@app.get("/benchmark", dependencies=[Depends(authorize)])
def benchmark(request: Request):
    path = Path(os.getenv("SEARCH_BENCHMARK_FILE", "benchmark/results.json"))
    if not path.is_file():
        return {"available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    state = active_index(request)
    if (report.get("corpus_sha256"), report.get("index_profile", "plain")) != (
        state["fingerprint"], state["profile"],
    ):
        return {"available": False}
    return {**report, "available": True}


@app.get("/search", response_model=SearchPage, dependencies=[Depends(authorize)])
def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH)],
    mode: Mode = "reranked",
    page: Annotated[int, Query(ge=1, le=CANDIDATE_DEPTH)] = 1,
    page_size: Annotated[int, Query(ge=1, le=20)] = 8,
    allow_fallback: bool = True,
):
    query = " ".join(q.split())
    if not query:
        raise HTTPException(status_code=422, detail="Enter a search query.")
    start = (page - 1) * page_size
    if start >= CANDIDATE_DEPTH:
        raise HTTPException(status_code=422, detail="Page exceeds the 50-result ranking window.")
    began = perf_counter()
    shed_reranking = request.app.state.shed_reranking and allow_fallback and page == 1
    try:
        for _ in range(2):
            state = active_index(request)
            hits, cache_hit, effective_mode = request.app.state.engine.search_for_serving(
                query, mode, shed_reranking=shed_reranking, index_generation=state["generation"],
            )
            if active_index(request)["generation"] == state["generation"]:
                break
        else:
            raise HTTPException(status_code=503, detail="Search index is updating. Retry shortly.")
    except (psycopg.Error, PoolTimeout) as exc:
        log.exception("Search database request failed")
        raise HTTPException(status_code=503, detail="Search database is temporarily unavailable.") from exc
    return {
        "query": query, "mode": mode, "effective_mode": effective_mode,
        "index_profile": state["profile"],
        "page": page, "page_size": page_size,
        "total": len(hits), "has_next": start + page_size < len(hits),
        "cache_hit": cache_hit, "latency_ms": round((perf_counter() - began) * 1000, 2),
        "results": hits[start:start + page_size],
    }
