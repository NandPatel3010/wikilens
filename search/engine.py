from dataclasses import replace
from threading import Lock
from typing import Literal

from cachetools import TTLCache

from .settings import CANDIDATE_DEPTH

Mode = Literal["keyword", "semantic", "reranked"]
MODES = ("keyword", "semantic", "reranked")


class Engine:
    def __init__(self, store, models, *, cache_size=512, ttl=300):
        self.store = store
        self.models = models
        self.cache = TTLCache(maxsize=cache_size, ttl=ttl)
        self.cache_lock = Lock()
        # Stripes coalesce duplicate requests while permitting unrelated searches.
        self.query_locks = tuple(Lock() for _ in range(32))
        # The CPU cross-encoder processes one request at a time.
        self.rerank_slot = Lock()

    def rank(self, query, mode):
        hits, _ = self._rank(query, mode, shed_reranking=False)
        return hits

    def _rank(self, query, mode, *, shed_reranking):
        if mode == "keyword":
            return self.store.keyword(query, CANDIDATE_DEPTH), mode
        candidates = self.store.dense(self.models.embed([query])[0], CANDIDATE_DEPTH)
        if mode == "semantic" or not candidates:
            return candidates, mode
        if not self.rerank_slot.acquire(blocking=not shed_reranking):
            return candidates, "semantic"
        try:
            scores = self.models.rerank(query, candidates)
        finally:
            self.rerank_slot.release()
        rescored = [replace(hit, score=float(score))
                    for hit, score in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda hit: -hit.score), mode

    def search(self, query, mode):
        hits, cache_hit, _ = self.search_for_serving(query, mode, shed_reranking=False)
        return hits, cache_hit

    def search_for_serving(self, query, mode, *, shed_reranking, index_generation=None):
        normalized = " ".join(query.split())
        scope = () if index_generation is None else (index_generation,)
        key = (mode, normalized, *scope)
        with self.query_locks[hash(key) % len(self.query_locks)]:
            with self.cache_lock:
                cached = self.cache.get(key)
            if cached is not None:
                return cached, True, mode
            ranked, effective_mode = self._rank(normalized, mode, shed_reranking=shed_reranking)
            hits = tuple(ranked)
            with self.cache_lock:
                hits = self.cache.setdefault((effective_mode, normalized, *scope), hits)
            return hits, False, effective_mode
