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

    def rank(self, query, mode):
        if mode == "keyword":
            return self.store.keyword(query, CANDIDATE_DEPTH)
        candidates = self.store.dense(self.models.embed([query])[0], CANDIDATE_DEPTH)
        if mode == "semantic" or not candidates:
            return candidates
        scores = self.models.rerank(query, candidates)
        rescored = [replace(hit, score=float(score))
                    for hit, score in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda hit: -hit.score)

    def search(self, query, mode):
        normalized = " ".join(query.split())
        key = (mode, normalized)
        with self.query_locks[hash(key) % len(self.query_locks)]:
            with self.cache_lock:
                cached = self.cache.get(key)
            if cached is not None:
                return cached, True
            hits = tuple(self.rank(normalized, mode))
            with self.cache_lock:
                self.cache[key] = hits
            return hits, False
