import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

from search.engine import Engine


@dataclass
class Hit:
    score: float


class Store:
    def dense(self, vector, depth):
        return [Hit(0.25), Hit(0.5)]


class Models:
    def __init__(self):
        self.reranking = Event()
        self.release = Event()

    def embed(self, texts):
        return [[0.0]]

    def rerank(self, query, hits):
        if query == "hold":
            self.reranking.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Test did not release the reranker")
        if query == "fail":
            raise ValueError("Reranker failed")
        return [10.0, 5.0]


class OverloadTests(unittest.TestCase):
    def setUp(self):
        self.models = Models()
        self.engine = Engine(Store(), self.models)

    def serving(self, query):
        return self.engine.search_for_serving(query, "reranked", shed_reranking=True)

    def test_busy_reranker_serves_semantic_and_keeps_mode_caches_separate(self):
        stripes = len(self.engine.query_locks)
        active = hash(("reranked", "hold")) % stripes
        query = next(
            f"query {i}" for i in range(1000)
            if hash(("reranked", f"query {i}")) % stripes != active
            and hash(("semantic", f"query {i}")) % stripes != active
        )
        cached_query = next(
            f"cached {i}" for i in range(1000)
            if hash(("reranked", f"cached {i}")) % stripes != active
        )
        self.serving(cached_query)
        with ThreadPoolExecutor(max_workers=2) as pool:
            active_request = pool.submit(self.serving, "hold")
            try:
                self.assertTrue(self.models.reranking.wait(timeout=2))
                hits, cached, mode = pool.submit(self.serving, query).result(timeout=2)
                self.assertEqual([hit.score for hit in hits], [0.25, 0.5])
                self.assertEqual((cached, mode), (False, "semantic"))
                self.assertTrue(self.engine.search(query, "semantic")[1])
                self.assertEqual(self.serving(cached_query)[1:], (True, "reranked"))
            finally:
                self.models.release.set()
            self.assertEqual(active_request.result(timeout=2)[2], "reranked")
        self.assertEqual(self.serving(query)[1:], (False, "reranked"))

    def test_failed_rerank_releases_capacity(self):
        with self.assertRaisesRegex(ValueError, "Reranker failed"):
            self.serving("fail")
        self.assertEqual(self.serving("next")[2], "reranked")

    def test_index_activation_uses_a_fresh_cache(self):
        def query(generation):
            return self.engine.search_for_serving(
                "same query", "semantic", shed_reranking=True, index_generation=generation,
            )

        self.assertFalse(query(3)[1])
        self.assertTrue(query(3)[1])
        self.assertFalse(query(4)[1])
