import math
import re
import unittest

from search.chunks import chunk_article
from search.dataset import sample_articles, title_key
from search.evaluate import latency_summary
from search.metrics import retrieval_metrics


class Tokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        return text.split() + (["[CLS]", "[SEP]"] if add_special_tokens else [])

    def num_special_tokens_to_add(self):
        return 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class RetrievalTests(unittest.TestCase):
    def test_chunks_preserve_text_and_overlap_inside_budget(self):
        chunks = list(chunk_article("Paper", "a b c d e f g", Tokenizer(), 7, overlap=1))
        self.assertEqual(chunks, [("a b c d", "Paper\na b c d"),
                                  ("d e f g", "Paper\nd e f g")])

    def test_metrics_use_document_ranks_and_fixed_precision_denominator(self):
        scores = retrieval_metrics(["wrong", "partial", "partial", "strong"],
                                   {"partial": 1, "strong": 2, "missed": 1}, k=5)
        self.assertEqual(scores["precision_at_5"], 0.4)
        self.assertAlmostEqual(scores["recall_at_5"], 2 / 3)
        self.assertEqual(scores["mrr_at_5"], 0.5)
        self.assertAlmostEqual(scores["ndcg_at_5"],
                               (1 / math.log2(3) + 3 / 2) / (3 + 1 / math.log2(3) + 1 / 2))

    def test_sampling_keeps_anchors_and_is_order_independent(self):
        rows = [
            {"id": str(n), "title": f"Article {n}", "text": "Article text.",
             "url": f"https://simple.wikipedia.org/wiki/Article_{n}"}
            for n in range(15)
        ]
        titles = {title_key("Article_14")}
        sample = sample_articles(rows, 5, titles)
        self.assertEqual(sample, sample_articles(reversed(rows), 5, titles))
        self.assertIn("14", {row["id"] for row in sample})
        self.assertEqual(len(sample), 5)

    def test_empty_ranking_and_latency(self):
        self.assertEqual(set(retrieval_metrics([], {"a": 1}).values()), {0.0})
        self.assertEqual(latency_summary([0.25]), {"p50_ms": 0.25, "p95_ms": 0.25})
