import unittest

from scripts.compare_benchmarks import compare


def report(p95, profile="plain"):
    return {
        "corpus_sha256": "corpus", "queries": 42, "documents": 5000, "candidate_depth": 50,
        "embedding_model": "pinned embedder", "reranker": "pinned reranker", "unmatched_questions": [],
        "index_profile": profile,
        "modes": {"reranked": {"recall_at_10": 0.9, "uncached": {"p95_ms": p95}}},
    }


class BenchmarkComparisonTests(unittest.TestCase):
    def test_latency_gate_flags_a_regression_and_allows_small_variation(self):
        self.assertEqual(compare(report(100), report(125), True)[1], [])
        self.assertEqual(len(compare(report(100), report(126), True)[1]), 1)
        self.assertEqual(compare(report(20), report(34), True)[1], [])

    def test_profile_changes_report_quality_without_latency_gate(self):
        table, failures = compare(report(100), report(200, "contextual"), False)
        self.assertIn("plain → contextual", table)
        self.assertEqual(failures, [])
        with self.assertRaisesRegex(ValueError, "same index profile"):
            compare(report(100), report(200, "contextual"), True)

    def test_comparison_rejects_a_different_corpus(self):
        candidate = report(100) | {"corpus_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "corpus_sha256"):
            compare(report(100), candidate, True)
