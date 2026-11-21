import os
import unittest


@unittest.skipUnless(os.environ.get("WIKILENS_TEST_RELEASE") == "1", "Requires the benchmark CI database")
class IndexReleaseTests(unittest.TestCase):
    def test_bad_candidate_preserves_active_index_and_can_restore_previous(self):
        from search.index_release import activate, index_update, retain_previous, rollback
        from search.store import Store, open_pool

        pool = open_pool()
        self.addCleanup(pool.close)
        pool.open(wait=True, timeout=30)
        store = Store(pool)
        active = store.index_state()
        self.assertEqual(active["profile"], "contextual")
        with self.assertRaisesRegex(ValueError, "Index validation failed"):
            with index_update() as connection:
                retain_previous(connection)
                connection.execute("DELETE FROM evidence.passages")
                activate(connection, "invalid", "plain")
        self.assertEqual(store.index_state(), active)
        self.assertEqual(store.counts()["passages"], active["passages"])

        rollback()
        restored = store.index_state()
        self.assertEqual(restored["profile"], "plain")
        self.assertEqual(restored["fingerprint"], active["fingerprint"])
        self.assertEqual(restored["generation"], active["generation"] + 1)
        self.assertEqual(store.counts()["passages"], restored["passages"])
