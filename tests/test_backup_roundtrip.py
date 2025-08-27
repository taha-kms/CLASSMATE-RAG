import json
import os
import tempfile
import unittest
from pathlib import Path

from rag.admin.backup import dump_index, restore_dump
from rag.pipeline import index_stats


class TestBackupRoundtrip(unittest.TestCase):
    def test_dump_then_restore_counts_nonzero(self):
        # Skip if indexes are empty
        stats = index_stats()
        if stats.get("bm25", 0) <= 0 and stats.get("vectors", 0) <= 0:
            self.skipTest("Indexes empty; add a sample document first.")

        with tempfile.TemporaryDirectory() as td:
            dump_file = Path(td) / "corpus.jsonl"
            # Dump current index (without embedding checksums to keep it fast)
            n = dump_index(dump_file, include_embedding_checksum=False, batch_size=8)
            self.assertGreaterEqual(n, 1)

            # Restore into the same indexes (idempotent upsert)
            restored = restore_dump(dump_file, batch_size=8)
            self.assertGreaterEqual(restored, 1)


if __name__ == "__main__":
    unittest.main()
