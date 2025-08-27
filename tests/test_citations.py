import unittest

from rag.generation.post import enforce_citations


class TestCitations(unittest.TestCase):
    def test_removes_out_of_range(self):
        prov = ["a", "b", "c"]
        ans = "See [1] and [4] for details."
        out = enforce_citations(ans, prov)
        self.assertIn("[1]", out)
        self.assertNotIn("[4]", out)

    def test_compacts_adjacent(self):
        prov = ["a", "b"]
        ans = "Related work [1] [2] shows..."
        out = enforce_citations(ans, prov)
        self.assertIn("[1][2]", out)
        self.assertNotIn("[1] [2]", out)

    def test_sources_block(self):
        prov = ["src A", "src B"]
        ans = "Refs: [2]"
        out = enforce_citations(ans, prov, add_sources_block=True, sources_title="Sources")
        self.assertTrue(out.strip().endswith("[2] src B"))

    def test_handles_empty_answer(self):
        out = enforce_citations("", ["x"])
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
