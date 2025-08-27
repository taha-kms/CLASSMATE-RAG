import tempfile
from pathlib import Path
import unittest

from rag.loaders import infer_doc_type_from_path, load_document_by_type


class TestLoaders(unittest.TestCase):
    def test_html_readable(self):
        html = """<!doctype html><html><head><title>T</title></head>
        <body><article><h1>Header</h1><p>Hello <b>world</b>!</p></article></body></html>"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.html"
            p.write_text(html, encoding="utf-8")
            self.assertEqual(infer_doc_type_from_path(p), "html")
            pages = load_document_by_type(p, "html")
            self.assertTrue(len(pages) >= 1)
            self.assertIn("Hello", pages[0][1])

    def test_csv_bullets(self):
        csv_text = "name,age\nAlice,20\nBob,21\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.csv"
            p.write_text(csv_text, encoding="utf-8")
            self.assertEqual(infer_doc_type_from_path(p), "csv")
            pages = load_document_by_type(p, "csv")
            text = "\n".join(t for _, t in pages)
            self.assertIn("- name: Alice; age: 20", text)
            self.assertIn("- name: Bob; age: 21", text)

    def test_epub_inference(self):
        # We only assert inference here; building a real EPUB is verbose.
        self.assertEqual(infer_doc_type_from_path("book.epub"), "epub")


if __name__ == "__main__":
    unittest.main()
