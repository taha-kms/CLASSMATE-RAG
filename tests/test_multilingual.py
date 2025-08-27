import os
import unittest

from rag.generation.prompting import build_grounded_messages, build_general_messages
from rag.pipeline import rag as rag_mod


class FakeRunner:
    def chat(self, messages, **kwargs):
        # simulate a "translation" that simply echoes with a marker while preserving [n]
        content = messages[-1]["content"]
        # If target is Italian, return a string containing [1] to test preservation
        return "tradotto [1]" if "Traduci" in messages[0]["content"] or "Translate" in messages[0]["content"] else content


class TestMultilingual(unittest.TestCase):
    def test_prompt_language_selection_en(self):
        msgs = build_grounded_messages(question="What is a stack?", contexts=[], forced_language=None, default_language="en")
        sys = msgs[0]["content"]
        usr = msgs[1]["content"]
        self.assertIn("reply entirely in the requested language".lower(), sys.lower())
        self.assertIn("Answer in English", usr)

    def test_prompt_language_selection_it(self):
        msgs = build_grounded_messages(question="Che cos'è una pila?", contexts=[], forced_language=None, default_language="auto")
        sys = msgs[0]["content"]
        usr = msgs[1]["content"]
        self.assertIn("Rispondi interamente in italiano".lower(), sys.lower())
        self.assertIn("Rispondi in italiano", usr)

    def test_general_prompt_it(self):
        msgs = build_general_messages(question="Definisci un grafo", language="it")
        self.assertIn("Rispondi interamente in italiano".lower(), msgs[0]["content"].lower())

    def test_translate_on_miss_preserves_citations(self):
        # simulate wrong-language answer: detected 'en' but target 'it'
        text = "This is in English [1]."
        out = rag_mod._translate_text(text, "it", runner=FakeRunner())
        self.assertIn("[1]", out)
        self.assertTrue(out.startswith("tradotto"))

    def test_needs_translation(self):
        self.assertTrue(rag_mod._needs_translation("This is English.", "it"))
        self.assertFalse(rag_mod._needs_translation("Questo è italiano.", "it"))


if __name__ == "__main__":
    unittest.main()
