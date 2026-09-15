"""
Lock in the LlamaCppRunner public surface:
- no-arg constructible
- exposes .chat(messages, **kw) returning the assistant text
- exposes .generate(prompt, **kw) for backward compatibility

Llama is stubbed out throughout, so these run without llama-cpp-python
installed. That used to be impossible: the module imported llama_cpp at
top level, so the whole file skipped in CI and the signature it claims to
lock in was never actually checked there.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.generation.llama_cpp_runner import LlamaCppRunner


def _patched_runner():
    """Build a LlamaCppRunner with the heavy bits stubbed out."""
    fake_llama = MagicMock()
    fake_llama.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    fake_llama.return_value = {"choices": [{"text": "raw-ok"}]}

    return patch.multiple(
        "rag.generation.llama_cpp_runner",
        Llama=MagicMock(return_value=fake_llama),
        ensure_llama_model_available=MagicMock(return_value=Path("/tmp/dummy.gguf")),
    ), fake_llama


def test_runner_constructs_without_args_and_chat_works():
    patcher, fake_llama = _patched_runner()
    with patcher, patch("pathlib.Path.exists", return_value=True):
        r = LlamaCppRunner()  # no args -> must not raise
        assert r.model is fake_llama
        out = r.chat([{"role": "user", "content": "hi"}])
        assert out == "ok"
        fake_llama.create_chat_completion.assert_called_once()


def test_runner_generate_still_works():
    patcher, fake_llama = _patched_runner()
    with patcher, patch("pathlib.Path.exists", return_value=True):
        r = LlamaCppRunner()
        out = r.generate("hello")
        assert out == "raw-ok"


def test_a_clear_error_when_llama_cpp_is_not_installed():
    with patch("rag.generation.llama_cpp_runner.Llama", None):
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            LlamaCppRunner(model_path="/tmp/whatever.gguf")


def test_the_module_imports_without_llama_cpp():
    # The guarded import is what lets CI exercise everything above.
    import rag.generation.llama_cpp_runner as mod

    assert hasattr(mod, "Llama")
