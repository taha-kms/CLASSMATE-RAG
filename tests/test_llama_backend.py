"""The llama.cpp plumbing shared by the runner and the sticky loader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.generation import llama_backend


def test_require_llama_names_the_missing_package():
    with patch.object(llama_backend, "Llama", None):
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            llama_backend.require_llama()


def test_load_llama_reports_a_missing_model_file(tmp_path):
    with patch.object(llama_backend, "Llama", MagicMock()):
        with pytest.raises(FileNotFoundError):
            llama_backend.load_llama(tmp_path / "absent.gguf")


def test_load_llama_passes_the_knobs_through(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    fake = MagicMock()

    with patch.object(llama_backend, "Llama", fake):
        llama_backend.load_llama(gguf, n_ctx=2048, n_gpu_layers=12, seed=7, verbose=True)

    kwargs = fake.call_args.kwargs
    assert kwargs["n_ctx"] == 2048
    assert kwargs["n_gpu_layers"] == 12
    assert kwargs["seed"] == 7
    assert kwargs["verbose"] is True
    assert Path(kwargs["model_path"]) == gguf


def test_chat_completion_returns_the_assistant_text():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "  hello  "}}]}
    assert llama_backend.chat_completion(llm, [{"role": "user", "content": "hi"}]) == "hello"


@pytest.mark.parametrize("bad", [{}, {"choices": []}, {"choices": [{}]}, None])
def test_an_unexpected_completion_shape_yields_an_empty_answer(bad):
    # Better an empty answer than a KeyError from deep inside the pipeline.
    llm = MagicMock()
    llm.create_chat_completion.return_value = bad
    assert llama_backend.chat_completion(llm, [{"role": "user", "content": "hi"}]) == ""


def test_generation_parameters_reach_the_model():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}

    llama_backend.chat_completion(
        llm,
        [{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.9,
        top_p=0.5,
        repeat_penalty=1.2,
        stop=["</s>"],
    )

    kwargs = llm.create_chat_completion.call_args.kwargs
    assert (kwargs["max_tokens"], kwargs["temperature"]) == (128, 0.9)
    assert (kwargs["top_p"], kwargs["repeat_penalty"]) == (0.5, 1.2)
    assert kwargs["stop"] == ["</s>"]
