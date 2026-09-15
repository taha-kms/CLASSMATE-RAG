"""Route types, prompts and the model registry.

Pure Python, but previously unreachable: rag.routing imported the
embedding-backed classifier and the llama.cpp loader on the way in.
"""

import pytest

from rag.config import load_config
from rag.routing import DEFAULT_ROUTE, ROUTES, get_model_spec, system_prompt_for


def test_the_default_route_is_one_of_the_declared_routes():
    assert DEFAULT_ROUTE in ROUTES


@pytest.mark.parametrize("route", list(ROUTES))
def test_every_route_has_a_prompt_in_both_languages(route):
    for language in ("en", "it"):
        prompt = system_prompt_for(route, language=language)
        assert isinstance(prompt, str) and prompt.strip()


def test_prompts_differ_between_routes():
    prompts = {system_prompt_for(r, language="en") for r in ROUTES}
    assert len(prompts) > 1, "routing is pointless if every route shares one prompt"


def _point_all_routes_at(monkeypatch, path):
    for var in ("MATH", "CODE", "TRANSLATION", "DEFAULT"):
        monkeypatch.setenv(f"ROUTE_{var}_MODEL_PATH", str(path))


@pytest.mark.parametrize("route", list(ROUTES))
def test_a_present_model_file_resolves_to_a_spec_for_that_route(tmp_path, monkeypatch, route):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"not really a model")
    _point_all_routes_at(monkeypatch, gguf)

    spec = get_model_spec(route, cfg=load_config(reload=True))

    assert spec.route == route
    assert spec.model_path == gguf
    assert spec.n_ctx > 0


def test_a_missing_specialist_falls_back_to_the_default_model(tmp_path, monkeypatch):
    default_gguf = tmp_path / "default.gguf"
    default_gguf.write_bytes(b"not really a model")
    monkeypatch.setenv("ROUTE_DEFAULT_MODEL_PATH", str(default_gguf))
    monkeypatch.setenv("ROUTE_MATH_MODEL_PATH", str(tmp_path / "absent.gguf"))

    spec = get_model_spec("math", cfg=load_config(reload=True))

    # Demoted, and says so, so the caller can log which model actually ran.
    assert spec.route == DEFAULT_ROUTE
    assert spec.model_path == default_gguf


def test_fallback_can_be_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTE_MATH_MODEL_PATH", str(tmp_path / "absent.gguf"))
    with pytest.raises(FileNotFoundError):
        get_model_spec("math", cfg=load_config(reload=True), fallback_to_default=False)


def test_a_missing_default_model_is_reported_clearly(tmp_path, monkeypatch):
    _point_all_routes_at(monkeypatch, tmp_path / "absent.gguf")
    with pytest.raises(FileNotFoundError) as excinfo:
        get_model_spec("math", cfg=load_config(reload=True))
    assert "default" in str(excinfo.value).lower()
