"""Settings the pipeline reads actually exist on Config.

These five were reached through getattr(cfg, name, fallback) against a
Config that never declared them, so the fallback always won and the typed
config path was dead.
"""

import pytest

from rag.config import load_config


PIPELINE_SETTINGS = [
    "enable_neighbor_expansion",
    "neighbor_radius",
    "doc_diversity_cap",
    "strict_citations",
    "append_sources_block",
    "translate_on_miss",
]


@pytest.mark.parametrize("name", PIPELINE_SETTINGS)
def test_setting_is_declared_on_config(name):
    assert hasattr(load_config(reload=True), name), f"Config is missing {name}"


def test_defaults_match_the_documented_behaviour(monkeypatch):
    for name in PIPELINE_SETTINGS:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.delenv("EMB_CACHE_DIR", raising=False)
    cfg = load_config(reload=True)

    assert cfg.enable_neighbor_expansion is True
    assert cfg.neighbor_radius == 1
    assert cfg.doc_diversity_cap == 3
    # Citation and translation post-processing stay opt-in.
    assert cfg.strict_citations is False
    assert cfg.append_sources_block is False
    assert cfg.translate_on_miss is False


@pytest.mark.parametrize(
    "env_name, attr, raw, expected",
    [
        ("STRICT_CITATIONS", "strict_citations", "true", True),
        ("STRICT_CITATIONS", "strict_citations", "1", True),
        ("STRICT_CITATIONS", "strict_citations", "no", False),
        ("TRANSLATE_ON_MISS", "translate_on_miss", "yes", True),
        ("APPEND_SOURCES_BLOCK", "append_sources_block", "on", True),
        ("ENABLE_NEIGHBOR_EXPANSION", "enable_neighbor_expansion", "false", False),
        ("NEIGHBOR_RADIUS", "neighbor_radius", "3", 3),
        ("DOC_DIVERSITY_CAP", "doc_diversity_cap", "5", 5),
    ],
)
def test_environment_overrides_reach_config(monkeypatch, env_name, attr, raw, expected):
    monkeypatch.setenv(env_name, raw)
    assert getattr(load_config(reload=True), attr) == expected


def test_a_non_numeric_value_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("NEIGHBOR_RADIUS", "not-a-number")
    assert load_config(reload=True).neighbor_radius == 1
