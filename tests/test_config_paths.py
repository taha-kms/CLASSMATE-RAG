"""Data paths resolve against the project root, not the working directory.

Running the CLI from anywhere other than the repo root used to create a
fresh, empty index next to wherever the process started, instead of finding
the real one.
"""

from pathlib import Path

from rag.config import load_config, resolve_data_path


def test_relative_paths_resolve_against_the_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_data_path("./indexes/bm25")

    assert resolved.is_absolute()
    assert (resolved.parent.parent / "pyproject.toml").is_file()
    assert tmp_path not in resolved.parents


def test_the_same_relative_path_resolves_identically_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_data_path("./indexes/bm25")

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from_root = resolve_data_path("./indexes/bm25")

    assert from_tmp == from_root


def test_absolute_paths_are_left_alone(tmp_path):
    target = tmp_path / "somewhere" / "bm25"
    assert resolve_data_path(target) == target


def test_config_exposes_absolute_index_directories(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(reload=True)

    for path in (cfg.chroma_persist_directory, cfg.bm25_directory, cfg.emb_cache_directory):
        assert path.is_absolute(), f"{path} should be absolute"
        assert tmp_path not in path.parents


def test_env_override_still_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom-bm25"
    monkeypatch.setenv("BM25_DIRECTORY", str(override))
    cfg = load_config(reload=True)
    assert cfg.bm25_directory == override
