"""Fetching model files: where they land, and what happens when it goes wrong."""

import collections
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.model_download import (
    DownloadError,
    _explain,
    download_model,
    download_profile,
    list_local_models,
    models_dir,
)

Usage = collections.namedtuple("Usage", "total used free")


def test_models_dir_ignores_the_working_directory(monkeypatch, tmp_path):
    # model_fetch.py used Path("./models").resolve(), so running from
    # elsewhere started a second models directory (#15).
    monkeypatch.chdir(tmp_path)
    resolved = models_dir()
    assert resolved.is_absolute()
    assert (resolved.parent / "pyproject.toml").is_file()
    assert tmp_path not in resolved.parents


def test_listing_an_absent_directory_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path / "nope")
    assert list_local_models() == []


def test_models_are_listed_largest_first(monkeypatch, tmp_path):
    (tmp_path / "small.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "big.gguf").write_bytes(b"x" * 4096)
    (tmp_path / "notamodel.txt").write_bytes(b"x")
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path)

    names = [m.path.name for m in list_local_models()]
    assert names == ["big.gguf", "small.gguf"], "only .gguf, biggest first"


def test_an_existing_file_is_not_downloaded_again(monkeypatch, tmp_path):
    (tmp_path / "already.gguf").write_bytes(b"x")
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path)

    with patch("rag.model_download._hf_download") as fake:
        assert download_model("some/repo", "already.gguf") == tmp_path / "already.gguf"
    fake.assert_not_called()


def test_too_little_disk_refuses_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path)
    monkeypatch.setattr("rag.model_download._remote_size_gb", lambda *a, **k: 4.0)

    with patch.object(shutil, "disk_usage", return_value=Usage(0, 0, 1 * 1024**3)):
        with patch("rag.model_download._hf_download") as fake:
            with pytest.raises(DownloadError, match="Not enough disk space"):
                download_model("some/repo", "big.gguf")
    # The point is refusing first, rather than dying at 95%.
    fake.assert_not_called()


def test_an_unknown_remote_size_does_not_block_the_download(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path)
    monkeypatch.setattr("rag.model_download._remote_size_gb", lambda *a, **k: None)

    with patch.object(shutil, "disk_usage", return_value=Usage(0, 0, 1 * 1024**3)):
        with patch(
            "rag.model_download._hf_download",
            return_value=str(tmp_path / "m.gguf"),
        ):
            assert download_model("some/repo", "m.gguf") == tmp_path / "m.gguf"


@pytest.mark.parametrize(
    "raised, expected",
    [
        ("401 Client Error: Unauthorized", "HF_TOKEN"),
        ("403 Forbidden: gated repo", "Accept the licence"),
        ("404 Client Error: Entry Not Found", "Check the exact filename"),
    ],
)
def test_hub_errors_become_advice(raised, expected):
    message = _explain(RuntimeError(raised), "some/repo", "m.gguf")
    assert expected in message
    # Always point at something the reader can open.
    assert "huggingface.co/some/repo" in message or "settings/tokens" in message


def test_an_unrecognised_error_still_names_the_file():
    message = _explain(RuntimeError("the network is on fire"), "some/repo", "m.gguf")
    assert "m.gguf" in message and "the network is on fire" in message


def test_downloading_a_profile_fetches_each_file_once(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.model_download.models_dir", lambda: tmp_path)
    calls = []

    def fake(repo_id, filename, **kw):
        calls.append(filename)
        p = tmp_path / filename
        p.write_bytes(b"x")
        return str(p)

    with patch("rag.model_download.download_model", side_effect=lambda r, f, **k: Path(fake(r, f))):
        download_profile("light")

    # light uses one general model across three routes plus a coder model.
    assert len(calls) == len(set(calls)) == 2


def test_an_unknown_profile_says_which_ones_exist():
    with pytest.raises(DownloadError, match="light"):
        download_profile("enormous")
