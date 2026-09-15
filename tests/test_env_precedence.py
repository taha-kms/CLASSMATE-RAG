"""The shell environment beats the project .env file.

cli/main.py used to load .env with override=True, so a variable written in
the file silently won over one exported in the shell and per-command
overrides such as `LOG_LEVEL=DEBUG rag stats` did nothing.
"""

from pathlib import Path

import cli.main as cli_main


def test_project_env_is_loaded_without_overriding_the_shell(monkeypatch):
    seen = {}

    def fake_load_dotenv(*args, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    cli_main._load_project_env()

    assert seen.get("override") is False, "shell variables must win over .env"


def test_it_reads_the_env_file_at_the_project_root(monkeypatch, tmp_path):
    seen = {}

    def fake_load_dotenv(*args, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    monkeypatch.chdir(tmp_path)  # must not depend on the working directory
    cli_main._load_project_env()

    path = Path(seen["dotenv_path"])
    assert path.name == ".env"
    assert (path.parent / "pyproject.toml").is_file()


def test_a_missing_or_broken_dotenv_does_not_crash_the_cli(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no dotenv here")

    monkeypatch.setattr("dotenv.load_dotenv", boom)
    cli_main._load_project_env()  # must swallow it
