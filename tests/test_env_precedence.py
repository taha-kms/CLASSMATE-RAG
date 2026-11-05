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


def test_a_broken_dotenv_at_import_time_warns_instead_of_crashing():
    """The bootstrap has to survive *during import*, not only when called.

    The test above calls _load_project_env() on an already-imported module,
    where every module-level name is bound. That is not the situation the
    handler runs in: the call happens at the top of cli/main.py, before the
    rest of its imports. A warning that referenced `sys` therefore raised
    NameError at the only moment it could ever fire, and no test could see it,
    because by the time a test could call the function `sys` was bound.

    Hence a subprocess: it exercises the real import.
    """
    import subprocess
    import sys

    code = (
        "import dotenv\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('broken .env')\n"
        "dotenv.load_dotenv = boom\n"
        "import cli.main\n"
        "print('imported')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert proc.returncode == 0, f"importing the CLI crashed:\n{proc.stderr}"
    assert "imported" in proc.stdout
    assert "broken .env" in proc.stderr, f"the failure was swallowed entirely:\n{proc.stderr}"
    assert "NameError" not in proc.stderr
