"""`rag config`: the command-line half of the settings surface (#96)."""

import io
import json

import pytest

import rag.config as config
from cli.main import main
from rag import secrets, settings
from rag.settings import REGISTRY


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# my settings\nK_VECTOR=8\n", encoding="utf-8")
    monkeypatch.setattr(settings, "env_file_path", lambda: env)
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config, "__CONFIG_SINGLETON", getattr(config, "__CONFIG_SINGLETON"), raising=False)
    for s in REGISTRY:
        monkeypatch.delenv(s.key, raising=False)
    return env


def run(capsys, *argv):
    code = main(["config", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_set_changes_the_file_and_reports_what_happened(capsys, isolated):
    code, out, _ = run(capsys, "set", "K_VECTOR=12", "DEFAULT_LANGUAGE=it")

    assert code == 0
    changes = {c["key"]: c for c in json.loads(out)["changes"]}
    assert changes["K_VECTOR"]["applied"] is True
    assert isolated.read_text(encoding="utf-8").startswith("# my settings\nK_VECTOR=12\n")
    assert "DEFAULT_LANGUAGE=it" in isolated.read_text(encoding="utf-8")


def test_a_restart_only_setting_is_flagged(capsys):
    code, out, _ = run(capsys, "set", "LOG_LEVEL=debug")
    (change,) = json.loads(out)["changes"]
    assert (code, change["restart_required"], change["applied"]) == (0, True, False)


def test_an_invalid_value_fails_with_a_reason_and_changes_nothing(capsys, isolated):
    before = isolated.read_text(encoding="utf-8")
    code, out, err = run(capsys, "set", "K_VECTOR=5", "K_BM25=lots")

    assert code == 1 and out == ""
    assert "K_BM25" in json.loads(err)["error"]
    assert isolated.read_text(encoding="utf-8") == before


def test_a_value_is_required_for_an_ordinary_setting(capsys):
    code, _, err = run(capsys, "set", "K_VECTOR")
    assert code == 1
    assert "needs a value" in json.loads(err)["error"]


def test_a_credential_is_refused_on_the_command_line_and_not_stored(capsys):
    code, out, err = run(capsys, "set", "OPENAI_API_KEY=sk-fake-on-argv-0000000000")

    assert code == 1
    assert "command line" in json.loads(err)["error"]
    assert secrets.read_secret("OPENAI_API_KEY") is None
    assert "sk-fake-on-argv" not in out + err  # not echoed back either


def test_a_credential_can_be_piped_in_and_is_stored_but_never_printed(capsys, monkeypatch, isolated):
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-fake-piped-0000000000\n"))
    code, out, err = run(capsys, "set", "OPENAI_API_KEY")

    assert code == 0
    assert secrets.read_secret("OPENAI_API_KEY") == "sk-fake-piped-0000000000"
    assert "sk-fake-piped" not in out + err
    assert "sk-fake-piped" not in isolated.read_text(encoding="utf-8")


def test_get_prints_the_bare_value_for_scripts(capsys):
    code, out, _ = run(capsys, "get", "k_vector")  # keys are case-insensitive
    assert (code, out.strip()) == (0, "8")


def test_get_masks_a_credential(capsys):
    secrets.store_secret("ANTHROPIC_API_KEY", "sk-ant-fake-fake-fake-5678")
    code, out, _ = run(capsys, "get", "ANTHROPIC_API_KEY")
    assert code == 0
    assert "fake-fake" not in out and out.strip().endswith("5678")


def test_unset_removes_the_line(capsys, isolated):
    code, _, _ = run(capsys, "unset", "K_VECTOR")
    assert code == 0
    assert "K_VECTOR" not in isolated.read_text(encoding="utf-8")
    assert isolated.read_text(encoding="utf-8").startswith("# my settings")


def test_list_json_has_what_a_ui_needs(capsys):
    code, out, _ = run(capsys, "list", "--json")
    rows = {r["key"]: r for r in json.loads(out)}

    assert code == 0
    assert len(rows) == len(REGISTRY)
    assert rows["K_VECTOR"]["source"] == "file" and rows["K_VECTOR"]["value"] == "8"
    assert rows["ENABLE_ROUTING"]["restart_required"] is True
    assert rows["K_BM25"]["source"] == "default"


def test_list_can_be_limited_to_a_group(capsys):
    code, out, _ = run(capsys, "list", "--json", "--group", "retrieval")
    groups = {r["group"] for r in json.loads(out)}
    assert (code, groups) == (0, {"Retrieval"})


def test_list_says_when_the_shell_overrides_the_file(capsys, monkeypatch):
    monkeypatch.setenv("K_VECTOR", "30")
    code, out, _ = run(capsys, "list")
    line = next(line for line in out.splitlines() if "K_VECTOR" in line)
    assert code == 0
    assert "30" in line and "environment" in line and "shell overrides .env (8)" in line


@pytest.mark.parametrize("key", ["OPENAI_BASE_URL", "CHROMA_HTTP_URL"])
def test_the_command_line_can_set_and_unset_the_settings_the_api_may_not(capsys, isolated, key):
    code, _, _ = run(capsys, "set", f"{key}=https://proxy.example/v1")
    assert code == 0
    assert f"{key}=https://proxy.example/v1" in isolated.read_text(encoding="utf-8")

    code, _, _ = run(capsys, "unset", key)
    assert code == 0
    assert key not in isolated.read_text(encoding="utf-8")
