"""Changing settings without editing .env by hand (#96).

The failure modes worth pinning are the quiet ones: a typo that the loaders
would silently replace with the default, a "saved" that did nothing because the
shell overrides it, a credential written into .env, and an edit that eats the
comments around it.
"""

import os
import re
import stat
from pathlib import Path

import pytest
from dotenv import dotenv_values

import rag.config as config
from rag import secrets
from rag.settings import REGISTRY, SettingError, apply_changes, describe, get_setting, normalise, write_env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """A temp .env, a temp credential store, and no leakage from the real ones."""
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)  # never read the real .env
    # Record the current Config so the test's reloads are undone afterwards.
    monkeypatch.setattr(config, "__CONFIG_SINGLETON", getattr(config, "__CONFIG_SINGLETON"), raising=False)
    for setting in REGISTRY:
        monkeypatch.delenv(setting.key, raising=False)
    # Credentials can also arrive under alias names.
    for alias in ("GOOGLE_API_KEY", "HUGGINGFACE_HUB_TOKEN", "CLASSMATE_RAG_HF_TOKEN"):
        monkeypatch.delenv(alias, raising=False)


@pytest.fixture
def env_file(tmp_path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "# Models\nLLM_PROVIDER=llama_cpp\n\n# How much to fetch\nK_VECTOR=8\nK_BM25=8\nCHROMA_BIND_HOST=127.0.0.1\n",
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------ validation


@pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "on", " y "])
def test_truthy_spellings_normalise_to_true(raw):
    assert normalise(get_setting("USE_HYBRID"), raw) == "true"


@pytest.mark.parametrize("raw", ["false", "0", "No", "off", "n"])
def test_falsy_spellings_normalise_to_false(raw):
    assert normalise(get_setting("USE_HYBRID"), raw) == "false"


def test_a_typo_in_a_boolean_is_rejected_not_read_as_false():
    with pytest.raises(SettingError, match="true or false"):
        normalise(get_setting("USE_HYBRID"), "ture")


@pytest.mark.parametrize("raw", ["abc", "1.5", ""])
def test_numbers_must_be_numbers(raw):
    if raw == "":
        assert normalise(get_setting("K_VECTOR"), raw) == ""  # empty means the default
    else:
        with pytest.raises(SettingError, match="must be a number"):
            normalise(get_setting("K_VECTOR"), raw)


def test_bounds_are_enforced_on_both_sides():
    with pytest.raises(SettingError, match="at least 1"):
        normalise(get_setting("K_VECTOR"), "0")
    with pytest.raises(SettingError, match="at most 100"):
        normalise(get_setting("K_VECTOR"), "101")
    assert normalise(get_setting("K_VECTOR"), "100") == "100"


def test_a_choice_is_matched_case_insensitively_and_stored_canonically():
    assert normalise(get_setting("LOG_LEVEL"), "debug") == "DEBUG"
    assert normalise(get_setting("LLM_PROVIDER"), "Anthropic") == "anthropic"
    with pytest.raises(SettingError, match="one of"):
        normalise(get_setting("LLM_PROVIDER"), "bard")


def test_a_value_cannot_contain_a_newline():
    with pytest.raises(SettingError, match="single line"):
        normalise(get_setting("OPENAI_BASE_URL"), "http://x\nEVIL=1")


def test_an_empty_credential_is_refused():
    with pytest.raises(SettingError, match="unset"):
        normalise(get_setting("OPENAI_API_KEY"), "  ")


def test_unknown_setting_is_an_error_that_points_at_the_list():
    with pytest.raises(SettingError, match="rag config list"):
        get_setting("NOT_A_SETTING")


@pytest.mark.parametrize("setting", [s for s in REGISTRY if s.kind != "secret"], ids=lambda s: s.key)
def test_every_default_is_valid_for_its_own_rules(setting):
    normalise(setting, setting.default)


@pytest.mark.parametrize("raw", ["a\u2028b", "a\u2029b", "a\x0bb", "a\x0cb", "a\x85b", "a\tb", "a\x00b"])
def test_control_and_line_separator_characters_are_rejected(raw):
    """str.splitlines() breaks on more than \\n; a value carrying one could become a second entry."""
    with pytest.raises(SettingError, match="single line"):
        normalise(get_setting("OPENAI_MODEL"), raw)


def test_a_line_separator_cannot_smuggle_in_a_second_setting(env_file):
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(SettingError):
        apply_changes({"OPENAI_MODEL": "x\u2028LLM_PROVIDER=openai"}, env_path=env_file)
    assert env_file.read_text(encoding="utf-8") == before


def test_dotenv_interpolation_is_refused(monkeypatch):
    """${NAME} is expanded from the environment, so it would read back as that variable's value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-fake-fake-fake-1234")
    with pytest.raises(SettingError, match="expand"):
        normalise(get_setting("OPENAI_MODEL"), "${OPENAI_API_KEY}")
    assert normalise(get_setting("OPENAI_MODEL"), "costs $5") == "costs $5"  # a bare $ is not interpolation


# ------------------------------------------------------------ .env editing


def test_editing_keeps_comments_order_and_unrelated_lines(env_file):
    write_env(env_file, {"K_VECTOR": "12"})
    assert env_file.read_text(encoding="utf-8") == (
        "# Models\nLLM_PROVIDER=llama_cpp\n\n# How much to fetch\nK_VECTOR=12\nK_BM25=8\nCHROMA_BIND_HOST=127.0.0.1\n"
    )


def test_a_new_key_is_appended_and_none_removes(env_file):
    write_env(env_file, {"DEFAULT_LANGUAGE": "it", "K_BM25": None})
    values = dotenv_values(env_file)
    assert values["DEFAULT_LANGUAGE"] == "it"
    assert "K_BM25" not in values
    assert env_file.read_text(encoding="utf-8").startswith("# Models\n")


def test_a_commented_out_key_is_left_alone_and_the_real_one_added(env_file):
    env_file.write_text("# K_VECTOR=99\n", encoding="utf-8")
    write_env(env_file, {"K_VECTOR": "5"})
    assert env_file.read_text(encoding="utf-8") == "# K_VECTOR=99\nK_VECTOR=5\n"


def test_duplicate_lines_collapse_so_a_stale_one_cannot_win(env_file):
    env_file.write_text("K_VECTOR=1\nX=2\nK_VECTOR=3\n", encoding="utf-8")
    write_env(env_file, {"K_VECTOR": "9"})
    assert dotenv_values(env_file)["K_VECTOR"] == "9"
    assert env_file.read_text(encoding="utf-8").count("K_VECTOR") == 1


@pytest.mark.parametrize(
    "value",
    ["plain", "has space", "has # hash", 'has "quotes"', "back\\slash", "$HOME/x", "it's"],
)
def test_values_survive_a_round_trip_through_the_dotenv_parser(env_file, value):
    write_env(env_file, {"OPENAI_BASE_URL": value})
    assert dotenv_values(env_file)["OPENAI_BASE_URL"] == value


def test_file_mode_is_preserved_and_no_temp_file_is_left(env_file):
    env_file.chmod(0o640)
    write_env(env_file, {"K_VECTOR": "2"})
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o640
    assert [p.name for p in env_file.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_missing_file_is_created_private(tmp_path):
    path = tmp_path / "new" / ".env"
    write_env(path, {"K_VECTOR": "2"})
    assert dotenv_values(path) == {"K_VECTOR": "2"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ------------------------------------------------------------ precedence


def test_rewriting_does_not_split_a_line_the_dotenv_parser_keeps_whole(env_file):
    """A hand-edited file may hold a form feed; only \\n separates entries."""
    env_file.write_text("A=1\x0cB=2\nK_VECTOR=3\n", encoding="utf-8")
    write_env(env_file, {"K_VECTOR": "9"})
    assert env_file.read_text(encoding="utf-8") == "A=1\x0cB=2\nK_VECTOR=9\n"


def test_default_when_nothing_sets_it(tmp_path):
    state = describe("K_VECTOR", env_path=tmp_path / "absent.env")
    assert (state.value, state.source, state.overridden) == ("8", "default", False)


def test_file_value_is_reported_as_file(env_file):
    env_file.write_text("K_VECTOR=5\n", encoding="utf-8")
    state = describe("K_VECTOR", env_path=env_file)
    assert (state.value, state.source) == ("5", "file")


def test_a_value_dotenv_already_loaded_still_counts_as_the_file(env_file, monkeypatch):
    """load_dotenv copies the file into os.environ; that is not a shell override."""
    env_file.write_text("K_VECTOR=5\n", encoding="utf-8")
    monkeypatch.setenv("K_VECTOR", "5")
    state = describe("K_VECTOR", env_path=env_file)
    assert (state.source, state.overridden) == ("file", False)


def test_a_shell_value_that_disagrees_with_the_file_wins_and_says_so(env_file, monkeypatch):
    env_file.write_text("K_VECTOR=5\n", encoding="utf-8")
    monkeypatch.setenv("K_VECTOR", "20")
    state = describe("K_VECTOR", env_path=env_file)
    assert (state.value, state.source, state.overridden, state.file_value) == ("20", "environment", True, "5")


def test_a_shell_value_the_file_does_not_mention_is_environment_without_an_override(env_file, monkeypatch):
    monkeypatch.setenv("DEFAULT_LANGUAGE", "it")
    state = describe("DEFAULT_LANGUAGE", env_path=env_file)
    assert (state.value, state.source, state.overridden) == ("it", "environment", False)


def test_credentials_are_never_returned_in_full(monkeypatch):
    secrets.store_secret("OPENAI_API_KEY", "sk-fake-fake-fake-fake-1234")
    state = describe("OPENAI_API_KEY")
    assert state.source == "stored"
    assert "fake-fake" not in state.value
    assert state.value.endswith("1234")


# ------------------------------------------------------------ applying


def test_a_change_is_written_applied_and_visible_to_the_loader(env_file):
    (result,) = apply_changes({"K_VECTOR": "5"}, env_path=env_file)

    assert dotenv_values(env_file)["K_VECTOR"] == "5"
    assert (result.applied, result.restart_required) == (True, False)
    assert os.environ["K_VECTOR"] == "5"
    assert config.load_config().k_vector == 5


def test_a_restart_only_setting_says_so_instead_of_claiming_it_applied(env_file):
    (result,) = apply_changes({"ENABLE_ROUTING": "yes"}, env_path=env_file)
    assert dotenv_values(env_file)["ENABLE_ROUTING"] == "true"
    assert (result.applied, result.restart_required) == (False, True)
    assert "restart" in result.note


def test_nothing_is_written_if_any_change_is_invalid(env_file):
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(SettingError, match="K_BM25"):
        apply_changes({"K_VECTOR": "5", "K_BM25": "lots"}, env_path=env_file)
    assert env_file.read_text(encoding="utf-8") == before
    assert "K_VECTOR" not in os.environ


def test_a_shell_override_blocks_the_change_and_says_why(env_file, monkeypatch):
    monkeypatch.setenv("K_VECTOR", "20")
    (result,) = apply_changes({"K_VECTOR": "5"}, env_path=env_file)

    assert dotenv_values(env_file)["K_VECTOR"] == "5"  # the file is still updated
    assert os.environ["K_VECTOR"] == "20"  # but the process keeps the shell's value
    assert result.applied is False
    assert "shell" in result.note


def test_unset_removes_the_line_and_falls_back_to_the_default(env_file):
    apply_changes({"K_VECTOR": "5"}, env_path=env_file)
    (result,) = apply_changes({"K_VECTOR": None}, env_path=env_file)

    assert "K_VECTOR" not in dotenv_values(env_file)
    assert result.action == "unset"
    assert "K_VECTOR" not in os.environ
    assert config.load_config().k_vector == 8


def test_a_credential_goes_to_the_store_and_never_into_dotenv(env_file):
    (result,) = apply_changes({"ANTHROPIC_API_KEY": "sk-ant-fake-fake-fake-9999"}, env_path=env_file)

    assert "sk-ant-fake" not in env_file.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in dotenv_values(env_file)
    assert secrets.read_secret("ANTHROPIC_API_KEY") == "sk-ant-fake-fake-fake-9999"
    assert result.applied is True


def test_an_exported_credential_beats_a_stored_one_and_the_result_says_so(env_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-exported-0000000000")
    (result,) = apply_changes({"OPENAI_API_KEY": "sk-fake-stored-1111111111"}, env_path=env_file)
    assert result.applied is False
    assert "shell" in result.note


def test_overlap_must_stay_below_chunk_size_judged_on_the_resulting_values(env_file):
    with pytest.raises(SettingError, match="smaller than CHUNK_SIZE"):
        apply_changes({"CHUNK_OVERLAP": "1200"}, env_path=env_file)  # default size is 1000
    # Raising the size in the same request makes it valid.
    apply_changes({"CHUNK_SIZE": "2000", "CHUNK_OVERLAP": "1200"}, env_path=env_file)
    assert config.load_config().chunk_overlap == 1200


def test_overlap_check_uses_the_current_size_when_only_size_shrinks(env_file):
    apply_changes({"CHUNK_OVERLAP": "150"}, env_path=env_file)
    with pytest.raises(SettingError, match="CHUNK_OVERLAP"):
        apply_changes({"CHUNK_SIZE": "100"}, env_path=env_file)


# ------------------------------------------------------------ the registry


def _documented_variables() -> set[str]:
    docs = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*`([A-Z_0-9]+)`", docs, re.M))


# Documented, but not user-facing settings: applied by compose or the libraries,
# aliases of a credential already listed, or legacy.
_NOT_SETTINGS = {
    "CHROMA_BIND_HOST",
    "CHROMA_HOST_PORT",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "LLM_BACKEND",
    "LLM_REPO_ID",
    "LLM_FILENAME",
    "GOOGLE_API_KEY",
    "HUGGINGFACE_HUB_TOKEN",
    "CLASSMATE_RAG_HF_TOKEN",
}


def test_every_documented_setting_is_in_the_registry():
    missing = _documented_variables() - {s.key for s in REGISTRY} - _NOT_SETTINGS
    assert not missing, f"documented but not in rag/settings.py REGISTRY: {sorted(missing)}"


def test_the_registry_has_no_duplicates():
    keys = [s.key for s in REGISTRY]
    assert len(keys) == len(set(keys))


def test_every_credential_in_the_registry_is_known_to_the_secret_store():
    for s in REGISTRY:
        if s.kind == "secret":
            assert secrets.is_secret(s.key), s.key


# ------------------------------------------------------------ local-only settings


def test_a_setting_that_redirects_credentials_is_refused_unless_the_caller_vouches_for_itself(env_file):
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(SettingError, match="rag config set OPENAI_BASE_URL"):
        apply_changes({"K_VECTOR": "5", "OPENAI_BASE_URL": "https://evil.example"}, env_path=env_file)
    assert env_file.read_text(encoding="utf-8") == before  # the allowed one was not applied either


def test_the_same_change_is_allowed_when_the_caller_is_trusted(env_file):
    apply_changes({"OPENAI_BASE_URL": "https://proxy.example/v1"}, env_path=env_file, trusted_local=True)
    assert dotenv_values(env_file)["OPENAI_BASE_URL"] == "https://proxy.example/v1"


def test_ordinary_settings_need_no_vouching(env_file):
    apply_changes({"K_VECTOR": "5", "LLM_PROVIDER": "openai"}, env_path=env_file)
    assert dotenv_values(env_file)["K_VECTOR"] == "5"


def test_every_url_setting_is_local_only():
    """A future *_URL setting decides where data is sent, so it must opt out of HTTP deliberately."""
    urls = [s for s in REGISTRY if s.key.endswith("_URL")]
    assert urls, "expected at least one URL setting"
    assert all(s.local_only for s in urls), [s.key for s in urls if not s.local_only]


@pytest.mark.parametrize("key", ["OPENAI_BASE_URL", "CHROMA_HTTP_URL"])
def test_the_default_is_to_refuse_so_a_caller_that_forgets_the_flag_fails_closed(env_file, key):
    """No argument at all must be the safe call, or a new endpoint would skip the gate silently."""
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(SettingError, match="command line"):
        apply_changes({key: "https://elsewhere.example"}, env_path=env_file)
    assert env_file.read_text(encoding="utf-8") == before


# ------------------------------------------------------------ interpolation on read


def test_a_value_with_an_interpolation_is_shown_as_written_never_expanded(env_file, monkeypatch):
    """Even a credential with no recognisable shape must not surface through another setting."""
    monkeypatch.setenv("OPENAI_API_KEY", "plain-long-credential-value-no-prefix")
    env_file.write_text('OPENAI_MODEL="${OPENAI_API_KEY}"\n', encoding="utf-8")

    state = describe("OPENAI_MODEL", env_path=env_file)

    assert state.value == "${OPENAI_API_KEY}"
    assert "plain-long-credential" not in repr(state)


def test_an_expanded_value_the_loader_put_in_the_environment_is_not_read_as_a_shell_override(env_file, monkeypatch):
    env_file.write_text('OPENAI_MODEL="${OPENAI_API_KEY}"\n', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "plain-long-credential-value-no-prefix")
    monkeypatch.setenv("OPENAI_MODEL", "plain-long-credential-value-no-prefix")  # what load_dotenv would set

    state = describe("OPENAI_MODEL", env_path=env_file)

    assert (state.source, state.overridden) == ("file", False)
    assert "plain-long-credential" not in repr(state)
