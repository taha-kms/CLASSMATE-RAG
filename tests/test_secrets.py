"""Storing credentials without leaking them."""

import os
import stat

import pytest

from rag import secrets


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    for key in secrets.SECRET_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---- masking ---------------------------------------------------------------


def test_a_masked_key_cannot_be_used():
    masked = secrets.mask("sk-ant-api03-abcdefghijklmnop1234")
    assert "abcdefghij" not in masked
    assert masked.endswith("1234"), "the tail identifies which key, without revealing it"


def test_a_short_value_reveals_nothing():
    # Four characters of an eight-character secret is half of it.
    assert set(secrets.mask("abcdefgh")) == {"*"}


def test_masking_nothing_gives_nothing():
    assert secrets.mask("") == ""
    assert secrets.mask(None) == ""


# ---- what counts as a secret ----------------------------------------------


@pytest.mark.parametrize("key", ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])
def test_known_credentials_are_recognised(key):
    assert secrets.is_secret(key)


@pytest.mark.parametrize("key", ["MISTRAL_API_KEY", "SOME_TOKEN", "DB_PASSWORD", "X_SECRET"])
def test_credential_shaped_names_are_recognised_too(key):
    # A provider added later should be masked before someone remembers to
    # add it to the list.
    assert secrets.is_secret(key)


@pytest.mark.parametrize("key", ["CHUNK_SIZE", "LOG_LEVEL", "MODEL_PROFILE"])
def test_ordinary_settings_are_not_masked(key):
    assert not secrets.is_secret(key)


# ---- storage ---------------------------------------------------------------


def test_a_stored_secret_can_be_read_back():
    secrets.store_secret("ANTHROPIC_API_KEY", "sk-ant-secret-value-1234")
    assert secrets.read_secret("ANTHROPIC_API_KEY") == "sk-ant-secret-value-1234"


def test_the_file_is_not_readable_by_anyone_else():
    secrets.store_secret("OPENAI_API_KEY", "sk-openai-value-abcd")
    mode = stat.S_IMODE(os.stat(secrets.secrets_path()).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_storing_replaces_rather_than_appends():
    secrets.store_secret("HF_TOKEN", "hf_first0000")
    secrets.store_secret("HF_TOKEN", "hf_second000")
    assert secrets.read_secret("HF_TOKEN") == "hf_second000"
    assert secrets.secrets_path().read_text().count("HF_TOKEN=") == 1


def test_other_secrets_survive_a_write():
    secrets.store_secret("HF_TOKEN", "hf_keepme0000")
    secrets.store_secret("OPENAI_API_KEY", "sk-openai-value-abcd")
    assert secrets.read_secret("HF_TOKEN") == "hf_keepme0000"


def test_deleting_removes_it():
    secrets.store_secret("HF_TOKEN", "hf_gone000000")
    secrets.delete_secret("HF_TOKEN")
    assert secrets.read_secret("HF_TOKEN") is None


def test_reading_from_an_absent_store_is_not_an_error():
    assert secrets.read_secret("ANTHROPIC_API_KEY") is None
    assert secrets.load_secrets() == {}


# ---- precedence ------------------------------------------------------------


def test_the_environment_beats_the_stored_file(monkeypatch):
    # Same rule as #48: a one-off override has to actually override.
    secrets.store_secret("ANTHROPIC_API_KEY", "sk-ant-stored-value-1111")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-exported-value-2222")
    assert secrets.read_secret("ANTHROPIC_API_KEY") == "sk-ant-exported-value-2222"


# ---- status ----------------------------------------------------------------


def test_status_says_what_is_set_without_saying_what_it_is():
    secrets.store_secret("OPENAI_API_KEY", "sk-openai-value-wxyz")
    status = secrets.secret_status("OPENAI_API_KEY")

    assert status.configured is True
    assert status.source == "file"
    assert status.masked.endswith("wxyz")
    assert "sk-openai-value-wxyz" != status.masked
    # The dataclass must not carry the value anywhere.
    assert "sk-openai-value-wxyz" not in repr(status)


def test_status_reports_where_it_came_from(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fromenvironment")
    assert secrets.secret_status("HF_TOKEN").source == "environment"


def test_an_unset_secret_says_so():
    status = secrets.secret_status("GEMINI_API_KEY")
    assert status.configured is False
    assert status.source == "unset"
    assert status.masked == ""


# ---- redaction -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "failed with key sk-ant-api03-abcdefghijklmnop",
        "Authorization: Bearer sk-abcdefghijklmnopqrstuvwx",
        "token hf_abcdefghijklmnop rejected",
        "using AIzaSyAbCdEfGhIjKlMnOpQrSt",
    ],
)
def test_key_shaped_values_are_scrubbed_from_text(text):
    out = secrets.redact(text)
    assert "*" in out
    # Nothing long and key-shaped survives intact.
    assert not any(len(w) > 20 and ("sk-" in w or "hf_" in w or "AIza" in w) for w in out.split())


def test_a_configured_value_is_scrubbed_even_if_its_shape_is_unknown():
    secrets.store_secret("MISTRAL_API_KEY", "totally-custom-shape-9876")
    assert "totally-custom-shape-9876" not in secrets.redact("provider said: totally-custom-shape-9876 is invalid")


def test_redacting_ordinary_text_changes_nothing():
    assert secrets.redact("no credentials here at all") == "no credentials here at all"
    assert secrets.redact("") == ""
