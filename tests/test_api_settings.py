"""The settings endpoints (#96).

Over HTTP the things to pin are the ones a browser would otherwise get wrong:
a credential must never come back in full, an invalid batch must change
nothing, and a change that did not take effect must say so.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import rag.config as config  # noqa: E402
from rag import secrets, settings  # noqa: E402
from rag.api.app import create_app  # noqa: E402
from rag.settings import REGISTRY  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# mine\nK_VECTOR=8\n", encoding="utf-8")
    monkeypatch.setattr(settings, "env_file_path", lambda: env)
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config, "__CONFIG_SINGLETON", getattr(config, "__CONFIG_SINGLETON"), raising=False)
    for s in REGISTRY:
        monkeypatch.delenv(s.key, raising=False)
    return env


@pytest.fixture
def client():
    return TestClient(create_app())


def by_key(client) -> dict[str, dict]:
    return {s["key"]: s for s in client.get("/settings").json()["settings"]}


def test_every_setting_is_listed_with_what_a_form_needs(client):
    listed = by_key(client)
    assert set(listed) == {s.key for s in REGISTRY}

    k = listed["K_VECTOR"]
    assert (k["kind"], k["value"], k["source"], k["restart_required"]) == ("int", "8", "file", False)
    assert (k["minimum"], k["maximum"], k["default"]) == (1, 100, "8")

    provider = listed["LLM_PROVIDER"]
    assert provider["choices"] == ["llama_cpp", "anthropic", "openai", "gemini"]
    assert listed["ENABLE_ROUTING"]["restart_required"] is True


def test_put_changes_the_file_and_the_next_read_sees_it(client, isolated):
    r = client.put("/settings", json={"changes": {"K_VECTOR": 12, "USE_HYBRID": False}})

    assert r.status_code == 200
    assert {c["key"]: c["applied"] for c in r.json()["changes"]} == {"K_VECTOR": True, "USE_HYBRID": True}
    assert isolated.read_text(encoding="utf-8").startswith("# mine\nK_VECTOR=12\n")
    assert "USE_HYBRID=false" in isolated.read_text(encoding="utf-8")
    assert by_key(client)["K_VECTOR"]["value"] == "12"


def test_a_json_boolean_is_the_word_not_python_repr(client, isolated):
    client.put("/settings", json={"changes": {"USE_HYBRID": True}})
    assert "USE_HYBRID=true" in isolated.read_text(encoding="utf-8")  # not "True"


def test_an_invalid_batch_is_rejected_with_the_reason_and_changes_nothing(client, isolated):
    before = isolated.read_text(encoding="utf-8")
    r = client.put("/settings", json={"changes": {"K_VECTOR": 5, "K_BM25": "lots"}})

    assert r.status_code == 422
    assert "K_BM25" in r.json()["detail"]
    assert isolated.read_text(encoding="utf-8") == before


def test_an_unknown_setting_is_a_422_not_a_500(client):
    r = client.put("/settings", json={"changes": {"NOT_REAL": "1"}})
    assert r.status_code == 422


def test_an_empty_request_is_rejected(client):
    assert client.put("/settings", json={"changes": {}}).status_code == 422


def test_null_removes_a_setting(client, isolated):
    r = client.put("/settings", json={"changes": {"K_VECTOR": None}})
    assert r.json()["changes"][0]["action"] == "unset"
    assert "K_VECTOR" not in isolated.read_text(encoding="utf-8")
    assert by_key(client)["K_VECTOR"]["source"] == "default"


def test_a_restart_only_change_says_it_has_not_applied(client):
    (change,) = client.put("/settings", json={"changes": {"ENABLE_ROUTING": True}}).json()["changes"]
    assert (change["applied"], change["restart_required"]) == (False, True)
    assert "restart" in change["note"]


def test_a_shell_override_is_reported_on_read_and_on_write(client, monkeypatch):
    monkeypatch.setenv("K_VECTOR", "30")

    k = by_key(client)["K_VECTOR"]
    assert (k["value"], k["source"], k["overridden"], k["file_value"]) == ("30", "environment", True, "8")

    (change,) = client.put("/settings", json={"changes": {"K_VECTOR": 5}}).json()["changes"]
    assert change["applied"] is False and "shell" in change["note"]


def test_a_credential_can_be_written_and_is_never_returned(client, isolated):
    secret = "sk-ant-fake-fake-fake-4242"
    put = client.put("/settings", json={"changes": {"ANTHROPIC_API_KEY": secret}})

    assert put.status_code == 200
    assert secret not in put.text

    read = client.get("/settings")
    assert secret not in read.text
    key = by_key(client)["ANTHROPIC_API_KEY"]
    assert (key["source"], key["value"].endswith("4242")) == ("stored", True)
    assert secret not in isolated.read_text(encoding="utf-8")


def test_removing_a_credential(client):
    client.put("/settings", json={"changes": {"OPENAI_API_KEY": "sk-fake-fake-fake-fake-1234"}})
    client.put("/settings", json={"changes": {"OPENAI_API_KEY": None}})
    assert by_key(client)["OPENAI_API_KEY"]["source"] == "unset"


def test_an_unwritable_env_file_is_a_500_that_does_not_leak_the_path(client, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("/home/someone/private/.env")

    monkeypatch.setattr(settings, "write_env", boom)
    r = client.put("/settings", json={"changes": {"K_VECTOR": 5}})

    assert r.status_code == 500
    assert "/home/someone" not in r.text


@pytest.mark.parametrize("key", ["OPENAI_BASE_URL", "CHROMA_HTTP_URL"])
def test_a_setting_that_redirects_credentials_cannot_be_written_over_http(client, isolated, key):
    before = isolated.read_text(encoding="utf-8")
    r = client.put("/settings", json={"changes": {"K_VECTOR": 5, key: "https://evil.example"}})

    assert r.status_code == 422
    assert f"rag config set {key}" in r.json()["detail"]
    assert isolated.read_text(encoding="utf-8") == before  # the allowed change was not applied either


def test_the_listing_marks_the_local_only_settings(client):
    listed = by_key(client)
    assert listed["OPENAI_BASE_URL"]["local_only"] is True
    assert listed["CHROMA_HTTP_URL"]["local_only"] is True
    assert listed["K_VECTOR"]["local_only"] is False


def test_an_interpolated_value_from_a_hand_edited_env_does_not_leak_an_exported_key(client, isolated, monkeypatch):
    """python-dotenv expands ${NAME} from the environment; the listing must not print what it finds."""
    key = "sk-fake-fake-fake-fake-1234"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    isolated.write_text('OPENAI_MODEL="${OPENAI_API_KEY}"\n', encoding="utf-8")

    body = client.get("/settings").text

    assert key not in body
    assert "OPENAI_MODEL" in body  # the setting is still listed, just masked


def test_credentials_embedded_in_a_url_setting_are_not_returned(client, isolated):
    isolated.write_text(
        "OPENAI_BASE_URL=https://someone:hunter2-not-a-real-password@proxy.example/v1\n", encoding="utf-8"
    )

    body = client.get("/settings").text

    assert "hunter2" not in body and "someone:" not in body
    assert "https://proxy.example/v1" in body  # the useful part is still shown


def test_an_exported_credential_with_no_recognisable_shape_cannot_surface_through_another_setting(
    client, isolated, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "plain-long-credential-value-no-prefix")
    isolated.write_text('OPENAI_MODEL="${OPENAI_API_KEY}"\n', encoding="utf-8")

    body = client.get("/settings").text

    assert "plain-long-credential" not in body


@pytest.mark.parametrize(
    "url",
    [
        "https://someone:p@ssword-fake@proxy.example/v1",  # '@' in the password
        "https://someone:pa/ssword-fake@proxy.example/v1",  # unencoded '/' in the password
        "https://someone:p@ss/wo@rd-fake@proxy.example/v1",  # both
    ],
)
def test_no_part_of_a_password_survives_even_when_it_contains_at_or_slash(client, isolated, url):
    isolated.write_text(f"OPENAI_BASE_URL={url}\n", encoding="utf-8")

    body = client.get("/settings").text

    for fragment in ("ssword", "ss/wo", "wo@rd", "someone"):
        assert fragment not in body, fragment
    assert "https://proxy.example/v1" in body
