"""Checking a saved provider key against the provider (#147).

No test here reaches a real provider. Most use a fake opener; the ones that
need the real urllib behaviour (redirects, headers on the wire) talk to a
throwaway server on 127.0.0.1.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import rag.config as config
from rag import keycheck, secrets, settings
from rag.settings import REGISTRY

# Obviously fake and low entropy: the secret scan reads every commit.
KEY = "sk-fake-fake-fake-fake-1234"
OTHER = "sk-other-other-other-9999"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("K_VECTOR=8\n", encoding="utf-8")
    monkeypatch.setattr(settings, "env_file_path", lambda: env)
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    monkeypatch.setattr(settings, "_PENDING", {})
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    for s in REGISTRY:
        monkeypatch.delenv(s.key, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config.load_config(reload=True)
    yield
    config.load_config(reload=True)


class Reply:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeOpener:
    """Records requests and answers with a canned status or error."""

    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        self.status, self.error, self.requests = status, error, []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, self.status, "x", {}, io.BytesIO(f"body mentions {KEY}".encode())
            )
        return Reply(self.status)


def store(**values: str) -> None:
    for k, v in values.items():
        secrets.store_secret(k, v)


# ---------------------------------------------------------------- outcomes


@pytest.mark.parametrize(
    ("provider", "status", "outcome"),
    [
        ("anthropic", 200, "valid"),
        ("openai", 200, "valid"),
        ("gemini", 200, "valid"),
        ("anthropic", 401, "rejected"),
        ("openai", 403, "rejected"),
        ("gemini", 403, "rejected"),
        # Google answers a bad key with 400, other providers do not.
        ("gemini", 400, "rejected"),
        ("openai", 400, "unexpected"),
        ("anthropic", 429, "rate_limited"),
        ("openai", 500, "unexpected"),
        ("openai", 302, "unexpected"),
    ],
)
def test_status_maps_to_a_fixed_outcome(provider, status, outcome):
    assert keycheck.classify(provider, status) == outcome


def test_a_missing_key_makes_no_request():
    opener = FakeOpener()
    result = keycheck.check_key("anthropic", opener=opener)
    assert (result.outcome, result.source, result.masked) == ("missing", "", "")
    assert opener.requests == []


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        keycheck.check_key("evil.example.com")


def test_a_network_failure_is_unreachable_and_leaks_nothing(monkeypatch):
    store(ANTHROPIC_API_KEY=KEY)
    opener = FakeOpener(error=OSError(f"connect to proxy failed using {KEY}"))
    result = keycheck.check_key("anthropic", opener=opener)
    assert result.outcome == "unreachable"
    assert result.status is None
    assert KEY not in json.dumps(result.as_dict())


def test_a_refusal_does_not_echo_the_provider_body():
    store(ANTHROPIC_API_KEY=KEY)
    result = keycheck.check_key("anthropic", opener=FakeOpener(status=401))
    assert result.outcome == "rejected"
    assert result.status == 401
    assert KEY not in json.dumps(result.as_dict())


# ------------------------------------------------- which key was tested


def test_the_stored_key_is_used_and_reported_as_stored():
    store(ANTHROPIC_API_KEY=KEY)
    opener = FakeOpener()
    result = keycheck.check_key("anthropic", opener=opener)
    assert result.source == "stored"
    assert opener.requests[0].get_header("X-api-key") == KEY


def test_an_exported_key_wins_and_the_result_says_so(monkeypatch):
    # The case that would otherwise report "valid" for a key nobody just typed.
    store(ANTHROPIC_API_KEY=KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    opener = FakeOpener()
    result = keycheck.check_key("anthropic", opener=opener)
    assert result.source == "environment"
    assert opener.requests[0].get_header("X-api-key") == OTHER
    assert result.masked.endswith("9999")


def test_the_result_carries_only_the_masked_tail():
    store(OPENAI_API_KEY=KEY)
    payload = json.dumps(keycheck.check_key("openai", opener=FakeOpener()).as_dict())
    assert KEY not in payload
    assert "1234" in payload


# ------------------------------------------------------------- the wire


def test_each_provider_gets_its_key_in_a_header_not_the_url(monkeypatch):
    store(ANTHROPIC_API_KEY=KEY, OPENAI_API_KEY=KEY, GEMINI_API_KEY=KEY)
    for provider, header in (
        ("anthropic", "X-api-key"),
        ("openai", "Authorization"),
        ("gemini", "X-goog-api-key"),
    ):
        opener = FakeOpener()
        keycheck.check_key(provider, opener=opener)
        request = opener.requests[0]
        assert KEY in request.get_header(header)
        assert KEY not in request.full_url, provider
        assert request.get_method() == "GET"


def test_gemini_also_accepts_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", KEY)
    result = keycheck.check_key("gemini", opener=FakeOpener())
    assert (result.outcome, result.source) == ("valid", "environment")


def test_openai_uses_the_configured_base_url_and_reports_it(monkeypatch):
    store(OPENAI_API_KEY=KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1/")
    config.load_config(reload=True)
    opener = FakeOpener()
    result = keycheck.check_key("openai", opener=opener)
    assert opener.requests[0].full_url == "http://localhost:8080/v1/models"
    assert result.target == "http://localhost:8080/v1/models"


def test_openai_defaults_to_the_real_service():
    store(OPENAI_API_KEY=KEY)
    opener = FakeOpener()
    keycheck.check_key("openai", opener=opener)
    assert opener.requests[0].full_url == "https://api.openai.com/v1/models"


# ------------------------------- real urllib, against a local throwaway server


class _Handler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []
    mode = "ok"

    def do_GET(self):  # noqa: N802 - http.server API
        _Handler.seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/v1/models" and _Handler.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()
        elif self.path == "/v1/models" and _Handler.mode == "deny":
            self.send_response(401)
            self.end_headers()
        elif self.path == "/v1/models":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def log_message(self, *args):  # silence
        return None


@pytest.fixture
def server(monkeypatch):
    _Handler.seen, _Handler.mode = [], "ok"
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    store(OPENAI_API_KEY=KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/v1")
    config.load_config(reload=True)
    yield _Handler
    httpd.shutdown()
    httpd.server_close()


def test_a_real_request_carries_the_bearer_token_and_succeeds(server):
    result = keycheck.check_key("openai")
    assert result.outcome == "valid"
    assert server.seen == [("/v1/models", f"Bearer {KEY}")]


def test_a_real_401_is_rejected(server):
    server.mode = "deny"
    assert keycheck.check_key("openai").outcome == "rejected"


def test_a_redirect_is_not_followed_so_the_key_is_not_resent(server):
    server.mode = "redirect"
    result = keycheck.check_key("openai")
    assert result.outcome == "unexpected"
    assert result.status == 302
    assert [path for path, _ in server.seen] == ["/v1/models"]


# --------------------------------------------------------------- endpoint


def test_the_endpoint_reports_the_outcome_and_never_the_key(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag.api.app import create_app

    store(ANTHROPIC_API_KEY=KEY)
    monkeypatch.setattr(keycheck, "_default_opener", lambda: FakeOpener(status=401))
    r = TestClient(create_app()).post("/settings/validate", json={"provider": "anthropic"})
    assert r.status_code == 200
    body = r.json()
    assert (body["outcome"], body["ok"], body["source"]) == ("rejected", False, "stored")
    assert KEY not in r.text


def test_the_endpoint_only_takes_a_known_provider():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag.api.app import create_app

    client = TestClient(create_app())
    for bad in ("evil.example.com", "", "http://169.254.169.254/"):
        assert client.post("/settings/validate", json={"provider": bad}).status_code == 422
    # A key in the request is not a thing: unknown fields are ignored, not used.
    assert client.post("/settings/validate", json={"provider": "openai", "key": KEY}).status_code == 200


# --------------------------------------------------------------------- CLI


def test_cli_test_exits_nonzero_unless_every_key_is_valid(monkeypatch, capsys):
    from cli.main import main

    store(ANTHROPIC_API_KEY=KEY)
    monkeypatch.setattr(keycheck, "_default_opener", lambda: FakeOpener(status=401))
    code = main(["config", "test", "anthropic"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["checks"][0]["outcome"] == "rejected"
    assert KEY not in json.dumps(out)

    monkeypatch.setattr(keycheck, "_default_opener", lambda: FakeOpener(status=200))
    assert main(["config", "test", "anthropic"]) == 0
