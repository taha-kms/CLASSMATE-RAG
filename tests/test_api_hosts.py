"""Requests whose Host header is not this machine are refused (#142).

A DNS-rebinding page is same-origin with its own hostname, so the only thing
that distinguishes its requests is the Host header it has to send.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag.api.app import create_app  # noqa: E402
from rag.api.hosts import LOOPBACK_HOSTS, HostGuard, hostname_of  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """These tests send writes. If the guard ever failed, they would land here, not in the real .env."""
    from rag import secrets, settings

    env = tmp_path / ".env"
    env.write_text("K_VECTOR=8\n", encoding="utf-8")
    monkeypatch.setattr(settings, "env_file_path", lambda: env)
    monkeypatch.setattr(secrets, "secrets_path", lambda: tmp_path / ".secrets.env")
    monkeypatch.setattr(settings, "_PENDING", {})
    return env


def client(base_url: str, *, guarded: bool = True) -> TestClient:
    app = create_app(allowed_hosts=LOOPBACK_HOSTS if guarded else None)
    return TestClient(app, base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://LOCALHOST:8080",
    ],
)
def test_this_machine_is_let_through(base_url):
    assert client(base_url).get("/docs").status_code == 200


@pytest.mark.parametrize(
    "base_url",
    [
        "http://evil.example",  # a rebinding page
        "http://evil.example:8080",
        "http://localhost.evil.example",  # looks like localhost, is not
        "http://127.0.0.1.evil.example",
        "http://0.0.0.0:8080",  # a public page can request this directly
        "http://192.168.1.20:8080",
    ],
)
def test_anything_else_is_refused_before_a_route_runs(base_url):
    r = client(base_url).get("/stats")
    assert r.status_code == 400
    assert "Invalid Host" in r.text


def test_a_rebinding_page_cannot_write_settings_or_reach_destructive_routes(isolated):
    before = isolated.read_text(encoding="utf-8")
    c = client("http://evil.example:8080")

    assert c.put("/settings", json={"changes": {"K_VECTOR": 5}}).status_code == 400
    # A dry run: this proves the request is refused before the route runs without
    # ever being able to delete anything if the guard were broken.
    assert c.request("DELETE", "/chunks", json={"course": "x", "dry_run": True}).status_code == 400
    assert isolated.read_text(encoding="utf-8") == before


def test_a_request_with_no_host_header_is_refused():
    c = client("http://localhost")
    assert c.get("/stats", headers={"host": ""}).status_code == 400


def test_the_check_is_off_unless_asked_for():
    """Not every bind address has names the server can know; the caller decides."""
    assert client("http://evil.example", guarded=False).get("/docs").status_code == 200


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("localhost", "localhost"),
        ("localhost:8080", "localhost"),
        ("127.0.0.1:8080", "127.0.0.1"),
        ("[::1]:8080", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("EVIL.example:80", "evil.example"),
        ("", ""),
    ],
)
def test_hostname_parsing(header, expected):
    assert hostname_of(header) == expected


async def _status_for(host_header: bytes) -> int | None:
    """Drive the guard directly: TestClient cannot be given an IPv6 base URL."""
    seen: list[int] = []

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        if message["type"] == "http.response.start":
            seen.append(message["status"])

    guard = HostGuard(downstream, LOOPBACK_HOSTS)
    await guard({"type": "http", "headers": [(b"host", host_header)]}, None, send)
    return seen[0] if seen else None


@pytest.mark.parametrize(
    ("header", "status"), [(b"[::1]:8080", 200), (b"[::1]", 200), (b"[::2]:8080", 400), (b"[evil", 400)]
)
def test_an_ipv6_host_header_is_judged_correctly(header, status):
    import asyncio

    assert asyncio.run(_status_for(header)) == status
