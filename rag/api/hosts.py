"""Refuse requests whose Host header is not this machine (#142).

The API has no authentication and relies on binding to loopback. That does not
stop a web page you visit: with DNS rebinding, ``evil.example`` resolves to
127.0.0.1 after the browser has loaded it, so the page becomes same-origin
with ``http://evil.example:8080`` and can call every endpoint. No CORS header
is involved, because as far as the browser knows nothing crosses an origin.

What the attacker cannot change is the Host header the browser sends, which is
the attacker's own hostname. Checking it against the names that mean "this
machine" turns those requests away.

Written out rather than using Starlette's TrustedHostMiddleware, which splits
the header on the first ':' and so reads ``[::1]:8080`` as ``[``.
"""

from __future__ import annotations

from collections.abc import Iterable

#: The names a browser on this machine uses for it. ``0.0.0.0`` is deliberately
#: absent: a public page can request http://0.0.0.0:PORT directly, and that
#: request carries ``Host: 0.0.0.0:PORT``.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def hostname_of(host_header: str) -> str:
    """The host part of a Host header, without a port and without IPv6 brackets."""
    value = host_header.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    # A bare IPv6 literal has several colons and no port; a name or IPv4 address has at most one.
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class HostGuard:
    """ASGI middleware: answer 400 unless the Host header names an allowed host."""

    def __init__(self, app, allowed: Iterable[str]) -> None:
        self.app = app
        self.allowed = frozenset(h.lower() for h in allowed)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            raw = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
            if hostname_of(raw) not in self.allowed:
                if scope["type"] == "http":
                    body = b"Invalid Host header. Open this page as http://localhost:PORT.\n"
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [
                                (b"content-type", b"text/plain; charset=utf-8"),
                                (b"content-length", str(len(body)).encode()),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                else:
                    await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)
