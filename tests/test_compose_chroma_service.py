"""The compose database service agrees with the chromadb pin.

Two mistakes in this file fail silently rather than loudly, which is why they
are worth a test:

* Chroma 1.x reads its persistence path from ``/config.yaml`` and stores data
  under ``/data``. It ignores ``PERSIST_DIRECTORY``. Mounting the host index at
  the 0.6 location leaves the server pointed at an empty directory, and the
  corpus reads as zero chunks with nothing logged.
* The 1.x image ships no curl, wget or nc, and ``/bin/sh`` is dash. A
  healthcheck written with any of those never passes, so ``--wait`` hangs until
  it gives up rather than reporting anything useful.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
REQUIREMENTS = ROOT / "requirements.txt"


@pytest.fixture(scope="module")
def chroma_service() -> dict:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return compose["services"]["chroma"]


def _pinned_major() -> int:
    """The chromadb major version requirements.txt asks for."""
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^chromadb>=(\d+)\.", line.strip())
        if m:
            return int(m.group(1))
    raise AssertionError("no chromadb>= pin found in requirements.txt")


def test_server_image_major_matches_the_client_pin(chroma_service):
    """A 1.x client against a 0.6 server is a connection error at best."""
    image = chroma_service["image"]
    m = re.match(r"^chromadb/chroma:(\d+)\.", image)
    assert m, f"expected a pinned chromadb/chroma:X.Y.Z image, got {image!r}"
    assert int(m.group(1)) == _pinned_major(), (
        f"compose runs {image} but requirements.txt pins chromadb "
        f"{_pinned_major()}.x. The on-disk layout and the wire API both differ "
        f"between majors; keep the two in step."
    )


def test_index_is_mounted_where_the_server_actually_looks(chroma_service):
    targets = [str(v).split(":")[1] for v in chroma_service["volumes"]]
    assert "/data" in targets, (
        f"chroma 1.x persists to /data (set in its /config.yaml), but the "
        f"volume targets are {targets}. Mounting elsewhere gives the server an "
        f"empty directory and the corpus silently reads as 0 chunks."
    )


def test_no_dead_persistence_env_vars(chroma_service):
    """0.6 read these. 1.x ignores them, so leaving them in misleads."""
    env = chroma_service.get("environment") or {}
    names = set(env) if isinstance(env, dict) else {e.split("=")[0] for e in env}
    dead = names & {"PERSIST_DIRECTORY", "IS_PERSISTENT"}
    assert not dead, (
        f"{sorted(dead)} are read by chroma 0.6 and ignored by 1.x. They imply "
        f"a persistence path that is not the one in use."
    )


def test_healthcheck_uses_a_shell_the_image_actually_has(chroma_service):
    test = chroma_service["healthcheck"]["test"]
    assert test[0] == "CMD", f"expected an exec-form healthcheck, got {test!r}"
    probe = " ".join(test[1:])

    missing = [tool for tool in ("curl", "wget", "nc") if re.search(rf"\b{tool}\b", probe)]
    assert not missing, f"the chromadb/chroma image ships none of {missing}; a healthcheck using them can never pass."
    assert test[1] == "bash", (
        f"the probe relies on /dev/tcp, which is a bash builtin and not "
        f"available in dash (/bin/sh in this image); got {test[1]!r}."
    )


def test_healthcheck_checks_the_response_not_just_the_socket(chroma_service):
    """The port accepts connections before the routes are mounted."""
    probe = " ".join(chroma_service["healthcheck"]["test"][1:])
    assert "200 OK" in probe, (
        "a bare TCP connect succeeds against a half-started server; the probe "
        "should require a 200 from the heartbeat route."
    )
    assert "/api/v2/" in probe, "v1 routes are gone in chroma 1.x"
