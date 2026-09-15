"""Every environment variable the code reads is documented.

docs/configuration.md had drifted far enough that it listed LLAMA_MODEL_PATH
and LLAMA_CONTEXT_SIZE, which nothing reads, while omitting eleven variables
that are read. This keeps the two in step.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "configuration.md"
EXAMPLE = ROOT / ".env.example"

# Read by libraries or docker-compose rather than by our Python, so they are
# explained in prose instead of the reference tables.
EXTERNALLY_CONSUMED = {
    "CHROMA_BIND_HOST",
    "CHROMA_HOST_PORT",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
}

# Accepted as fallbacks for HF_TOKEN and described in its row.
TOKEN_ALIASES = {"HUGGINGFACE_HUB_TOKEN", "CLASSMATE_RAG_HF_TOKEN"}

_READ_RE = re.compile(
    r'_getenv_\w+\(\s*"([A-Z_0-9]+)"'
    r'|os\.getenv\(\s*"([A-Z_0-9]+)"'
    r'|os\.environ(?:\.get)?\(?\[?\s*"([A-Z_0-9]+)"'
)


def _variables_read_by_code() -> set[str]:
    found: set[str] = set()
    for package in ("rag", "cli"):
        for path in (ROOT / package).rglob("*.py"):
            for match in _READ_RE.finditer(path.read_text(encoding="utf-8")):
                found.update(g for g in match.groups() if g)
    return found


def _variables_in_docs() -> set[str]:
    return set(re.findall(r"^\|\s*`([A-Z_0-9]+)`", DOCS.read_text(encoding="utf-8"), re.M))


def test_every_variable_the_code_reads_is_documented():
    undocumented = _variables_read_by_code() - _variables_in_docs() - TOKEN_ALIASES
    assert not undocumented, f"undocumented environment variables: {sorted(undocumented)}"


def test_the_docs_do_not_describe_variables_nothing_reads():
    stale = _variables_in_docs() - _variables_read_by_code() - EXTERNALLY_CONSUMED
    assert not stale, f"documented but read by nothing: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted({"DEDUP_CHUNKS", "NEIGHBOR_RADIUS", "DOC_DIVERSITY_CAP",
                                         "INGEST_THREADS", "ROUTE_MAX_TOKENS", "LLAMA_GPU_LAYERS"}))
def test_previously_hidden_settings_appear_in_the_example_file(name):
    # These were readable only by grepping the source.
    assert name in EXAMPLE.read_text(encoding="utf-8")
