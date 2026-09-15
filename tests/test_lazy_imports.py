"""The package layer must not drag the ML stack in on import.

rag/pipeline, rag/retrieval, rag/routing and rag/generation used to import
sentence-transformers, chromadb and llama_cpp transitively, so CI could not
import them and nothing under them could be tested.
"""

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import rag

ROOT = Path(__file__).resolve().parents[1]


# Leaf modules that each wrap one third-party parser or runtime. These are
# expected to need their library; everything else should import bare.
NEEDS_ITS_OWN_LIBRARY = {
    "rag.generation.llama_cpp_runner",
    "rag.loaders.docx_loader",
    "rag.loaders.epub_loader",
    "rag.loaders.html_loader",
    "rag.loaders.html_readable",
    "rag.loaders.pdf_loader",
    "rag.loaders.pptx_loader",
    "rag.loaders.text_loader",
}

HEAVY = ("torch", "sentence_transformers", "llama_cpp", "chromadb")

ALL_MODULES = sorted(m.name for m in pkgutil.walk_packages(rag.__path__, "rag."))


@pytest.mark.parametrize("name", [m for m in ALL_MODULES if m not in NEEDS_ITS_OWN_LIBRARY])
def test_module_imports_without_the_ml_stack(name):
    importlib.import_module(name)


def test_the_public_packages_know_every_name_they_advertise():
    # Deliberately not hasattr: resolving LlamaCppRunner would import
    # llama_cpp, which is the very thing being avoided. dir() goes through
    # the lazy __dir__, so it proves the name is wired up without loading it.
    for package in ("rag.retrieval", "rag.routing", "rag.generation"):
        module = importlib.import_module(package)
        advertised, reachable = set(module.__all__), set(dir(module))
        assert advertised <= reachable, f"{package} cannot resolve {advertised - reachable}"


@pytest.mark.parametrize("heavy", HEAVY)
def test_importing_the_pipeline_does_not_load_the_heavy_stack(heavy):
    # In a subprocess, so the result does not depend on what earlier tests
    # in this session happened to import.
    code = f"import rag.pipeline, sys; print({heavy!r} in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", f"importing rag.pipeline loaded {heavy}"


def test_an_unknown_export_still_raises_attribute_error():
    import rag.retrieval as retrieval

    with pytest.raises(AttributeError):
        retrieval.NoSuchThing
