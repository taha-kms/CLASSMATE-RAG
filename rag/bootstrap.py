"""Read the project .env before anything heavy is imported.

This lives in its own module, kept free of imports, because both entry points
need it and both need it *first*. `rag.config` is not the right home: it
imports dotenv at module scope, which would make the optional-dependency path
below unreachable, and importing it is already more than a bootstrap should
cost.
"""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """The directory holding pyproject.toml and .env."""
    return Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    """Load the project .env before the heavy imports in the caller.

    The timing matters: transformers and sentence-transformers read their cache
    variables (HF_HOME, HUGGINGFACE_HUB_CACHE, SENTENCE_TRANSFORMERS_HOME) at
    import time, so the file has to be read first. Chroma connection settings
    and HF_TOKEN come from here too.

    override=False on purpose. A variable already exported in the shell beats
    the file, which is how dotenv is normally expected to work and what
    rag.config has always done. With override=True, `LOG_LEVEL=DEBUG rag stats`
    was silently ignored because .env happened to set LOG_LEVEL, while a
    setting merely commented out in .env worked fine -- so whether an override
    took effect depended on whether a line was uncommented.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=project_root() / ".env", override=False)
    except ImportError:
        # python-dotenv is optional; the OS environment is a complete config
        # source on its own, so this one stays quiet.
        pass
    except Exception as e:  # noqa: BLE001 - bootstrap: never block `rag --help`
        # This runs before any command, so a failure here must not take the
        # whole process down over an optional convenience. It must not be
        # silent either: the user set those variables expecting them to apply,
        # and a quietly ignored .env is a long afternoon.
        print(f"Warning: .env could not be loaded ({e}); using the environment only.", file=sys.stderr)
