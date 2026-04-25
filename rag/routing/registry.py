"""
Per-route model spec.

Holds the mapping from a route ("math", "code", "translation", "default")
to its GGUF path and per-model llama.cpp parameters. Paths come from
Config (env-overridable).

If a route's model file isn't present on disk, get_model_spec() can either
raise (strict) or transparently fall back to the default route's spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rag.config import Config, load_config

from .types import DEFAULT_ROUTE, Route


@dataclass(frozen=True)
class ModelSpec:
    """All the knobs the loader needs to instantiate one llama.cpp model."""

    route: Route
    model_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    seed: int = 42
    verbose: bool = False


def route_model_paths(cfg: Optional[Config] = None) -> Dict[Route, Path]:
    """Return the configured GGUF path for each route."""
    cfg = cfg or load_config()
    return {
        "math": cfg.route_math_model_path,
        "code": cfg.route_code_model_path,
        "translation": cfg.route_translation_model_path,
        "default": cfg.route_default_model_path,
    }


def get_model_spec(
    route: Route,
    *,
    cfg: Optional[Config] = None,
    fallback_to_default: bool = True,
) -> ModelSpec:
    """
    Resolve a ModelSpec for the given route.

    - If the route's model file is missing on disk and fallback_to_default
      is True, return the default route's spec instead (with a corrected
      `route` field so the caller can log the demotion). This is the
      pragmatic behavior on a machine where the user hasn't downloaded
      every specialist yet.
    - If fallback_to_default is False, raise FileNotFoundError instead.
    """
    cfg = cfg or load_config()
    paths = route_model_paths(cfg)

    target_path = paths[route]
    resolved = target_path.expanduser().resolve()

    if not resolved.exists():
        if not fallback_to_default or route == DEFAULT_ROUTE:
            raise FileNotFoundError(
                f"Model file for route '{route}' not found: {resolved}. "
                f"Set the matching ROUTE_*_MODEL_PATH env var or download "
                f"the GGUF into the configured location."
            )
        default_path = paths[DEFAULT_ROUTE].expanduser().resolve()
        if not default_path.exists():
            raise FileNotFoundError(
                f"Neither route '{route}' nor the default route's model file "
                f"is present (default: {default_path}). At least the default "
                f"GGUF must exist on disk."
            )
        return ModelSpec(
            route=DEFAULT_ROUTE,
            model_path=default_path,
            n_ctx=int(cfg.route_n_ctx),
            n_gpu_layers=int(cfg.route_n_gpu_layers),
        )

    return ModelSpec(
        route=route,
        model_path=resolved,
        n_ctx=int(cfg.route_n_ctx),
        n_gpu_layers=int(cfg.route_n_gpu_layers),
    )
