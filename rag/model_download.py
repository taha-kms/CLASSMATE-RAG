"""
Fetching GGUF model files, with enough feedback to tell it apart from a hang.

The first `rag add` on a clean install took ten minutes and printed nothing
while it pulled the embedding model, which is indistinguishable from a
crash. Anything that downloads gigabytes needs to say so.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from rag.config import load_config, resolve_data_path

log = logging.getLogger(__name__)


def models_dir() -> Path:
    """
    Where GGUF files live. Resolved against the project root rather than the
    working directory, so running the CLI from elsewhere does not start a
    second, empty models directory (#15).
    """
    return resolve_data_path("./models")


def hf_token() -> str | None:
    """The Hugging Face token, under any of the three names accepted."""
    cfg = load_config()
    return cfg.hf_token or None


@dataclass(frozen=True)
class LocalModel:
    path: Path
    size_gb: float


def list_local_models() -> list[LocalModel]:
    """Every .gguf under the models directory, largest first."""
    root = models_dir()
    if not root.exists():
        return []
    found = [
        LocalModel(path=p, size_gb=round(p.stat().st_size / (1024**3), 2))
        for p in sorted(root.rglob("*.gguf"))
        if p.is_file()
    ]
    return sorted(found, key=lambda m: m.size_gb, reverse=True)


class DownloadError(RuntimeError):
    """Raised with something a person can act on."""


def _hf_download(**kwargs) -> str:
    """
    Thin wrapper around hf_hub_download.

    Exists so the call can be replaced in tests without huggingface_hub
    being installed, which keeps it out of the CI test requirements.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _remote_size_gb(repo_id: str, filename: str, token: str | None) -> float | None:
    """Size of the remote file, or None when it cannot be determined."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except Exception:  # noqa: BLE001 - size is a courtesy, not a requirement
        return None

    for sibling in info.siblings or []:
        if sibling.rfilename == filename and sibling.size:
            return round(sibling.size / (1024**3), 2)
    return None


def _explain(error: Exception, repo_id: str, filename: str) -> str:
    """Turn a hub exception into advice rather than a status code."""
    text = str(error)
    lowered = text.lower()
    url = f"https://huggingface.co/{repo_id}"

    if "401" in text or "unauthorized" in lowered:
        return f"{repo_id} needs authentication.\nSet HF_TOKEN to a token from https://huggingface.co/settings/tokens."
    if "403" in text or "gated" in lowered or "awaiting" in lowered:
        return (
            f"{repo_id} is gated: access has to be granted before it can be downloaded.\n"
            f"Accept the licence at {url} with the same account your HF_TOKEN belongs to,\n"
            f"then run this again."
        )
    if "404" in text or "not found" in lowered or "entrynotfound" in lowered:
        return f"No file called {filename} in {repo_id}.\nCheck the exact filename under Files and versions at {url}."
    return f"Could not download {filename} from {repo_id}: {text}"


def download_model(
    repo_id: str,
    filename: str,
    *,
    token: str | None = None,
    check_space: bool = True,
) -> Path:
    """
    Fetch one file into the models directory and return where it landed.

    Downloads are resumable, so an interrupted one continues rather than
    restarting, which matters at these sizes.
    """
    token = token or hf_token()
    target_dir = models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = target_dir / filename
    if existing.is_file():
        log.info("%s is already here, nothing to download", filename)
        return existing

    size_gb = _remote_size_gb(repo_id, filename, token)

    if check_space:
        free_gb = round(shutil.disk_usage(target_dir).free / (1024**3), 1)
        if size_gb is not None and free_gb < size_gb + 0.5:
            # Better to refuse now than to die at 95% of a 4 GB download.
            raise DownloadError(
                f"Not enough disk space: {filename} is {size_gb} GB and {free_gb} GB is free at {target_dir}."
            )

    if size_gb is not None:
        log.info("Downloading %s from %s (%.2f GB)", filename, repo_id, size_gb)
    else:
        log.info("Downloading %s from %s", filename, repo_id)

    try:
        # Progress goes to stderr, so stdout stays parseable.
        path = _hf_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            token=token,
        )
    except Exception as e:
        raise DownloadError(_explain(e, repo_id, filename)) from e

    resolved = Path(path).resolve()
    log.info("Saved to %s", resolved)
    return resolved


def download_profile(name: str, *, token: str | None = None) -> list[Path]:
    """
    Fetch every distinct model a profile needs.

    Several routes share a model, so this downloads each file once rather
    than once per route.
    """
    from rag.routing.profiles import get_profile

    profile = get_profile(name)
    if profile is None:
        raise DownloadError(
            f"'{name}' is not a profile with models of its own. Try light, balanced or heavy, or see `rag profiles`."
        )

    wanted = {(c.repo_id, c.filename) for c in profile.models.values()}
    log.info("Profile %s needs %d file(s), %.1f GB total", name, len(wanted), profile.total_download_gb)

    return [download_model(repo, fn, token=token) for repo, fn in sorted(wanted)]
