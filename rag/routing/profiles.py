"""
Named model profiles, so a machine can be matched to models that fit it.

Every route model is configurable individually, which is flexible and
useless as a starting point: it requires knowing which GGUFs exist, how
large they are, and how much of one a given card can hold. The defaults
were chosen for a machine with considerably more VRAM than the laptops
this project is aimed at, and came to roughly 18 GB of downloads (#11).

A profile is a named set of per-route models with honest size figures, so
"which of these can I actually run" has an answer before anything is
downloaded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

from .types import ROUTES, Route

# Profile names. "custom" means the per-route ROUTE_*_MODEL_PATH settings
# are used verbatim, which is the behaviour that existed before profiles.
ProfileName = str

PROFILE_NAMES: tuple[ProfileName, ...] = ("light", "balanced", "heavy", "custom")
DEFAULT_PROFILE: ProfileName = "custom"


@dataclass(frozen=True)
class ModelChoice:
    """One model in a profile, with what it costs to run."""

    repo_id: str
    filename: str
    #: Approximate on-disk size. Used for fit checks and for telling people
    #: what they are about to download, so it is deliberately a real number
    #: rather than a category.
    size_gb: float

    @property
    def vram_gb(self) -> float:
        """
        Roughly what full GPU offload needs: the weights plus room for the
        context window and llama.cpp's own overhead.
        """
        return round(self.size_gb + 0.8, 1)


@dataclass(frozen=True)
class Profile:
    """A named set of per-route models."""

    name: ProfileName
    summary: str
    models: dict[Route, ModelChoice] = field(default_factory=dict)

    @property
    def total_download_gb(self) -> float:
        """Total download for the distinct models this profile uses."""
        seen = {(m.repo_id, m.filename): m.size_gb for m in self.models.values()}
        return round(sum(seen.values()), 1)

    @property
    def peak_vram_gb(self) -> float:
        """
        VRAM needed for the largest single model. Only one is resident at a
        time, so the peak is what matters, not the sum.
        """
        return max((m.vram_gb for m in self.models.values()), default=0.0)


def _same(choice: ModelChoice) -> dict[Route, ModelChoice]:
    return dict.fromkeys(ROUTES, choice)


_QWEN3B = ModelChoice("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", 2.0)
_QWEN_CODER_3B = ModelChoice("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "qwen2.5-coder-3b-instruct-q4_k_m.gguf", 2.0)
_QWEN7B = ModelChoice("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf", 4.4)
_QWEN_CODER_7B = ModelChoice("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", 4.4)
_QWEN14B = ModelChoice("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf", 8.9)
_QWEN_CODER_14B = ModelChoice("Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", "qwen2.5-coder-14b-instruct-q4_k_m.gguf", 8.9)

PROFILES: dict[ProfileName, Profile] = {
    "light": Profile(
        name="light",
        summary="3B models. Runs on CPU or a 4 GB card. Modest answers, small download.",
        models={**_same(_QWEN3B), "code": _QWEN_CODER_3B},
    ),
    "balanced": Profile(
        name="balanced",
        summary="7B models. Wants 8 GB of VRAM to offload fully, or patience on CPU.",
        models={**_same(_QWEN7B), "code": _QWEN_CODER_7B},
    ),
    "heavy": Profile(
        name="heavy",
        summary="14B models. Needs 16 GB of VRAM. Best answers, largest download.",
        models={**_same(_QWEN14B), "code": _QWEN_CODER_14B},
    ),
}


def get_profile(name: ProfileName) -> Profile | None:
    """Look up a profile. Returns None for 'custom' and unknown names."""
    return PROFILES.get(name)


# ---------------------------------------------------------------------------
# What this machine can run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """What was detected. Any field may be None when it could not be read."""

    vram_gb: float | None
    ram_gb: float | None
    free_disk_gb: float | None

    def describe(self) -> str:
        def g(v: float | None) -> str:
            return f"{v:.1f} GB" if v is not None else "unknown"

        return f"VRAM {g(self.vram_gb)}, RAM {g(self.ram_gb)}, free disk {g(self.free_disk_gb)}"


def _detect_vram_gb() -> float | None:
    """Total VRAM of the first GPU, via nvidia-smi. None when there is none."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.splitlines()[0].strip() if out else ""
    try:
        return round(float(first) / 1024, 1)
    except ValueError:
        return None


def _detect_ram_gb() -> float | None:
    """Total RAM. Linux only; returns None elsewhere rather than guessing."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return round(pages * page_size / (1024**3), 1)


def _detect_free_disk_gb(path: str = ".") -> float | None:
    try:
        return round(shutil.disk_usage(path).free / (1024**3), 1)
    except OSError:
        return None


def detect_hardware(path: str = ".") -> Hardware:
    """Best-effort look at what this machine has."""
    return Hardware(
        vram_gb=_detect_vram_gb(),
        ram_gb=_detect_ram_gb(),
        free_disk_gb=_detect_free_disk_gb(path),
    )


def recommend_profile(hw: Hardware | None = None) -> ProfileName:
    """
    The largest profile this machine can hold on the GPU.

    Falls back to "light" when there is no usable GPU: a bigger model still
    runs on CPU, it just takes minutes per answer, and recommending that to
    someone who has not asked for it is not a kindness.
    """
    hw = hw or detect_hardware()
    vram = hw.vram_gb
    if vram is None:
        return "light"
    if vram >= PROFILES["heavy"].peak_vram_gb:
        return "heavy"
    if vram >= PROFILES["balanced"].peak_vram_gb:
        return "balanced"
    return "light"


def check_fit(name: ProfileName, hw: Hardware | None = None) -> list[str]:
    """
    Warnings about running `name` on this machine. Empty means it fits.

    Deliberately warnings rather than errors: a model too large for the GPU
    still works on CPU, and refusing to run would be wrong.
    """
    profile = get_profile(name)
    if profile is None:
        return []

    hw = hw or detect_hardware()
    warnings: list[str] = []

    if hw.free_disk_gb is not None and hw.free_disk_gb < profile.total_download_gb:
        warnings.append(f"{profile.total_download_gb} GB to download, {hw.free_disk_gb} GB free on disk.")

    if hw.vram_gb is None:
        warnings.append(f"No GPU detected, so this runs on CPU. Expect minutes per answer with {profile.name} models.")
    elif hw.vram_gb < profile.peak_vram_gb:
        warnings.append(
            f"{profile.peak_vram_gb} GB of VRAM needed to offload fully, "
            f"{hw.vram_gb} GB present. Partial offload only; lower "
            f"ROUTE_N_GPU_LAYERS until it fits."
        )

    if hw.ram_gb is not None and hw.ram_gb < profile.peak_vram_gb + 2:
        warnings.append(f"{hw.ram_gb} GB of RAM is tight for a {profile.peak_vram_gb} GB model on CPU.")

    return warnings
