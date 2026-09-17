"""Model profiles, and whether a given machine can run one."""

from unittest.mock import patch

import pytest

from rag.config import load_config
from rag.routing.profiles import (
    PROFILES,
    Hardware,
    check_fit,
    detect_hardware,
    get_profile,
    recommend_profile,
)
from rag.routing.registry import route_model_paths
from rag.routing.types import ROUTES


def test_every_profile_covers_every_route():
    for name, profile in PROFILES.items():
        assert set(profile.models) == set(ROUTES), f"{name} is missing a route"


def test_profiles_get_larger_in_the_order_they_are_named():
    light, balanced, heavy = (PROFILES[n] for n in ("light", "balanced", "heavy"))
    assert light.peak_vram_gb < balanced.peak_vram_gb < heavy.peak_vram_gb
    assert light.total_download_gb < balanced.total_download_gb < heavy.total_download_gb


def test_the_download_total_counts_a_shared_model_once():
    # Three routes share one model in every profile, so summing per route
    # would roughly double the figure people are shown.
    light = PROFILES["light"]
    per_route = sum(m.size_gb for m in light.models.values())
    assert light.total_download_gb < per_route


def test_vram_allows_for_more_than_the_weights():
    choice = PROFILES["light"].models["default"]
    assert choice.vram_gb > choice.size_gb


def test_custom_is_not_a_profile():
    # "custom" means the per-route paths are used verbatim.
    assert get_profile("custom") is None
    assert get_profile("nonsense") is None


@pytest.mark.parametrize(
    "vram, expected",
    [(None, "light"), (2.0, "light"), (4.0, "light"), (8.0, "balanced"), (24.0, "heavy")],
)
def test_the_recommendation_follows_the_available_vram(vram, expected):
    hw = Hardware(vram_gb=vram, ram_gb=32.0, free_disk_gb=200.0)
    assert recommend_profile(hw) == expected


def test_no_gpu_recommends_light_rather_than_the_biggest_that_would_run():
    # Anything runs on CPU eventually. Recommending a 14B to someone with no
    # GPU is technically true and useless.
    assert recommend_profile(Hardware(None, 64.0, 500.0)) == "light"


def test_a_machine_with_room_gets_no_warnings():
    assert check_fit("heavy", Hardware(vram_gb=24.0, ram_gb=64.0, free_disk_gb=500.0)) == []


def test_too_little_vram_warns_but_does_not_refuse():
    warnings = check_fit("heavy", Hardware(vram_gb=4.0, ram_gb=32.0, free_disk_gb=500.0))
    assert any("VRAM" in w for w in warnings)
    assert any("Partial offload" in w for w in warnings)


def test_too_little_disk_is_called_out():
    warnings = check_fit("heavy", Hardware(vram_gb=24.0, ram_gb=64.0, free_disk_gb=2.0))
    assert any("free on disk" in w for w in warnings)


def test_no_gpu_says_so_plainly():
    warnings = check_fit("balanced", Hardware(vram_gb=None, ram_gb=32.0, free_disk_gb=500.0))
    assert any("CPU" in w for w in warnings)


def test_checking_a_custom_profile_warns_about_nothing():
    assert check_fit("custom", Hardware(1.0, 1.0, 1.0)) == []


def test_detection_survives_a_machine_with_no_nvidia_smi():
    with patch("rag.routing.profiles.shutil.which", return_value=None):
        hw = detect_hardware()
    assert hw.vram_gb is None
    # The other two are still worth reporting.
    assert hw.free_disk_gb is not None


def test_describe_does_not_print_none():
    assert "None" not in Hardware(None, None, None).describe()


def test_a_profile_overrides_the_per_route_paths(monkeypatch):
    monkeypatch.setenv("MODEL_PROFILE", "light")
    paths = route_model_paths(load_config(reload=True))
    assert all("3b" in p.name.lower() for p in paths.values()), paths


def test_custom_keeps_the_per_route_paths(monkeypatch):
    monkeypatch.setenv("MODEL_PROFILE", "custom")
    monkeypatch.setenv("ROUTE_MATH_MODEL_PATH", "/tmp/my-math-model.gguf")
    paths = route_model_paths(load_config(reload=True))
    assert paths["math"].name == "my-math-model.gguf"
