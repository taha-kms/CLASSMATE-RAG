"""Suite-wide safety net.

The settings code writes to the project's real ``.env`` and credential file.
A test that sends a write and forgets to redirect them would edit the
developer's actual configuration, and a mutation or a regression that disables
a guard is exactly when such a test stops being a harmless "expect 400" and
starts writing. That happened once: a Host-check test changed K_VECTOR in a
real ``.env`` the moment the check was broken on purpose.

So every test gets throwaway locations by default. Tests that need particular
contents override these with their own monkeypatch, which runs later and wins.
"""

import pytest


@pytest.fixture(autouse=True)
def _never_touch_the_real_configuration(monkeypatch, tmp_path_factory):
    from rag import secrets, settings

    scratch = tmp_path_factory.mktemp("config")
    monkeypatch.setattr(settings, "env_file_path", lambda: scratch / ".env")
    monkeypatch.setattr(secrets, "secrets_path", lambda: scratch / ".secrets.env")
