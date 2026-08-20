"""Shared pytest fixtures for the Grafy test suite."""

import pytest

from tests.support.identity import TEST_COMMAND_HMAC_KEY


@pytest.fixture(autouse=True)
def _disable_single_api_owner_by_default(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Unit tests create many short-lived apps; the owner lease is opt-in."""

    if request.node.get_closest_marker("single_api_owner") is not None:
        return
    monkeypatch.setenv("GRAFY_REQUIRE_SINGLE_API_OWNER", "false")


@pytest.fixture(autouse=True)
def _configure_command_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAFY_COMMAND_HMAC_KEY", TEST_COMMAND_HMAC_KEY)
    monkeypatch.setenv("GRAFY_COMMAND_HMAC_KEY_VERSION", "1")