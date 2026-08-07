from uuid import UUID

import pytest
from pydantic import ValidationError

from notarius_mcp.settings import Settings


WORKSPACE_ID = UUID("11111111-2222-3333-4444-555555555555")


def test_settings_default_to_local_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTARIUS_MCP_API_URL", raising=False)
    monkeypatch.delenv("NOTARIUS_MCP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("NOTARIUS_MCP_WORKSPACE_ID", str(WORKSPACE_ID))

    settings = Settings()

    assert str(settings.api_url) == "http://127.0.0.1:8000/"
    assert settings.workspace_id == WORKSPACE_ID
    assert settings.timeout_seconds == 15.0


def test_settings_read_mcp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTARIUS_MCP_API_URL", "http://api.internal:9000")
    monkeypatch.setenv("NOTARIUS_MCP_WORKSPACE_ID", str(WORKSPACE_ID))
    monkeypatch.setenv("NOTARIUS_MCP_TIMEOUT_SECONDS", "2.5")

    settings = Settings()

    assert str(settings.api_url) == "http://api.internal:9000/"
    assert settings.workspace_id == WORKSPACE_ID
    assert settings.timeout_seconds == 2.5


def test_settings_require_workspace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTARIUS_MCP_WORKSPACE_ID", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("timeout_seconds", [0, -0.1])
def test_settings_require_positive_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        Settings(workspace_id=WORKSPACE_ID, timeout_seconds=timeout_seconds)
