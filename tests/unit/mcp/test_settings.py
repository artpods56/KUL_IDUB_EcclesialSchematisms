import pytest
from pydantic import ValidationError

from notarius_mcp.settings import Settings


def test_settings_default_to_local_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTARIUS_MCP_API_URL", raising=False)
    monkeypatch.delenv("NOTARIUS_MCP_TIMEOUT_SECONDS", raising=False)

    settings = Settings()

    assert str(settings.api_url) == "http://127.0.0.1:8000/"
    assert settings.timeout_seconds == 15.0


def test_settings_read_mcp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTARIUS_MCP_API_URL", "http://api.internal:9000")
    monkeypatch.setenv("NOTARIUS_MCP_TIMEOUT_SECONDS", "2.5")

    settings = Settings()

    assert str(settings.api_url) == "http://api.internal:9000/"
    assert settings.timeout_seconds == 2.5


@pytest.mark.parametrize("timeout_seconds", [0, -0.1])
def test_settings_require_positive_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        Settings(timeout_seconds=timeout_seconds)
