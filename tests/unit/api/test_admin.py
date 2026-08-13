import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from notarius_api import admin


def test_bootstrap_owner_defaults_to_configured_oidc_issuer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_issuer = "https://issuer.example.com"
    settings = SimpleNamespace(
        oidc_issuer=configured_issuer,
        resolved_database_url="sqlite+aiosqlite:///:memory:",
    )
    database = SimpleNamespace(sessions=object(), dispose=AsyncMock())
    identity_service = SimpleNamespace(bootstrap_oidc_owner=AsyncMock())
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    monkeypatch.setattr(admin, "create_database", lambda _database_url: database)
    monkeypatch.setattr(
        admin,
        "IdentityService",
        lambda _unit_of_work: identity_service,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "notarius-admin",
            "bootstrap-oidc-owner",
            "--subject",
            "first-owner-subject",
        ],
    )

    admin.main()

    identity_service.bootstrap_oidc_owner.assert_awaited_once_with(
        issuer=configured_issuer,
        subject="first-owner-subject",
    )
    database.dispose.assert_awaited_once_with()
