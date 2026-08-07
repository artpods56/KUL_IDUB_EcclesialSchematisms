from datetime import UTC, datetime, timedelta
from collections.abc import Callable
from typing import cast

import pytest
from authlib.jose import JsonWebKey, JsonWebToken
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.services import AuthService, OidcProtocolError
from notarius_core.application.identity import IdentityService
from notarius_core.ports.identity import IdentityUnitOfWorkPort


def _auth_service() -> AuthService:
    settings = Settings(
        public_origin="https://app.example.test",
        oidc_issuer="https://issuer.example.test",
        oidc_client_id="notarius-client",
        oidc_auth_wrapping_key=SecretStr("test-wrapping-key"),
        execution_backend="inline",
    )
    return AuthService(
        settings=settings,
        unit_of_work_factory=cast(Callable[[], IdentityUnitOfWorkPort], lambda: None),
        identity_service=IdentityService(
            cast(Callable[[], IdentityUnitOfWorkPort], lambda: None)
        ),
    )


def _signed_token(private_key: bytes, claims: dict[str, object]) -> str:
    jwk = JsonWebKey.import_key(private_key, {"kid": "test", "alg": "RS256"})
    encoded = JsonWebToken(["RS256"]).encode(
        {"alg": "RS256", "kid": "test"},
        claims,
        jwk,
    )
    return encoded.decode("ascii")


def _keys() -> tuple[bytes, list[object]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    private_jwk = JsonWebKey.import_key(
        private_bytes,
        {"kid": "test", "alg": "RS256"},
    )
    return private_bytes, [private_jwk.as_dict(is_private=False)]


def _claims(**overrides: object) -> dict[str, object]:
    now = int(datetime.now(UTC).timestamp())
    claims: dict[str, object] = {
        "iss": "https://issuer.example.test",
        "aud": "notarius-client",
        "sub": "provider-user",
        "nonce": "expected-nonce",
        "exp": now + 300,
        "iat": now,
    }
    claims.update(overrides)
    return claims


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"nonce": None},
        {"nonce": "different-nonce"},
        {"exp": None},
        {"exp": int((datetime.now(UTC) - timedelta(minutes=2)).timestamp())},
        {"iat": int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())},
        {"nbf": int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())},
    ],
)
async def test_id_token_rejects_nonce_and_required_time_claim_failures(
    overrides: dict[str, object],
) -> None:
    auth = _auth_service()
    private_key, public_keys = _keys()
    auth._jwks = public_keys
    auth._jwks_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    token = _signed_token(private_key, _claims(**overrides))

    with pytest.raises(OidcProtocolError):
        await auth._validate_id_token(
            token,
            nonce_digest=auth.digest_secret("expected-nonce"),
        )


@pytest.mark.asyncio
async def test_unknown_signing_key_gets_one_forced_refresh_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _auth_service()
    _, old_public_keys = _keys()
    new_private_key, new_public_keys = _keys()
    token = _signed_token(new_private_key, _claims())
    calls: list[bool] = []

    async def keys(*, force_refresh: bool = False) -> list[object]:
        calls.append(force_refresh)
        return new_public_keys if force_refresh else old_public_keys

    monkeypatch.setattr(auth, "_keys", keys)
    claims = await auth._validate_id_token(
        token,
        nonce_digest=auth.digest_secret("expected-nonce"),
    )

    assert claims["sub"] == "provider-user"
    assert calls == [False, True]

    async def unavailable(*, force_refresh: bool = False) -> list[object]:
        del force_refresh
        raise OidcProtocolError("provider_keys_unavailable")

    monkeypatch.setattr(auth, "_keys", unavailable)
    with pytest.raises(OidcProtocolError):
        await auth._validate_id_token(
            token,
            nonce_digest=auth.digest_secret("expected-nonce"),
        )
