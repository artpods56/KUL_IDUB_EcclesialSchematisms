import base64
import hashlib
from datetime import UTC, datetime, timedelta
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from authlib.jose import JsonWebKey, JsonWebToken
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.services import (
    AuthService,
    OidcProtocolError,
    ProviderMetadata,
)
from notarius_core.application.identity import IdentityService
from notarius_core.domain.identity import OidcLoginTransaction, User, Workspace
from notarius_core.ports.identity import IdentityUnitOfWorkPort
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


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


def _signed_token(
    private_key: bytes,
    claims: dict[str, object],
    *,
    algorithm: str = "RS256",
) -> str:
    jwk = JsonWebKey.import_key(private_key, {"kid": "test", "alg": algorithm})
    encoded = JsonWebToken([algorithm]).encode(
        {"alg": algorithm, "kid": "test"},
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


def test_oidc_transaction_return_path_is_sensitive_state() -> None:
    sentinel = "/after-login?one-time=return-path-sentinel"
    transaction = OidcLoginTransaction(
        state_digest=b"state-digest",
        nonce_digest=b"nonce-digest",
        encrypted_pkce_verifier=b"encrypted-verifier",
        pkce_key_version=1,
        return_path=sentinel,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    assert sentinel not in repr(transaction)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"nonce": None},
        {"nonce": "different-nonce"},
        {"exp": None},
        {"iat": None},
        {"exp": int((datetime.now(UTC) - timedelta(minutes=2)).timestamp())},
        {"iat": int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())},
        {"nbf": int((datetime.now(UTC) + timedelta(minutes=2)).timestamp())},
        {"iss": "https://other-issuer.example.test"},
        {"aud": "other-client"},
        {"azp": "other-client"},
        {"aud": ["notarius-client", "second-client"]},
        {"exp": float("nan")},
        {"iat": True},
        {"nbf": float("inf")},
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
async def test_id_token_rejects_disallowed_algorithm_and_invalid_signature() -> None:
    auth = _auth_service()
    private_key, public_keys = _keys()
    auth._jwks = public_keys
    auth._jwks_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    with pytest.raises(OidcProtocolError):
        await auth._validate_id_token(
            _signed_token(private_key, _claims(), algorithm="PS256"),
            nonce_digest=auth.digest_secret("expected-nonce"),
        )
    other_private_key, _ = _keys()
    with pytest.raises(OidcProtocolError):
        await auth._validate_id_token(
            _signed_token(other_private_key, _claims()),
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


class _UnavailableProviderClient:
    def __init__(self, **kwargs: object) -> None:
        del kwargs

    async def __aenter__(self) -> "_UnavailableProviderClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def get(self, url: str) -> httpx.Response:
        request = httpx.Request("GET", url)
        raise httpx.ConnectError("provider unavailable", request=request)


@pytest.mark.asyncio
async def test_provider_and_jwks_caches_fail_closed_only_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _auth_service()
    monkeypatch.setattr(httpx, "AsyncClient", _UnavailableProviderClient)

    auth._provider_metadata = ProviderMetadata(
        issuer="https://issuer.example.test",
        authorization_endpoint="https://issuer.example.test/authorize",
        token_endpoint="https://issuer.example.test/token",
        jwks_uri="https://issuer.example.test/keys",
    )
    auth._provider_metadata_expires_at = datetime.now(UTC) + timedelta(minutes=1)
    assert await auth._provider() == auth._provider_metadata

    auth._provider_metadata_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(OidcProtocolError) as discovery_error:
        await auth._provider()
    assert discovery_error.value.code == "provider_discovery_unavailable"

    auth._provider_metadata_expires_at = datetime.now(UTC) + timedelta(minutes=1)
    auth._jwks = [{"kty": "RSA", "kid": "test"}]
    auth._jwks_expires_at = datetime.now(UTC) + timedelta(minutes=1)
    assert await auth._keys() == auth._jwks
    auth._jwks_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(OidcProtocolError) as keys_error:
        await auth._keys()
    assert keys_error.value.code == "provider_keys_unavailable"


@pytest.mark.asyncio
async def test_malformed_token_endpoint_response_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _auth_service()

    auth._provider_metadata = ProviderMetadata(
        issuer="https://issuer.example.test",
        authorization_endpoint="https://issuer.example.test/authorize",
        token_endpoint="https://issuer.example.test/token",
        jwks_uri="https://issuer.example.test/keys",
    )
    auth._provider_metadata_expires_at = datetime.now(UTC) + timedelta(minutes=1)

    class MalformedTokenClient(_UnavailableProviderClient):
        async def post(self, url: str, data: dict[str, str]) -> httpx.Response:
            del data
            return httpx.Response(
                200,
                text="provider sentinel payload",
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", MalformedTokenClient)
    with pytest.raises(OidcProtocolError) as error:
        await auth._exchange_code(
            code="provider-code",
            verifier="verifier",
            nonce_digest=auth.digest_secret("expected-nonce"),
        )
    assert error.value.code == "invalid_token_response"
    assert "provider sentinel payload" not in repr(error.value)


@pytest.mark.asyncio
async def test_protocol_issuer_successfully_provisions_identity_and_rotates_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private_key, public_keys = _keys()
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'oidc-protocol.sqlite3'}"
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    identity_service = IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))
    local_workspace = Workspace.shared(slug="local", name="Local workspace")
    old_user = User(id=UUID(int=99), email="old@example.test", display_name="Old")
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_workspace(local_workspace)
        await unit_of_work.identity.add_user(old_user)
        await unit_of_work.commit()
    await identity_service.bootstrap_oidc_owner(
        issuer="https://issuer.example.test",
        subject="provider-user",
    )

    settings = Settings(
        public_origin="https://app.example.test",
        oidc_issuer="https://issuer.example.test",
        oidc_client_id="notarius-client",
        oidc_auth_wrapping_key=SecretStr("protocol-wrapping-key"),
        auth_cookie_secure=False,
        database_url=SecretStr(database_url),
        execution_backend="inline",
    )
    auth = AuthService(
        settings=settings,
        unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
        identity_service=identity_service,
    )
    protocol_state: dict[str, object] = {}

    class ProtocolIssuerClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def __aenter__(self) -> "ProtocolIssuerClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def get(self, url: str) -> httpx.Response:
            if url.endswith("/.well-known/openid-configuration"):
                return httpx.Response(
                    200,
                    json={
                        "issuer": "https://issuer.example.test",
                        "authorization_endpoint": "https://issuer.example.test/authorize",
                        "token_endpoint": "https://issuer.example.test/token",
                        "jwks_uri": "https://issuer.example.test/keys",
                    },
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(
                200,
                json={"keys": public_keys},
                request=httpx.Request("GET", url),
            )

        async def post(self, url: str, data: dict[str, str]) -> httpx.Response:
            verifier = data["code_verifier"]
            challenge = parse_qs(urlsplit(cast(str, protocol_state["url"])).query)[
                "code_challenge"
            ][0]
            assert (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .rstrip(b"=")
                .decode()
                == challenge
            )
            nonce = parse_qs(urlsplit(cast(str, protocol_state["url"])).query)["nonce"][
                0
            ]
            claims = _claims(nonce=nonce, sub="provider-user")
            token = _signed_token(private_key, claims)
            return httpx.Response(
                200,
                json={"id_token": token},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", ProtocolIssuerClient)
    old_session = await auth.issue_session(old_user.id)
    tampered_url, tampered_id = await auth.start_login(return_path="/")
    tampered_params = parse_qs(urlsplit(tampered_url).query)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        tampered = await unit_of_work.identity.lock_login_transaction(tampered_id)
        assert tampered is not None
        tampered.encrypted_pkce_verifier = bytes(
            byte ^ 0xFF for byte in tampered.encrypted_pkce_verifier
        )
        await unit_of_work.commit()
    with pytest.raises(OidcProtocolError) as tampered_error:
        await auth.callback(
            transaction_id_value=str(tampered_id),
            state=tampered_params["state"][0],
            code="protocol-code",
            error=None,
        )
    assert tampered_error.value.code == "invalid_pkce_verifier"
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        consumed_tampered = await unit_of_work.identity.lock_login_transaction(
            tampered_id
        )
        assert consumed_tampered is not None and consumed_tampered.is_consumed

    callback_failures: list[tuple[UUID, str | None, str]] = []
    for callback_state in ("matching", "missing", "mismatched"):
        failure_url, failure_id = await auth.start_login(return_path="/")
        failure_params = parse_qs(urlsplit(failure_url).query)
        failure_state: str | None = failure_params["state"][0]
        if callback_state == "missing":
            failure_state = None
        elif callback_state == "mismatched":
            failure_state = "wrong-state"
        callback_failures.append((failure_id, failure_state, callback_state))
        with pytest.raises(OidcProtocolError) as failure:
            await auth.callback(
                transaction_id_value=str(failure_id),
                state=failure_state,
                code=None,
                error="access_denied",
            )
        assert failure.value.code == (
            "invalid_state"
            if callback_state == "mismatched"
            else "provider_callback_failed"
        )
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            consumed = await unit_of_work.identity.lock_login_transaction(failure_id)
            assert consumed is not None and consumed.is_consumed
    for failure_id, failure_state, _ in callback_failures:
        with pytest.raises(OidcProtocolError):
            await auth.callback(
                transaction_id_value=str(failure_id),
                state=failure_state,
                code=None,
                error="access_denied",
            )
    authorization_url, transaction_id = await auth.start_login(
        return_path="/after-login"
    )
    protocol_state["url"] = authorization_url
    params = parse_qs(urlsplit(authorization_url).query)
    provisioned, replacement, return_path = await auth.callback(
        transaction_id_value=str(transaction_id),
        state=params["state"][0],
        code="protocol-code",
        error=None,
        current_session_cookie=old_session.cookie_value,
    )

    assert provisioned.user.email is None
    assert return_path == "/after-login"
    assert replacement.cookie_value != old_session.cookie_value
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        old_stored = await unit_of_work.identity.get_auth_session(
            old_session.session.id
        )
        replacement_stored = await unit_of_work.identity.get_auth_session(
            replacement.session.id
        )
        assert old_stored is not None and old_stored.is_revoked
        assert replacement_stored is not None and not replacement_stored.is_revoked
    async with database.engine.connect() as connection:
        operations = (
            await connection.execute(
                text(
                    "SELECT operation FROM security_audit_events "
                    "WHERE operation = 'oidc.callback.success'"
                )
            )
        ).all()
    assert operations
    with pytest.raises(OidcProtocolError):
        await auth.callback(
            transaction_id_value=str(transaction_id),
            state=params["state"][0],
            code="protocol-code",
            error=None,
        )
    await database.dispose()
