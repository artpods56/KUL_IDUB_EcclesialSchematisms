import pytest

from notarius_api.v1.routes.auth.abuse import AuthAbuseControl


@pytest.mark.asyncio
async def test_abuse_windows_and_outstanding_logins_are_deterministic() -> None:
    current = 100.0

    def clock() -> float:
        return current

    abuse = AuthAbuseControl(
        window_seconds=10,
        login_start_limit=2,
        callback_limit=1,
        session_failure_limit=1,
        pat_creation_limit=1,
        outstanding_login_limit=2,
        outstanding_login_ttl_seconds=20,
        clock=clock,
    )

    assert await abuse.allow_login_start("browser")
    assert await abuse.allow_login_start("browser")
    assert not await abuse.allow_login_start("browser")
    assert await abuse.reserve_login("browser")
    assert await abuse.reserve_login("browser")
    assert not await abuse.reserve_login("browser")

    current += 10
    assert await abuse.allow_login_start("browser")
    assert await abuse.allow_callback("browser")
    assert not await abuse.allow_callback("browser")
    await abuse.release_login("browser")
    assert await abuse.reserve_login("browser")

    current += 20
    assert await abuse.reserve_login("browser")
    assert await abuse.allow_session_failure("browser")
    assert not await abuse.allow_session_failure("browser")
