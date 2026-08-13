"""Operator-only identity bootstrap commands."""

import argparse
import asyncio
from uuid import UUID

from notarius_api.settings import get_settings
from notarius_core.application.identity import IdentityService
from notarius_persistence.database import create_database
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    database = create_database(settings.resolved_database_url)
    try:
        service = IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))
        if args.command == "bootstrap-oidc-owner":
            configured_issuer = settings.oidc_issuer
            if configured_issuer is None:
                raise SystemExit("OIDC is not configured")
            if args.issuer is not None and args.issuer != configured_issuer:
                raise SystemExit("Bootstrap issuer must equal configured OIDC issuer")
            await service.bootstrap_oidc_owner(
                issuer=configured_issuer,
                subject=args.subject,
            )
            return
        if args.command == "disable-user":
            await service.disable_user(user_id=UUID(args.user_id))
            return
        raise ValueError("Unsupported admin command")
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="notarius-admin")
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap-oidc-owner")
    bootstrap.add_argument(
        "--issuer",
        help="optional assertion; defaults to configured NOTARIUS_OIDC_ISSUER",
    )
    bootstrap.add_argument("--subject", required=True)

    disable = commands.add_parser("disable-user")
    disable.add_argument("user_id")

    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
