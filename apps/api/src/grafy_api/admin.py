"""Operator-only identity bootstrap commands."""

import argparse
import asyncio
from uuid import UUID

from grafy_api.settings import get_settings
from grafy_core.application.identity import IdentityService
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    database = create_database(settings.resolved_database_url)
    try:
        service = IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))
        if args.command == "disable-user":
            await service.disable_user(user_id=UUID(args.user_id))
            return
        raise ValueError("Unsupported admin command")
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="grafy-admin")
    commands = parser.add_subparsers(dest="command", required=True)

    disable = commands.add_parser("disable-user")
    disable.add_argument("user_id")

    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
