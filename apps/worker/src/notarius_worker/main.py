"""Entry point for the Notarius Studio background worker."""

import asyncio

from notarius_worker.streaming import create_app


async def main() -> None:
    app = create_app()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())

