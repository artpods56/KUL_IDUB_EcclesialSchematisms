import argparse
import asyncio
import json
import os

from notarius_persistence.unit_of_work import create_sqlite_uow_factory
from notarius_core.application.operators import builtin_node_specs
from notarius_worker.dependencies import get_artifact_payload_storage
from notarius_worker.node_execution import NodeRunExecutor
from notarius_worker.operators import builtin_node_handlers
from notarius_worker.outbox import (
    LocalNodeRunOutboxDrainer,
    LocalNodeRunOutboxDrainResult,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drain pending node-run outbox messages without NATS.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("NOTARIUS_DATABASE_URL"),
        help="SQLAlchemy database URL shared with the Notarius API process.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=100,
        help="Maximum node-run outbox messages to drain.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Drain a single batch instead of looping until no node-run messages remain.",
    )
    args = parser.parse_args()

    if args.database_url is None:
        parser.error(
            "Set NOTARIUS_DATABASE_URL or pass --database-url. "
            "A separate script process cannot drain the API's in-memory store."
        )
    if args.max_messages < 1:
        parser.error("--max-messages must be greater than 0")

    result = asyncio.run(
        drain_node_run_outbox(
            database_url=args.database_url,
            max_messages=args.max_messages,
            once=args.once,
        )
    )
    print(json.dumps(result_to_json(result), indent=2))
    if result.errors:
        raise SystemExit(1)


async def drain_node_run_outbox(
    database_url: str,
    max_messages: int,
    once: bool,
) -> LocalNodeRunOutboxDrainResult:
    uow_factory = create_sqlite_uow_factory(database_url)
    executor = NodeRunExecutor(
        uow_factory,
        builtin_node_handlers(get_artifact_payload_storage()),
        builtin_node_specs(),
    )
    drainer = LocalNodeRunOutboxDrainer(uow_factory, executor)
    if once:
        return await drainer.drain_once(max_messages=max_messages)
    return await drainer.drain_until_idle(max_messages=max_messages)


def result_to_json(result: LocalNodeRunOutboxDrainResult) -> dict[str, object]:
    return {
        "processed_message_ids": [
            str(message_id) for message_id in result.processed_message_ids
        ],
        "processed_node_run_ids": [
            str(node_run_id) for node_run_id in result.processed_node_run_ids
        ],
        "errors": [
            {
                "outbox_message_id": str(error.outbox_message_id),
                "error": error.error,
            }
            for error in result.errors
        ],
    }


if __name__ == "__main__":
    main()
