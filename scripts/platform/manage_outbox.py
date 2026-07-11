import argparse
import json
from pathlib import Path

from ocr_script_support import ApiClient, api_base_url_default


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)

    if args.command == "list":
        query = {
            "status": args.status,
            "limit": args.limit,
            "offset": args.offset,
        }
        if args.subject_prefix is not None:
            query["subject_prefix"] = args.subject_prefix
        messages = client.request_json(
            "GET",
            "/v1/outbox-messages",
            query=query,
        )
        if not isinstance(messages, list):
            raise RuntimeError("Outbox list response is not a JSON array")
        print(
            json.dumps(
                {
                    "status": args.status,
                    "subject_prefix": args.subject_prefix,
                    "limit": args.limit,
                    "offset": args.offset,
                    "messages": messages,
                },
                indent=2,
            )
        )
        return

    if args.command == "dlq-summary":
        query = {}
        if args.status is not None:
            query["status"] = args.status
        if args.consumer_name is not None:
            query["consumer_name"] = args.consumer_name
        if args.error_code is not None:
            query["error_code"] = args.error_code
        if args.original_subject is not None:
            query["original_subject"] = args.original_subject
        summaries = client.request_json(
            "GET",
            "/v1/outbox-messages/dlq-summary",
            query=query,
        )
        if not isinstance(summaries, list):
            raise RuntimeError("DLQ summary response is not a JSON array")
        print(json.dumps({"summaries": summaries}, indent=2))
        return

    if args.command == "cleanup":
        statuses = args.status or ["published", "failed"]
        payload = {
            "statuses": statuses,
            "older_than": args.older_than,
            "subject_prefix": args.subject_prefix,
            "message_type": args.message_type,
            "dry_run": not args.execute,
        }
        archive_path = None
        archived_count = None
        if args.archive_path is not None and args.execute:
            preview_payload = {**payload, "dry_run": True}
            preview = client.request_object(
                "POST",
                "/v1/outbox-messages/cleanup",
                payload=preview_payload,
            )
            archive_path = write_cleanup_archive(
                args.archive_path,
                request_payload=preview_payload,
                cleanup=preview,
            )
            archived_count = preview["matched_count"]

        cleanup = client.request_object(
            "POST",
            "/v1/outbox-messages/cleanup",
            payload=payload,
        )
        if args.archive_path is not None and not args.execute:
            archive_path = write_cleanup_archive(
                args.archive_path,
                request_payload=payload,
                cleanup=cleanup,
            )
            archived_count = cleanup["matched_count"]

        result = {"cleanup": cleanup}
        if archive_path is not None:
            result["archive_path"] = str(archive_path)
            result["archived_count"] = archived_count
        print(json.dumps(result, indent=2))
        return

    outbox_message = client.request_object(
        "POST",
        f"/v1/outbox-messages/{args.outbox_message_id}/requeue",
    )
    print(json.dumps({"outbox_message": outbox_message}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and requeue Notarius outbox messages through the HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List outbox messages by status.")
    list_parser.add_argument(
        "--status",
        choices=("pending", "published", "failed"),
        default="pending",
    )
    list_parser.add_argument("--subject-prefix")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--offset", type=int, default=0)

    summary_parser = subparsers.add_parser(
        "dlq-summary",
        help="Group dead-letter messages by consumer, error code, and original subject.",
    )
    summary_parser.add_argument(
        "--status",
        choices=("pending", "published", "failed"),
    )
    summary_parser.add_argument("--consumer-name")
    summary_parser.add_argument("--error-code")
    summary_parser.add_argument("--original-subject")

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help=(
            "Preview or delete published/failed outbox messages older than an "
            "ISO timestamp."
        ),
    )
    cleanup_parser.add_argument(
        "--status",
        action="append",
        choices=("published", "failed"),
        help=(
            "Terminal status to target. Repeat to include both. "
            "Defaults to published and failed."
        ),
    )
    cleanup_parser.add_argument("--older-than", required=True)
    cleanup_parser.add_argument("--subject-prefix")
    cleanup_parser.add_argument("--message-type")
    cleanup_parser.add_argument(
        "--archive-path",
        type=Path,
        help=(
            "Write matched messages to this JSON file. With --execute, the "
            "archive is written from a dry-run preview before deletion."
        ),
    )
    cleanup_parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete matching messages instead of returning a dry-run preview.",
    )

    requeue_parser = subparsers.add_parser(
        "requeue",
        help="Requeue a terminal failed outbox message.",
    )
    requeue_parser.add_argument("outbox_message_id")
    return parser.parse_args()


def write_cleanup_archive(
    archive_path: Path,
    *,
    request_payload: dict[str, object],
    cleanup: dict[str, object],
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        json.dumps(
            {
                "request": request_payload,
                "cleanup": cleanup,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_path


if __name__ == "__main__":
    main()
