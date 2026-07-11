import argparse
import json
import time

from ocr_script_support import ApiClient, api_base_url_default, object_field


TERMINAL_WORKFLOW_RUN_STATUSES = {
    "succeeded",
    "failed_retryable",
    "failed_permanent",
    "cancelled",
}


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)
    snapshots = []
    latest = None
    for poll_index in range(args.poll_count):
        latest = client.request_object(
            "GET",
            f"/v1/workflow-runs/{args.workflow_run_id}/events",
        )
        workflow_run = object_field(latest, "workflow_run")
        events = latest["events"]
        if not isinstance(events, list):
            raise RuntimeError("Workflow run events response field events is not a list")
        snapshots.append(
            {
                "poll": poll_index + 1,
                "workflow_run_status": workflow_run["status"],
                "event_count": len(events),
            }
        )
        if workflow_run["status"] in TERMINAL_WORKFLOW_RUN_STATUSES:
            break
        if poll_index + 1 < args.poll_count:
            time.sleep(args.interval_seconds)

    print(
        json.dumps(
            {
                "workflow_run_id": args.workflow_run_id,
                "poll_count": len(snapshots),
                "snapshots": snapshots,
                "timeline": latest,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read or poll a workflow run event timeline through the HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    parser.add_argument("workflow_run_id")
    parser.add_argument("--poll-count", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.poll_count < 1:
        parser.error("--poll-count must be at least 1")
    if args.interval_seconds < 0:
        parser.error("--interval-seconds must not be negative")
    return args


if __name__ == "__main__":
    main()
