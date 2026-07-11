import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


JsonObject = dict[str, object]


def request_json(
    method: str,
    path: str,
    payload: JsonObject | None = None,
) -> object:
    base_url = os.getenv("NOTARIUS_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            decoded = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc

    return json.loads(decoded)


def request_object(
    method: str,
    path: str,
    payload: JsonObject | None = None,
) -> JsonObject:
    parsed = request_json(method, path, payload)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} returned a non-object JSON response")
    return parsed


def main() -> None:
    workflow = request_object(
        "POST",
        "/v1/workflows",
        {
            "name": "Script debug workflow",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "hello from a Python script"},
                }
            ],
        },
    )
    version = request_object(
        "POST",
        f"/v1/workflows/{workflow['id']}/versions",
        {"change_note": "Created by scripts/platform/run_debug_workflow.py"},
    )
    run = request_object(
        "POST",
        "/v1/workflow-runs",
        {"workflow_version_id": version["id"]},
    )

    execution = request_object(
        "POST",
        f"/v1/workflow-runs/{run['id']}/execute",
        {"max_node_runs": 100},
    )
    errors = execution["errors"]
    if errors != []:
        raise RuntimeError(f"Workflow run {run['id']} failed: {errors}")

    summary = request_object("GET", f"/v1/workflow-runs/{run['id']}/summary")
    workflow_run = summary["workflow_run"]

    print(
        json.dumps(
            {
                "workflow_id": workflow["id"],
                "workflow_version_id": version["id"],
                "workflow_run_id": run["id"],
                "workflow_run_status": workflow_run["status"],
                "processed_node_run_ids": execution["processed_node_run_ids"],
                "node_run_status_counts": summary["node_run_status_counts"],
                "artifact_counts": summary["artifact_counts"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
