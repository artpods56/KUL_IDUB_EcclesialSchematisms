from uuid import UUID

from fastapi.testclient import TestClient


def _graph_payload(name: str = "Draft graph") -> dict[str, object]:
    return {
        "name": name,
        "nodes": [
            {
                "id": "source",
                "operator_id": "plugin.source",
                "operator_version": 1,
                "config": {"text": "draft"},
                "position": {"x": 10.0, "y": 20.0},
            },
            {
                "id": "target",
                "operator_id": "plugin.target",
                "operator_version": 1,
                "config": {},
                "position": {"x": 300.0, "y": 20.0},
            },
        ],
        "edges": [
            {
                "id": "source-to-target",
                "from_node": "source",
                "from_port": "result",
                "to_node": "target",
                "to_port": "value",
                "collection_mode": "map",
                "projection": {"path": ["payload", "text"]},
                "conversion": {"id": "example.text.normalize", "version": 2},
                "route_offset": {"x": 5.0, "y": -3.0},
            }
        ],
    }


def test_saved_graph_crud_round_trip(builtin_client: TestClient) -> None:
    create_response = builtin_client.post(
        "/v1/graphs",
        json=_graph_payload("  Parish index draft  "),
    )

    assert create_response.status_code == 201
    created = create_response.json()
    graph_id = UUID(created["id"])
    assert created["name"] == "Parish index draft"
    assert created["revision"] == 1
    assert len(created["nodes"]) == 2
    assert len(created["edges"]) == 1
    assert created["edges"][0]["projection"] == {"path": ["payload", "text"]}
    assert created["edges"][0]["conversion"] == {
        "id": "example.text.normalize",
        "version": 2,
    }

    get_response = builtin_client.get(f"/v1/graphs/{graph_id}")
    assert get_response.status_code == 200
    assert get_response.json() == created

    list_response = builtin_client.get("/v1/graphs")
    assert list_response.status_code == 200
    assert list_response.json() == {
        "graphs": [
            {
                "id": str(graph_id),
                "name": "Parish index draft",
                "revision": 1,
                "node_count": 2,
                "edge_count": 1,
                "updated_at": created["updated_at"],
            }
        ]
    }

    update_payload = _graph_payload("Updated draft")
    update_payload["expected_revision"] = 1
    update_response = builtin_client.put(
        f"/v1/graphs/{graph_id}",
        json=update_payload,
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["id"] == str(graph_id)
    assert updated["name"] == "Updated draft"
    assert updated["revision"] == 2
    assert updated["created_at"] == created["created_at"]

    delete_response = builtin_client.delete(
        f"/v1/graphs/{graph_id}",
        params={"expected_revision": updated["revision"]},
    )
    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert builtin_client.get(f"/v1/graphs/{graph_id}").status_code == 404


def test_create_rejects_structurally_invalid_graph(builtin_client: TestClient) -> None:
    payload = _graph_payload()
    payload["edges"] = [
        {
            "id": "dangling",
            "from_node": "source",
            "from_port": "result",
            "to_node": "missing",
            "to_port": "value",
        }
    ]

    response = builtin_client.post("/v1/graphs", json=payload)

    assert response.status_code == 422
    assert "missing target node missing" in str(response.json())
    assert builtin_client.get("/v1/graphs").json() == {"graphs": []}


def test_update_requires_positive_expected_revision(builtin_client: TestClient) -> None:
    created = builtin_client.post("/v1/graphs", json=_graph_payload()).json()
    payload = _graph_payload("Invalid revision")
    payload["expected_revision"] = 0

    response = builtin_client.put(
        f"/v1/graphs/{created['id']}",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_saved_graph_returns_not_found_for_crud_operations(
    builtin_client: TestClient,
) -> None:
    graph_id = UUID("00000000-0000-0000-0000-000000000404")
    update_payload = _graph_payload()
    update_payload["expected_revision"] = 1

    get_response = builtin_client.get(f"/v1/graphs/{graph_id}")
    update_response = builtin_client.put(
        f"/v1/graphs/{graph_id}",
        json=update_payload,
    )
    delete_response = builtin_client.delete(
        f"/v1/graphs/{graph_id}",
        params={"expected_revision": 1},
    )

    assert get_response.status_code == 404
    assert update_response.status_code == 404
    assert delete_response.status_code == 404
    assert str(graph_id) in get_response.json()["detail"]


def test_stale_update_returns_revision_conflict(builtin_client: TestClient) -> None:
    created = builtin_client.post("/v1/graphs", json=_graph_payload()).json()
    graph_id = created["id"]
    first_update = _graph_payload("First update")
    first_update["expected_revision"] = 1
    stale_update = _graph_payload("Stale update")
    stale_update["expected_revision"] = 1

    assert (
        builtin_client.put(f"/v1/graphs/{graph_id}", json=first_update).status_code
        == 200
    )
    conflict_response = builtin_client.put(
        f"/v1/graphs/{graph_id}",
        json=stale_update,
    )

    assert conflict_response.status_code == 409
    assert "expected 1" in conflict_response.json()["detail"]
    assert "current revision is 2" in conflict_response.json()["detail"]


def test_stale_delete_returns_revision_conflict_and_preserves_graph(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post("/v1/graphs", json=_graph_payload()).json()
    update_payload = _graph_payload("Newer graph")
    update_payload["expected_revision"] = created["revision"]
    updated = builtin_client.put(
        f"/v1/graphs/{created['id']}",
        json=update_payload,
    ).json()

    response = builtin_client.delete(
        f"/v1/graphs/{created['id']}",
        params={"expected_revision": created["revision"]},
    )

    assert response.status_code == 409
    assert "current revision is 2" in response.json()["detail"]
    assert builtin_client.get(f"/v1/graphs/{created['id']}").json() == updated


def test_name_length_is_checked_after_whitespace_normalization(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/graphs",
        json=_graph_payload("x" * 160 + " "),
    )

    assert response.status_code == 201
    assert response.json()["name"] == "x" * 160
