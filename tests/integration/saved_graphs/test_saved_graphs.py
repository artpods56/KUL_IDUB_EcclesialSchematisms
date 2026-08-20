import asyncio
from typing import cast
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from grafy_core.domain.collaboration import RenameGraphCommand
from grafy_core.domain.errors import NotFoundError
from grafy_core.domain.identity import ActorContext

from tests.support.identity import TEST_USER_ID, WORKSPACE_ID, workspace_api_path


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
                "layout": {
                    "width": 420.0,
                    "body_height": 180.0,
                },
            },
            {
                "id": "target",
                "operator_id": "plugin.target",
                "operator_version": 1,
                "config": {},
                "position": {"x": 300.0, "y": 20.0},
                "layout": {
                    "appendix_height": 320.0,
                },
                "input_plugs": [
                    {"id": "primary-value", "port": "value"},
                ],
                "artifact_type_bindings": [
                    {
                        "variable": "T",
                        "artifact_type": {
                            "id": "image.raster",
                            "schema_version": 1,
                        },
                    }
                ],
            },
        ],
        "edges": [
            {
                "id": "source-to-target",
                "from_node": "source",
                "from_port": "result",
                "to_node": "target",
                "to_port": "value",
                "to_plug": "primary-value",
                "collection_mode": "map",
                "projection": {"path": ["payload", "text"]},
                "conversion": {"id": "example.text.normalize", "version": 2},
                "route_offset": {"x": 5.0, "y": -3.0},
            }
        ],
    }


def test_saved_graph_crud_round_trip(builtin_client: TestClient) -> None:
    create_response = builtin_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs",
        json=_graph_payload("  Parish index draft  "),
    )

    assert create_response.status_code == 201
    created = create_response.json()
    graph_id = UUID(created["id"])
    assert created["name"] == "Parish index draft"
    assert created["revision"] == 1
    assert len(created["nodes"]) == 2
    assert len(created["edges"]) == 1
    assert created["nodes"][0]["input_plugs"] == []
    assert created["nodes"][0]["layout"] == {
        "width": 420.0,
        "body_height": 180.0,
        "appendix_height": None,
    }
    assert created["nodes"][1]["input_plugs"] == [
        {"id": "primary-value", "port": "value"}
    ]
    assert created["nodes"][1]["layout"] == {
        "width": None,
        "body_height": None,
        "appendix_height": 320.0,
    }
    assert created["nodes"][1]["artifact_type_bindings"] == [
        {
            "variable": "T",
            "artifact_type": {
                "id": "image.raster",
                "schema_version": 1,
            },
        }
    ]
    assert created["edges"][0]["to_plug"] == "primary-value"
    assert created["edges"][0]["enabled"] is True
    assert created["edges"][0]["projection"] == {"path": ["payload", "text"]}
    assert created["edges"][0]["conversion_path"] == [
        {
            "id": "example.text.normalize",
            "version": 2,
        }
    ]
    assert "conversion" not in created["edges"][0]

    get_response = builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}")
    assert get_response.status_code == 200
    assert get_response.json() == created

    list_response = builtin_client.get("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs")
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
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}",
        json=update_payload,
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["id"] == str(graph_id)
    assert updated["name"] == "Updated draft"
    assert updated["revision"] == 2
    assert updated["created_at"] == created["created_at"]

    delete_response = builtin_client.delete(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}",
        params={"expected_revision": updated["revision"]},
    )
    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}").status_code == 404


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

    response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=payload)

    assert response.status_code == 422
    assert "missing target node missing" in str(response.json())
    assert builtin_client.get("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs").json() == {"graphs": []}


def test_create_rejects_ambiguous_conversion_fields(
    builtin_client: TestClient,
) -> None:
    payload = _graph_payload()
    edge = cast(list[dict[str, object]], payload["edges"])[0]
    edge["conversion_path"] = [edge["conversion"]]

    response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=payload)

    assert response.status_code == 422
    assert "both conversion and conversion_path" in str(response.json())


def test_create_rejects_duplicate_artifact_type_binding_variables(
    builtin_client: TestClient,
) -> None:
    payload = _graph_payload()
    nodes = cast(list[dict[str, object]], payload["nodes"])
    target = nodes[1]
    bindings = cast(list[dict[str, object]], target["artifact_type_bindings"])
    bindings.append(
        {
            "variable": "T",
            "artifact_type": {
                "id": "scalar.text",
                "schema_version": 1,
            },
        }
    )

    response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=payload)

    assert response.status_code == 422
    assert "binding variables must be unique" in str(response.json())


def test_saved_graph_preserves_conversion_path_order(
    builtin_client: TestClient,
) -> None:
    payload = _graph_payload("Conversion path")
    edge = cast(list[dict[str, object]], payload["edges"])[0]
    edge.pop("conversion")
    edge["conversion_path"] = [
        {"id": "example.text.normalize", "version": 2},
        {"id": "example.text.finalize", "version": 7},
    ]

    create_response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=payload)

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["edges"][0]["conversion_path"] == edge["conversion_path"]
    loaded = builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}")
    assert loaded.status_code == 200
    assert loaded.json()["edges"][0]["conversion_path"] == edge["conversion_path"]


def test_saved_graph_preserves_disabled_edges(builtin_client: TestClient) -> None:
    payload = _graph_payload("Disabled edge")
    edge = cast(list[dict[str, object]], payload["edges"])[0]
    edge["enabled"] = False

    create_response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=payload)

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["edges"][0]["enabled"] is False
    loaded = builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}")
    assert loaded.status_code == 200
    assert loaded.json()["edges"][0]["enabled"] is False


def test_update_requires_positive_expected_revision(builtin_client: TestClient) -> None:
    created = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=_graph_payload()).json()
    payload = _graph_payload("Invalid revision")
    payload["expected_revision"] = 0

    response = builtin_client.put(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_saved_graph_returns_not_found_for_crud_operations(
    builtin_client: TestClient,
) -> None:
    graph_id = UUID("00000000-0000-0000-0000-000000000404")
    update_payload = _graph_payload()
    update_payload["expected_revision"] = 1

    get_response = builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}")
    update_response = builtin_client.put(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}",
        json=update_payload,
    )
    delete_response = builtin_client.delete(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}",
        params={"expected_revision": 1},
    )

    assert get_response.status_code == 404
    assert update_response.status_code == 404
    assert delete_response.status_code == 404
    assert str(graph_id) in get_response.json()["detail"]


def test_stale_update_returns_revision_conflict(builtin_client: TestClient) -> None:
    created = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=_graph_payload()).json()
    graph_id = created["id"]
    first_update = _graph_payload("First update")
    first_update["expected_revision"] = 1
    stale_update = _graph_payload("Stale update")
    stale_update["expected_revision"] = 1

    assert (
        builtin_client.put(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}", json=first_update).status_code
        == 200
    )
    conflict_response = builtin_client.put(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph_id}",
        json=stale_update,
    )

    assert conflict_response.status_code == 409
    assert "expected 1" in conflict_response.json()["detail"]
    assert "current revision is 2" in conflict_response.json()["detail"]


def test_stale_delete_returns_revision_conflict_and_preserves_graph(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs", json=_graph_payload()).json()
    update_payload = _graph_payload("Newer graph")
    update_payload["expected_revision"] = created["revision"]
    updated = builtin_client.put(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}",
        json=update_payload,
    ).json()

    response = builtin_client.delete(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}",
        params={"expected_revision": created["revision"]},
    )

    assert response.status_code == 409
    assert "current revision is 2" in response.json()["detail"]
    assert builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{created['id']}").json() == updated


def test_name_length_is_checked_after_whitespace_normalization(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs",
        json=_graph_payload("x" * 160 + " "),
    )

    assert response.status_code == 201
    assert response.json()["name"] == "x" * 160


def test_http_create_bootstraps_collaborative_head(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Bootstrapped"),
    )
    assert response.status_code == 201
    created = response.json()
    graph_id = UUID(created["id"])
    assert created["revision"] == 1

    head = asyncio.run(
        builtin_client.app.state.resources.collaboration.initialize_head_for_existing_graph(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
        )
    )

    assert head.collaboration_sequence == 1
    assert head.checkpoint_sequence == 1
    assert head.checkpoint_revision == 1
    assert head.name == "Bootstrapped"


def test_http_replace_resets_collaborative_epoch_when_checkpointed(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Before replace"),
    ).json()
    graph_id = UUID(created["id"])
    prior_head = asyncio.run(
        builtin_client.app.state.resources.collaboration.initialize_head_for_existing_graph(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
        )
    )
    prior_epoch = prior_head.room_epoch

    payload = _graph_payload("After replace")
    payload["expected_revision"] = created["revision"]
    response = builtin_client.put(
        workspace_api_path(f"/graphs/{graph_id}"),
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 2
    assert response.json()["name"] == "After replace"

    head = asyncio.run(
        builtin_client.app.state.resources.collaboration.initialize_head_for_existing_graph(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
        )
    )
    assert head.room_epoch != prior_epoch
    assert head.collaboration_sequence == 0
    assert head.checkpoint_sequence == 0
    assert head.checkpoint_revision == 2
    assert head.name == "After replace"


def test_http_replace_rejects_uncheckpointed_head(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Live draft"),
    ).json()
    graph_id = UUID(created["id"])
    collaboration = builtin_client.app.state.resources.collaboration
    head = asyncio.run(
        collaboration.initialize_head_for_existing_graph(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
        )
    )
    asyncio.run(
        collaboration.accept_command(
            actor=ActorContext(
                user_id=TEST_USER_ID,
                credential_reference="test-session",
            ),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            command_id=uuid4(),
            observed_sequence=head.collaboration_sequence,
            observed_room_epoch=head.room_epoch,
            command=RenameGraphCommand(name="Uncheckpointed", expected_name="Live draft"),
        )
    )

    payload = _graph_payload("Should fail")
    payload["expected_revision"] = created["revision"]
    response = builtin_client.put(
        workspace_api_path(f"/graphs/{graph_id}"),
        json=payload,
    )

    assert response.status_code == 409
    assert "uncheckpointed" in response.json()["detail"].lower()


def test_http_delete_removes_collaborative_head(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Delete me"),
    ).json()
    graph_id = UUID(created["id"])

    response = builtin_client.delete(
        workspace_api_path(f"/graphs/{graph_id}"),
        params={"expected_revision": created["revision"]},
    )
    assert response.status_code == 204
    assert (
        builtin_client.get(workspace_api_path(f"/graphs/{graph_id}")).status_code
        == 404
    )

    with pytest.raises(NotFoundError):
        asyncio.run(
            builtin_client.app.state.resources.collaboration.initialize_head_for_existing_graph(
                workspace_id=WORKSPACE_ID,
                graph_id=graph_id,
            )
        )


def test_http_live_head_command_checkpoint_and_aware_delete(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Command graph"),
    ).json()
    graph_id = created["id"]

    head_response = builtin_client.get(workspace_api_path(f"/graphs/{graph_id}/head"))
    assert head_response.status_code == 200
    head = head_response.json()
    assert head["name"] == "Command graph"
    assert head["collaboration_sequence"] == 1
    assert head["checkpoint_sequence"] == 1

    command_response = builtin_client.post(
        workspace_api_path(f"/graphs/{graph_id}/commands"),
        json={
            "command_id": str(uuid4()),
            "room_epoch": head["room_epoch"],
            "observed_sequence": head["collaboration_sequence"],
            "command": {
                "kind": "rename_graph",
                "name": "Renamed live",
                "expected_name": "Command graph",
            },
        },
    )
    assert command_response.status_code == 200
    command_payload = command_response.json()
    assert command_payload["head"]["name"] == "Renamed live"
    assert command_payload["head"]["collaboration_sequence"] == 2
    assert command_payload["receipt"]["outcome"] == "accepted"
    assert command_payload["receipt"]["deduplicated"] is False

    checkpoint_response = builtin_client.post(
        workspace_api_path(f"/graphs/{graph_id}/checkpoint"),
        json={
            "expected_room_epoch": command_payload["head"]["room_epoch"],
            "expected_sequence": command_payload["head"]["collaboration_sequence"],
        },
    )
    assert checkpoint_response.status_code == 200
    assert checkpoint_response.json()["saved_revision"] == 2
    assert checkpoint_response.json()["head"]["checkpoint_sequence"] == 2

    saved = builtin_client.get(workspace_api_path(f"/graphs/{graph_id}")).json()
    assert saved["name"] == "Renamed live"
    assert saved["revision"] == 2

    # Leave an uncheckpointed command, then discard with exact-head delete.
    pending = builtin_client.post(
        workspace_api_path(f"/graphs/{graph_id}/commands"),
        json={
            "command_id": str(uuid4()),
            "room_epoch": checkpoint_response.json()["head"]["room_epoch"],
            "observed_sequence": checkpoint_response.json()["head"][
                "collaboration_sequence"
            ],
            "command": {
                "kind": "rename_graph",
                "name": "Discard me",
                "expected_name": "Renamed live",
            },
        },
    ).json()
    delete_response = builtin_client.delete(
        workspace_api_path(f"/graphs/{graph_id}"),
        params={
            "expected_revision": saved["revision"],
            "expected_room_epoch": pending["head"]["room_epoch"],
            "expected_sequence": pending["head"]["collaboration_sequence"],
        },
    )
    assert delete_response.status_code == 204


def test_http_copy_exact_head_into_same_workspace(
    builtin_client: TestClient,
) -> None:
    created = builtin_client.post(
        workspace_api_path("/graphs"),
        json=_graph_payload("Copy source"),
    ).json()
    head = builtin_client.get(
        workspace_api_path(f"/graphs/{created['id']}/head")
    ).json()

    copy_response = builtin_client.post(
        workspace_api_path("/graphs/copies"),
        json={
            "source_workspace_id": str(WORKSPACE_ID),
            "source_graph_id": created["id"],
            "expected_room_epoch": head["room_epoch"],
            "expected_sequence": head["collaboration_sequence"],
            "command_id": str(uuid4()),
            "name": "Copied graph",
        },
    )
    assert copy_response.status_code == 201
    copied = copy_response.json()
    assert copied["id"] != created["id"]
    assert copied["name"] == "Copied graph"
    assert copied["revision"] == 1

    copied_head = builtin_client.get(
        workspace_api_path(f"/graphs/{copied['id']}/head")
    ).json()
    assert copied_head["collaboration_sequence"] == 1
    assert copied_head["checkpoint_sequence"] == 1
    assert copied_head["checkpoint_revision"] == 1
