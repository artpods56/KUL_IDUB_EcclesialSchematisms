from fastapi.testclient import TestClient

from notarius_api.main import app


def test_openapi_contains_exact_public_routes() -> None:
    schema = app.openapi()

    assert set(schema["paths"]) == {
        "/v1/artifacts/{artifact_id}/content",
        "/v1/graphs",
        "/v1/graphs/{graph_id}",
        "/v1/graphs/{graph_id}/materializations",
        "/v1/nodes",
        "/v1/runs",
        "/v1/samples",
        "/v1/uploads",
    }
    assert set(schema["paths"]["/v1/graphs"]) == {"get", "post"}
    assert set(schema["paths"]["/v1/graphs/{graph_id}"]) == {
        "delete",
        "get",
        "put",
    }
    assert set(schema["paths"]["/v1/graphs/{graph_id}/materializations"]) == {"get"}
    node_schema = schema["components"]["schemas"]["NodeSpecResponse"]
    assert "config_schema" in node_schema["properties"]
    assert "supported_invocation_modes" not in node_schema["properties"]
    assert "map_inputs" not in node_schema["properties"]

    port_schema = schema["components"]["schemas"]["PortResponse"]
    assert "title" in port_schema["properties"]
    assert "description" in port_schema["properties"]
    assert port_schema["properties"]["accepted_shapes"]["items"] == {
        "$ref": "#/components/schemas/PortShape"
    }
    assert port_schema["properties"]["instance_plugs"] == {
        "default": False,
        "title": "Instance Plugs",
        "type": "boolean",
    }
    assert port_schema["properties"]["artifact_type"]["anyOf"] == [
        {"$ref": "#/components/schemas/ArtifactTypeKeyResponse"},
        {"type": "null"},
    ]
    assert port_schema["properties"]["artifact_type_variable"]["anyOf"] == [
        {"type": "string", "minLength": 1, "maxLength": 255},
        {"type": "null"},
    ]

    run_node_schema = schema["components"]["schemas"]["RunNodeRequest"]
    assert set(run_node_schema["required"]) == {
        "id",
        "operator_id",
        "operator_version",
    }
    assert run_node_schema["properties"]["input_plugs"]["items"] == {
        "$ref": "#/components/schemas/RunInputPlugRequest"
    }
    assert run_node_schema["properties"]["artifact_type_bindings"]["items"] == {
        "$ref": "#/components/schemas/ArtifactTypeBindingModel"
    }
    run_edge_schema = schema["components"]["schemas"]["RunEdgeRequest"]
    assert run_edge_schema["properties"]["collection_mode"] == {
        "default": "direct",
        "enum": ["direct", "map"],
        "title": "Collection Mode",
        "type": "string",
    }
    assert run_edge_schema["properties"]["to_plug"]["anyOf"] == [
        {"type": "string", "minLength": 1, "maxLength": 255},
        {"type": "null"},
    ]

    for schema_name in ("SavedGraphNodeModel-Input", "SavedGraphNodeModel-Output"):
        saved_node_schema = schema["components"]["schemas"][schema_name]
        assert saved_node_schema["properties"]["input_plugs"]["items"] == {
            "$ref": "#/components/schemas/SavedGraphInputPlugModel"
        }
        assert saved_node_schema["properties"]["artifact_type_bindings"][
            "items"
        ] == {"$ref": "#/components/schemas/ArtifactTypeBindingModel"}
    saved_edge_schema = schema["components"]["schemas"]["SavedGraphEdgeModel"]
    assert "to_plug" in saved_edge_schema["properties"]


def test_app_health_is_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_allows_local_web_origin() -> None:
    response = TestClient(app).options(
        "/v1/nodes",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ("http://localhost:3000")
