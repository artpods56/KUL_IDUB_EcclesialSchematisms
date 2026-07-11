from notarius_api.prototype_openapi import prototype_openapi_app


def test_prototype_openapi_contains_only_prototype_routes() -> None:
    schema = prototype_openapi_app.openapi()

    assert set(schema["paths"]) == {
        "/v1/prototype/artifacts/{artifact_id}/content",
        "/v1/prototype/nodes",
        "/v1/prototype/run",
        "/v1/prototype/samples",
        "/v1/prototype/uploads",
    }
    node_schema = schema["components"]["schemas"]["PrototypeNodeSpecResponse"]
    assert "config_schema" in node_schema["properties"]
