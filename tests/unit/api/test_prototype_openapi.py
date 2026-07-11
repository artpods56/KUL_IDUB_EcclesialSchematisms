from fastapi.testclient import TestClient

from notarius_api.main import app


def test_prototype_openapi_contains_only_prototype_routes() -> None:
    schema = app.openapi()

    assert set(schema["paths"]) == {
        "/v1/prototype/artifacts/{artifact_id}/content",
        "/v1/prototype/nodes",
        "/v1/prototype/run",
        "/v1/prototype/samples",
        "/v1/prototype/uploads",
    }
    node_schema = schema["components"]["schemas"]["PrototypeNodeSpecResponse"]
    assert "config_schema" in node_schema["properties"]


def test_prototype_app_health_is_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prototype_app_allows_local_web_origin() -> None:
    response = TestClient(app).options(
        "/v1/prototype/nodes",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:3000"
    )
