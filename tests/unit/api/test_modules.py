import asyncio
import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, cast, final, override

import pytest
from fastapi.testclient import TestClient
from pydantic import Field, SecretStr, StrictStr

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.artifacts import (
    InMemoryUnitOfWork,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.text import TEXT_VALUE
from notarius_core.plugins import NodeSecretInput, Plugin
from notarius_core.ports.node_secrets import NodeSecretResolverPort
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.node_secrets import NodeSecretService
from notarius_api.services.composition import (
    build_workbench_components,
)
from notarius_api.settings import Settings
from notarius_api.v1.routes.saved_graphs import saved_graph_service
from notarius_api.v1.routes.node_secrets import node_secret_service

from tests.unit.api.conftest import install_workbench_dependency_overrides


SECRET_MODULE_PLUGIN = Plugin(
    slug="test.module-secret",
    title="Module secret test",
)


class SecretGateConfig(NodeConfig):
    base_url: StrictStr = Field(min_length=1)


class SecretGateInput(NodeInput):
    text: Annotated[StrictStr, InPort(TEXT_VALUE)]


class SecretGateOutput(NodeOutput):
    text: Annotated[StrictStr, OutPort(TEXT_VALUE)]


@SECRET_MODULE_PLUGIN.node(
    operator_id="test.module_secret_gate",
    version=1,
    title="Module secret gate",
    factory=lambda context: SecretGateNode(context.node_secrets),
    secret_inputs=(
        NodeSecretInput(
            name="api_key",
            title="API key",
            config_dependencies=("base_url",),
        ),
    ),
)
@final
class SecretGateNode(Node[SecretGateConfig, SecretGateInput, SecretGateOutput]):
    def __init__(self, node_secrets: NodeSecretResolverPort) -> None:
        self._node_secrets = node_secrets

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        config: SecretGateConfig,
        inputs: SecretGateInput,
        /,
    ) -> SecretGateOutput:
        secret = await self._node_secrets.resolve_secret(
            graph_id=context.secret_graph_id,
            graph_revision=context.secret_graph_revision,
            node_id=context.node_id,
            name="api_key",
            dependencies={"base_url": config.base_url},
        )
        if secret.get_secret_value() != "module-only-key":
            raise RuntimeError("Module secret did not resolve to the expected value")
        return SecretGateOutput(text=f"{inputs.text} authorized")


class OptionalSuffixInput(NodeInput):
    text: Annotated[StrictStr, InPort(TEXT_VALUE)]
    suffix: Annotated[str | None, InPort(TEXT_VALUE)] = None


class OptionalSuffixOutput(NodeOutput):
    text: Annotated[StrictStr, OutPort(TEXT_VALUE)]


@SECRET_MODULE_PLUGIN.node(
    operator_id="test.module_optional_suffix",
    version=1,
    title="Optional module suffix",
)
@final
class OptionalSuffixNode(Node[NodeConfig, OptionalSuffixInput, OptionalSuffixOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NodeConfig,
        inputs: OptionalSuffixInput,
        /,
    ) -> OptionalSuffixOutput:
        suffix = inputs.suffix if inputs.suffix is not None else ""
        return OptionalSuffixOutput(text=f"{inputs.text}{suffix}")


async def _create_schema(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
    finally:
        await database.dispose()


@pytest.fixture
def module_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'modules.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    database = create_database(database_url)
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(SECRET_MODULE_PLUGIN,),
    )
    saved_graphs = SavedGraphService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        registry,
    )
    node_secrets = NodeSecretService(
        unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
        plugin_registry=registry,
        encryption_key=SecretStr(base64.b64encode(b"m" * 32).decode("ascii")),
    )
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=tmp_path / "workbench",
        unit_of_work=InMemoryUnitOfWork(),
        saved_graphs=saved_graphs,
        node_secrets=node_secrets,
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )
    install_workbench_dependency_overrides(application, components)
    application.dependency_overrides[saved_graph_service] = lambda: saved_graphs
    application.dependency_overrides[node_secret_service] = lambda: node_secrets
    with TestClient(application) as client:
        yield client
    asyncio.run(database.dispose())


def _artifact_binding() -> list[dict[str, object]]:
    return [
        {
            "variable": "T",
            "artifact_type": {
                "id": "scalar.text",
                "schema_version": 1,
            },
        }
    ]


def _text_module_payload(
    *,
    name: str = "Capitalize A",
    replacement: str = "A",
    input_required: bool = True,
) -> dict[str, object]:
    return {
        "name": name,
        "nodes": [
            {
                "id": "module-input",
                "operator_id": "module.input",
                "operator_version": 1,
                "config": {
                    "public_name": "text",
                    "description": "Text to transform",
                    "required": input_required,
                },
                "position": {"x": 0, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "replace",
                "operator_id": "text.replace",
                "operator_version": 1,
                "config": {"search": "a", "replacement": replacement},
                "position": {"x": 240, "y": 0},
            },
            {
                "id": "module-output",
                "operator_id": "module.output",
                "operator_version": 1,
                "config": {
                    "public_name": "result",
                    "description": "Transformed text",
                },
                "position": {"x": 480, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
        ],
        "edges": [
            {
                "id": "input-to-replace",
                "from_node": "module-input",
                "from_port": "value",
                "to_node": "replace",
                "to_port": "text",
            },
            {
                "id": "replace-to-output",
                "from_node": "replace",
                "from_port": "text",
                "to_node": "module-output",
                "to_port": "value",
            },
        ],
    }


def _secret_module_payload() -> dict[str, object]:
    return {
        "name": "Secret-bearing module",
        "nodes": [
            {
                "id": "module-input",
                "operator_id": "module.input",
                "operator_version": 1,
                "config": {"public_name": "text"},
                "position": {"x": 0, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "secret-gate",
                "operator_id": "test.module_secret_gate",
                "operator_version": 1,
                "config": {"base_url": "https://provider.example/v1"},
                "position": {"x": 240, "y": 0},
            },
            {
                "id": "module-output",
                "operator_id": "module.output",
                "operator_version": 1,
                "config": {"public_name": "result"},
                "position": {"x": 480, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
        ],
        "edges": [
            {
                "id": "input-to-gate",
                "from_node": "module-input",
                "from_port": "value",
                "to_node": "secret-gate",
                "to_port": "text",
            },
            {
                "id": "gate-to-output",
                "from_node": "secret-gate",
                "from_port": "text",
                "to_node": "module-output",
                "to_port": "value",
            },
        ],
    }


def _optional_input_module_payload() -> dict[str, object]:
    return {
        "name": "Optional suffix module",
        "nodes": [
            {
                "id": "text-input",
                "operator_id": "module.input",
                "operator_version": 1,
                "config": {"public_name": "text"},
                "position": {"x": 0, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "suffix-input",
                "operator_id": "module.input",
                "operator_version": 1,
                "config": {"public_name": "suffix", "required": False},
                "position": {"x": 0, "y": 120},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "append",
                "operator_id": "test.module_optional_suffix",
                "operator_version": 1,
                "config": {},
                "position": {"x": 240, "y": 0},
            },
            {
                "id": "collect",
                "operator_id": "sequence.collect",
                "operator_version": 1,
                "config": {},
                "position": {"x": 240, "y": 160},
                "input_plugs": [
                    {"id": "active-copy", "port": "items"},
                    {"id": "disabled-copy", "port": "items"},
                ],
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "pick",
                "operator_id": "sequence.item_at",
                "operator_version": 1,
                "config": {"index": 0},
                "position": {"x": 480, "y": 160},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "module-output",
                "operator_id": "module.output",
                "operator_version": 1,
                "config": {"public_name": "result"},
                "position": {"x": 480, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
        ],
        "edges": [
            {
                "id": "text-to-append",
                "from_node": "text-input",
                "from_port": "value",
                "to_node": "append",
                "to_port": "text",
            },
            {
                "id": "suffix-to-append",
                "from_node": "suffix-input",
                "from_port": "value",
                "to_node": "append",
                "to_port": "suffix",
            },
            {
                "id": "text-to-collect",
                "from_node": "text-input",
                "from_port": "value",
                "to_node": "collect",
                "to_port": "items",
                "to_plug": "active-copy",
            },
            {
                "id": "disabled-text-to-collect",
                "enabled": False,
                "from_node": "text-input",
                "from_port": "value",
                "to_node": "collect",
                "to_port": "items",
                "to_plug": "disabled-copy",
            },
            {
                "id": "collect-to-pick",
                "from_node": "collect",
                "from_port": "items",
                "to_node": "pick",
                "to_port": "items",
            },
            {
                "id": "append-to-output",
                "from_node": "append",
                "from_port": "text",
                "to_node": "module-output",
                "to_port": "value",
            },
        ],
    }


def _delegating_module_payload(
    *,
    name: str,
    target_graph_id: str,
    target_revision: int,
) -> dict[str, object]:
    return {
        "name": name,
        "nodes": [
            {
                "id": "module-input",
                "operator_id": "module.input",
                "operator_version": 1,
                "config": {"public_name": "text"},
                "position": {"x": 0, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
            {
                "id": "delegate",
                "operator_id": f"graph.module.{target_graph_id}",
                "operator_version": target_revision,
                "config": {},
                "position": {"x": 240, "y": 0},
            },
            {
                "id": "module-output",
                "operator_id": "module.output",
                "operator_version": 1,
                "config": {"public_name": "result"},
                "position": {"x": 480, "y": 0},
                "artifact_type_bindings": _artifact_binding(),
            },
        ],
        "edges": [
            {
                "id": "input-to-delegate",
                "from_node": "module-input",
                "from_port": "value",
                "to_node": "delegate",
                "to_port": "text",
            },
            {
                "id": "delegate-to-output",
                "from_node": "delegate",
                "from_port": "result",
                "to_node": "module-output",
                "to_port": "value",
            },
        ],
    }


def _module_node_run(
    result: dict[str, object],
    node_id: str = "module",
) -> dict[str, object]:
    node_runs = cast(list[dict[str, object]], result["node_runs"])
    return next(run for run in node_runs if run["node_id"] == node_id)


def test_saved_graph_module_is_discoverable_and_executes_once(
    module_client: TestClient,
) -> None:
    created = module_client.post("/v1/graphs", json=_text_module_payload()).json()
    graph_id = created["id"]

    registry_response = module_client.get("/v1/nodes")
    assert registry_response.status_code == 200
    registry = registry_response.json()
    assert {
        "slug": "graph.module",
        "title": "Modules",
        "origin": "module",
    } in registry["plugins"]
    module_spec = next(
        node for node in registry["nodes"] if node["module_graph_id"] == graph_id
    )
    assert module_spec["operator_id"] == f"graph.module.{graph_id}"
    assert module_spec["module_graph_revision"] == 1
    assert module_spec["catalog_visible"] is True
    assert [port["name"] for port in module_spec["inputs"]] == ["text"]
    assert [port["name"] for port in module_spec["outputs"]] == ["result"]
    assert all(
        module["graph_id"] != graph_id for module in registry["unavailable_modules"]
    )

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "a cat"},
                },
                {
                    "id": "module",
                    "operator_id": module_spec["operator_id"],
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "succeeded"
    module_run = _module_node_run(result)
    output = cast(list[dict[str, object]], module_run["outputs"])[0]
    artifacts = cast(list[dict[str, object]], output["artifacts"])
    assert output["kind"] == "single"
    assert artifacts[0]["text"] == '"A cAt"'
    metadata = cast(dict[str, object], artifacts[0]["metadata"])
    assert metadata["producer_node_id"] == "replace"


def test_module_uses_existing_map_semantics_and_keeps_revision_pinned(
    module_client: TestClient,
) -> None:
    created = module_client.post("/v1/graphs", json=_text_module_payload()).json()
    graph_id = created["id"]
    operator_id = f"graph.module.{graph_id}"

    mapped_response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "a|ba|ca"},
                },
                {
                    "id": "split",
                    "operator_id": "text.split",
                    "operator_version": 1,
                    "config": {"separator": "|"},
                },
                {
                    "id": "module",
                    "operator_id": operator_id,
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "split",
                    "to_port": "text",
                },
                {
                    "from_node": "split",
                    "from_port": "parts",
                    "to_node": "module",
                    "to_port": "text",
                    "collection_mode": "map",
                },
            ],
        },
    )
    assert mapped_response.status_code == 200
    mapped_result = mapped_response.json()
    assert mapped_result["status"] == "succeeded"
    mapped_output = cast(
        list[dict[str, object]],
        _module_node_run(mapped_result)["outputs"],
    )[0]
    assert mapped_output["kind"] == "sequence"
    assert [
        artifact["text"]
        for artifact in cast(list[dict[str, object]], mapped_output["artifacts"])
    ] == ['"A"', '"bA"', '"cA"']

    update_payload = _text_module_payload(replacement="X")
    update_payload["expected_revision"] = 1
    updated_response = module_client.put(
        f"/v1/graphs/{graph_id}",
        json=update_payload,
    )
    assert updated_response.status_code == 200
    assert updated_response.json()["revision"] == 2

    registry = module_client.get("/v1/nodes").json()
    module_specs = [
        node for node in registry["nodes"] if node["module_graph_id"] == graph_id
    ]
    assert [
        (spec["module_graph_revision"], spec["catalog_visible"])
        for spec in module_specs
    ] == [(2, True), (1, False)]

    pinned_response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "a"},
                },
                {
                    "id": "module",
                    "operator_id": operator_id,
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                }
            ],
        },
    )
    pinned_output = cast(
        list[dict[str, object]],
        _module_node_run(pinned_response.json())["outputs"],
    )[0]
    pinned_artifacts = cast(list[dict[str, object]], pinned_output["artifacts"])
    assert pinned_artifacts[0]["text"] == '"A"'


def test_nested_module_omits_absent_optional_input_and_disabled_edges(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json=_optional_input_module_payload(),
    ).json()
    graph_id = created["id"]
    registry = module_client.get("/v1/nodes").json()
    module_spec = next(
        node for node in registry["nodes"] if node["module_graph_id"] == graph_id
    )
    assert [(port["name"], port["required"]) for port in module_spec["inputs"]] == [
        ("text", True),
        ("suffix", False),
    ]

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "hello"},
                },
                {
                    "id": "module",
                    "operator_id": module_spec["operator_id"],
                    "operator_version": module_spec["operator_version"],
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "succeeded"
    output = cast(list[dict[str, object]], _module_node_run(result)["outputs"])[0]
    artifacts = cast(list[dict[str, object]], output["artifacts"])
    assert artifacts[0]["text"] == '"hello"'

    supplied_response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "text-source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "hello"},
                },
                {
                    "id": "suffix-source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "!"},
                },
                {
                    "id": "module",
                    "operator_id": module_spec["operator_id"],
                    "operator_version": module_spec["operator_version"],
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "text-source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                },
                {
                    "from_node": "suffix-source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "suffix",
                },
            ],
        },
    )

    assert supplied_response.status_code == 200
    supplied_result = supplied_response.json()
    assert supplied_result["status"] == "succeeded"
    supplied_output = cast(
        list[dict[str, object]],
        _module_node_run(supplied_result)["outputs"],
    )[0]
    supplied_artifacts = cast(list[dict[str, object]], supplied_output["artifacts"])
    assert supplied_artifacts[0]["text"] == '"hello!"'


def test_module_catalog_rejects_optional_input_targeting_required_input(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json=_text_module_payload(
            name="Invalid optional input",
            input_required=False,
        ),
    ).json()
    graph_id = created["id"]
    reason = (
        f"Graph module {graph_id} revision 1 optional public input 'text' edge "
        "'input-to-replace' targets required input 'replace'.'text' "
        "(text.replace@1)"
    )

    registry = module_client.get("/v1/nodes").json()
    assert all(
        node["module_graph_id"] != graph_id
        for node in registry["nodes"]
        if node["plugin_slug"] == "graph.module"
    )
    assert registry["unavailable_modules"] == [
        {
            "graph_id": graph_id,
            "revision": 1,
            "name": "Invalid optional input",
            "reason": reason,
        }
    ]

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "module",
                    "operator_id": f"graph.module.{graph_id}",
                    "operator_version": 1,
                    "config": {},
                }
            ]
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"Saved graph {graph_id} revision 1 is not a valid module: {reason}"
    )


def test_module_catalog_reports_invalid_boundary_wiring(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json={
            "name": "Input without output",
            "nodes": [
                {
                    "id": "module-input",
                    "operator_id": "module.input",
                    "operator_version": 1,
                    "config": {"public_name": "text"},
                    "position": {"x": 0, "y": 0},
                    "artifact_type_bindings": _artifact_binding(),
                }
            ],
            "edges": [],
        },
    ).json()
    graph_id = created["id"]

    registry = module_client.get("/v1/nodes").json()
    unavailable = [
        module
        for module in registry["unavailable_modules"]
        if module["graph_id"] == graph_id
    ]
    assert len(unavailable) == 1
    assert unavailable[0]["revision"] == 1
    assert unavailable[0]["name"] == "Input without output"
    assert "Module Input boundary must connect its 'value' output" in unavailable[0][
        "reason"
    ]
    assert all(
        node["module_graph_id"] != graph_id
        for node in registry["nodes"]
        if node["plugin_slug"] == "graph.module"
    )


def test_module_catalog_ignores_graphs_without_module_boundaries(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json={
            "name": "Ordinary workflow",
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "hello"},
                    "position": {"x": 0, "y": 0},
                }
            ],
            "edges": [],
        },
    ).json()
    graph_id = created["id"]

    registry = module_client.get("/v1/nodes").json()
    assert all(
        node["module_graph_id"] != graph_id
        for node in registry["nodes"]
        if node["plugin_slug"] == "graph.module"
    )
    assert all(
        module["graph_id"] != graph_id for module in registry["unavailable_modules"]
    )


def test_graph_module_required_input_is_rejected_by_compiler(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json=_optional_input_module_payload(),
    ).json()

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "module",
                    "operator_id": f"graph.module.{created['id']}",
                    "operator_version": created["revision"],
                    "config": {},
                }
            ]
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"Node 'module' (graph.module.{created['id']}@1) required input "
        "'text' has no incoming edge"
    )


def test_nested_module_resolves_secret_from_its_own_pinned_graph(
    module_client: TestClient,
) -> None:
    created = module_client.post(
        "/v1/graphs",
        json=_secret_module_payload(),
    ).json()
    graph_id = created["id"]
    configured = module_client.put(
        f"/v1/graphs/{graph_id}/nodes/secret-gate/secrets/api_key",
        json={"value": "module-only-key", "expected_graph_revision": 1},
    )
    assert configured.status_code == 200
    assert configured.json() == {
        "node_id": "secret-gate",
        "name": "api_key",
        "configured": True,
    }

    update_payload = _secret_module_payload()
    update_payload["name"] = "Renamed secret-bearing module"
    update_payload["expected_revision"] = 1
    assert (
        module_client.put(
            f"/v1/graphs/{graph_id}",
            json=update_payload,
        ).status_code
        == 200
    )

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "request"},
                },
                {
                    "id": "module",
                    "operator_id": f"graph.module.{graph_id}",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "succeeded"
    output = cast(
        list[dict[str, object]],
        _module_node_run(result)["outputs"],
    )[0]
    artifacts = cast(list[dict[str, object]], output["artifacts"])
    assert artifacts[0]["text"] == '"request authorized"'
    assert "module-only-key" not in response.text


def test_exact_module_revision_cycle_reports_the_nested_path(
    module_client: TestClient,
) -> None:
    first = module_client.post(
        "/v1/graphs",
        json=_text_module_payload(name="First"),
    ).json()
    second = module_client.post(
        "/v1/graphs",
        json=_text_module_payload(name="Second"),
    ).json()

    first_update = _delegating_module_payload(
        name="First",
        target_graph_id=second["id"],
        target_revision=2,
    )
    first_update["expected_revision"] = 1
    assert (
        module_client.put(
            f"/v1/graphs/{first['id']}",
            json=first_update,
        ).status_code
        == 200
    )

    second_update = _delegating_module_payload(
        name="Second",
        target_graph_id=first["id"],
        target_revision=2,
    )
    second_update["expected_revision"] = 1
    assert (
        module_client.put(
            f"/v1/graphs/{second['id']}",
            json=second_update,
        ).status_code
        == 200
    )

    response = module_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "a"},
                },
                {
                    "id": "module",
                    "operator_id": f"graph.module.{first['id']}",
                    "operator_version": 2,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "module",
                    "to_port": "text",
                }
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    error = cast(str, _module_node_run(result)["error"])
    assert "Graph module cycle detected" in error
    assert f"graph.module.{first['id']}@2" in error
    assert f"graph.module.{second['id']}@2" in error
