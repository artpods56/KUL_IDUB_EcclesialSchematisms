import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, cast, final, override
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, StrictStr

from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemySavedGraphUnitOfWork

from notarius_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    InMemoryUnitOfWork,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_core.conversions import ArtifactConversion, ArtifactConversionKey
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.arithmetic import INTEGER_VALUE
from notarius_core.operators.tables import TableArtifactWriter
from notarius_core.operators.text import TEXT_VALUE
from notarius_core.plugins import Plugin
from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.composition import (
    WorkbenchComponents,
    build_workbench_components,
)
from notarius_api.settings import Settings
from notarius_api.v1.routes.artifacts.dependencies import artifact_service
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.catalog.dependencies import (
    graph_module_catalog,
    graph_module_executor,
    plugin_registry,
)
from notarius_api.v1.routes.executions.dependencies import (
    execution_admission_limiter,
    execution_history_service,
    materialization_service,
    run_execution_manager,
    run_graph_service,
    run_result_presenter,
)
from notarius_api.v1.routes.uploads.dependencies import image_upload_service
from notarius_storage import LocalFileObjectStore


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")
TEST_USER_ID = UUID(int=1)
TEST_COMMAND_HMAC_KEY = "test-api-command-hmac-key"


@pytest.fixture(autouse=True)
def _configure_command_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTARIUS_COMMAND_HMAC_KEY", TEST_COMMAND_HMAC_KEY)
    monkeypatch.setenv("NOTARIUS_COMMAND_HMAC_KEY_VERSION", "1")


def workspace_api_path(suffix: str) -> str:
    normalized = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"/v1/workspaces/{WORKSPACE_ID}{normalized}"


class CompoundResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    addition: StrictInt
    subtraction: StrictInt


def install_browser_actor_override(application: FastAPI) -> None:
    def test_browser_actor() -> ActorContext:
        return ActorContext(
            user_id=TEST_USER_ID,
            credential_reference="test-session",
        )

    application.dependency_overrides[browser_actor] = test_browser_actor


TEST_COMPOUND_RESULT = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.compound_result", 1),
    title="Test compound result",
    payload_schema=cast(JsonObject, CompoundResultPayload.model_json_schema()),
)


def _text_to_compound_result(value: str) -> CompoundResultPayload:
    integer = int(value)
    return CompoundResultPayload(addition=integer + 1, subtraction=integer - 1)


def _failing_text_to_compound_result(value: str) -> CompoundResultPayload:
    raise ValueError(f"Cannot convert {value!r}")


def _invalid_text_to_compound_result(value: str) -> CompoundResultPayload:
    integer = int(value)
    return cast(
        CompoundResultPayload,
        {
            "addition": integer + 1,
            "subtraction": integer - 1,
        },
    )


def _text_to_integer(value: str) -> int:
    return int(value)


TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_compound_result", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="As compound result",
    convert=_text_to_compound_result,
)
FAILING_TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_compound_result_failure", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="Fail as compound result",
    convert=_failing_text_to_compound_result,
)
INVALID_TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_invalid_compound_result", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="As invalid compound result",
    convert=_invalid_text_to_compound_result,
)
TEXT_TO_INTEGER = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_integer", 1),
    source=TEXT_VALUE.key,
    target=INTEGER_VALUE.key,
    source_type=str,
    target_type=int,
    title="Back to integer",
    convert=_text_to_integer,
)
CONVERSION_PATH_PLUGIN = Plugin(
    slug="test.conversion-path",
    title="Conversion path test plugin",
)
CONVERSION_PATH_PLUGIN.register_artifact_type(TEST_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(FAILING_TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(INVALID_TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_INTEGER)
CONVERSION_PATH_PLUGIN.register_resolver(
    lambda context: InlineModelResolver(
        source=TEST_COMPOUND_RESULT.key,
        target=CompoundResultPayload,
        uow=context.uow,
    )
)
CONVERSION_PATH_PLUGIN.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=TEST_COMPOUND_RESULT.key,
        model=CompoundResultPayload,
        uow=context.uow,
    )
)


class CompoundProducerInput(NodeInput):
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class CompoundProducerOutput(NodeOutput):
    result: Annotated[CompoundResultPayload, OutPort(TEST_COMPOUND_RESULT)]


@CONVERSION_PATH_PLUGIN.node(
    operator_id="test.compound_producer",
    version=1,
    title="Compound producer",
)
@final
class CompoundProducerNode(
    Node[NoConfig, CompoundProducerInput, CompoundProducerOutput]
):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CompoundProducerInput,
        /,
    ) -> CompoundProducerOutput:
        return CompoundProducerOutput(
            result=CompoundResultPayload(
                addition=inputs.left + inputs.right,
                subtraction=inputs.left - inputs.right,
            )
        )


class CompoundResultConsumerInput(NodeInput):
    result: Annotated[CompoundResultPayload, InPort(TEST_COMPOUND_RESULT)]


class CompoundResultConsumerOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@CONVERSION_PATH_PLUGIN.node(
    operator_id="test.compound_result_consumer",
    version=1,
    title="Compound result consumer",
)
@final
class CompoundResultConsumerNode(
    Node[NoConfig, CompoundResultConsumerInput, CompoundResultConsumerOutput]
):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CompoundResultConsumerInput,
        /,
    ) -> CompoundResultConsumerOutput:
        return CompoundResultConsumerOutput(
            value=inputs.result.addition * inputs.result.subtraction
        )


class ApiCustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: StrictStr = Field(title="Display name")
    retry_count: StrictInt = Field(title="Retry count")


class ApiResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: ApiCustomerPayload = Field(title="Customer")


API_RESPONSE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.api_response", 1),
    title="API response",
    payload_schema=cast(JsonObject, ApiResponsePayload.model_json_schema()),
)
STRUCTURAL_PROJECTION_PLUGIN = Plugin(
    slug="test.structural-projection",
    title="Structural projection test plugin",
)
STRUCTURAL_PROJECTION_PLUGIN.register_artifact_type(API_RESPONSE)
STRUCTURAL_PROJECTION_PLUGIN.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=API_RESPONSE.key,
        model=ApiResponsePayload,
        uow=context.uow,
    )
)


class ApiResponseNodeInput(NodeInput):
    pass


class ApiResponseNodeOutput(NodeOutput):
    response: Annotated[ApiResponsePayload, OutPort(API_RESPONSE)]


@STRUCTURAL_PROJECTION_PLUGIN.node(
    operator_id="test.api_response",
    version=1,
    title="API response",
)
@final
class ApiResponseNode(Node[NoConfig, ApiResponseNodeInput, ApiResponseNodeOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: ApiResponseNodeInput,
        /,
    ) -> ApiResponseNodeOutput:
        return ApiResponseNodeOutput(
            response=ApiResponsePayload(
                customer=ApiCustomerPayload(
                    display_name="abc",
                    retry_count=42,
                )
            )
        )


async def _create_schema(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemySavedGraphUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.identity.add_user(
                User(
                    id=TEST_USER_ID,
                    email="owner@example.test",
                    display_name="Owner",
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_ID,
                    slug="local",
                    name="Local workspace",
                    kind="shared",
                )
            )
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=WORKSPACE_ID,
                    user_id=TEST_USER_ID,
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.commit()
    finally:
        await database.dispose()


def install_workbench_dependency_overrides(
    application: FastAPI,
    components: WorkbenchComponents,
) -> None:
    """Route every workbench endpoint to one cohesive component graph."""

    application.dependency_overrides.update(
        {
            plugin_registry: lambda: components.plugin_registry,
            image_upload_service: lambda: components.uploads,
            run_graph_service: lambda: components.run_graph,
            execution_admission_limiter: lambda: components.execution_admission,
            run_execution_manager: lambda: components.execution_manager,
            execution_history_service: lambda: components.execution_history,
            materialization_service: lambda: components.materializations,
            run_result_presenter: lambda: components.presenter,
            artifact_service: lambda: components.artifacts,
            graph_module_catalog: lambda: components.modules,
            graph_module_executor: lambda: components.run_graph,
        }
    )
    install_browser_actor_override(application)


@pytest.fixture
def builtin_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    saved_graph_database = create_database(database_url)
    saved_graphs = SavedGraphService(
        lambda: SqlAlchemySavedGraphUnitOfWork(saved_graph_database.sessions),
        registry,
    )
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=tmp_path / "workbench",
        saved_graphs=saved_graphs,
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )
    install_workbench_dependency_overrides(application, components)
    try:
        with TestClient(application) as client:
            yield client
    finally:
        asyncio.run(saved_graph_database.dispose())


@pytest.fixture
def table_artifact_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, TableArtifactWriter, WorkbenchComponents]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "workbench" / "objects")
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=tmp_path / "workbench",
        unit_of_work=unit_of_work,
        storage=storage,
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )
    install_workbench_dependency_overrides(application, components)
    writer = TableArtifactWriter(
        storage=storage,
        uow=unit_of_work,
        bucket="workbench-artifacts",
        storage_backend="local",
    )
    with TestClient(application) as client:
        yield client, writer, components


@pytest.fixture
def conversion_path_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, InMemoryUnitOfWork]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        (*builtin_plugins(), CONVERSION_PATH_PLUGIN),
        external_plugins=(),
    )
    uow = InMemoryUnitOfWork()
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=tmp_path / "workbench",
        unit_of_work=uow,
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )
    install_workbench_dependency_overrides(application, components)
    with TestClient(application) as client:
        yield client, uow


@pytest.fixture
def structural_projection_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        (*builtin_plugins(), STRUCTURAL_PROJECTION_PLUGIN),
        external_plugins=(),
    )
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=tmp_path / "workbench",
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )
    install_workbench_dependency_overrides(application, components)
    with TestClient(application) as client:
        yield client
