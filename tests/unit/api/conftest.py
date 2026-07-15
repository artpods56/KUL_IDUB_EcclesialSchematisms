import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, cast, final, override

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, StrictStr

from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata

from notarius_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    InMemoryUnitOfWork,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.conversions import ArtifactConversion, ArtifactConversionKey
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.arithmetic import (
    ARITHMETIC_RESULT,
    INTEGER_VALUE,
    ArithmeticResult,
)
from notarius_core.operators.text import TEXT_VALUE
from notarius_core.plugins import Plugin
from notarius_core.runtime.persistence import InlineModelOutputWriter

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.workbench import WorkbenchService
from notarius_api.settings import Settings
from notarius_api.v1.routes.workbench import workbench_service


def _text_to_arithmetic_result(value: str) -> ArithmeticResult:
    integer = int(value)
    return ArithmeticResult(addition=integer + 1, subtraction=integer - 1)


def _failing_text_to_arithmetic_result(value: str) -> ArithmeticResult:
    raise ValueError(f"Cannot convert {value!r}")


def _invalid_text_to_arithmetic_result(value: str) -> ArithmeticResult:
    integer = int(value)
    return cast(
        ArithmeticResult,
        {
            "addition": integer + 1,
            "subtraction": integer - 1,
        },
    )


def _text_to_integer(value: str) -> int:
    return int(value)


TEXT_TO_ARITHMETIC_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_arithmetic_result", 1),
    source=TEXT_VALUE.key,
    target=ARITHMETIC_RESULT.key,
    source_type=str,
    target_type=ArithmeticResult,
    title="As arithmetic result",
    convert=_text_to_arithmetic_result,
)
FAILING_TEXT_TO_ARITHMETIC_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_arithmetic_result_failure", 1),
    source=TEXT_VALUE.key,
    target=ARITHMETIC_RESULT.key,
    source_type=str,
    target_type=ArithmeticResult,
    title="Fail as arithmetic result",
    convert=_failing_text_to_arithmetic_result,
)
INVALID_TEXT_TO_ARITHMETIC_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_invalid_arithmetic_result", 1),
    source=TEXT_VALUE.key,
    target=ARITHMETIC_RESULT.key,
    source_type=str,
    target_type=ArithmeticResult,
    title="As invalid arithmetic result",
    convert=_invalid_text_to_arithmetic_result,
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
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_ARITHMETIC_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(FAILING_TEXT_TO_ARITHMETIC_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(INVALID_TEXT_TO_ARITHMETIC_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_INTEGER)


class ArithmeticResultConsumerInput(NodeInput):
    result: Annotated[ArithmeticResult, InPort(ARITHMETIC_RESULT)]


class ArithmeticResultConsumerOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@CONVERSION_PATH_PLUGIN.node(
    operator_id="test.arithmetic_result_consumer",
    version=1,
    title="Arithmetic result consumer",
)
@final
class ArithmeticResultConsumerNode(
    Node[NoConfig, ArithmeticResultConsumerInput, ArithmeticResultConsumerOutput]
):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ArithmeticResultConsumerInput,
        /,
    ) -> ArithmeticResultConsumerOutput:
        return ArithmeticResultConsumerOutput(
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
    finally:
        await database.dispose()


@pytest.fixture
def builtin_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    service = WorkbenchService(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        )
    )
    application.dependency_overrides[workbench_service] = lambda: service
    with TestClient(application) as client:
        yield client


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
    service = WorkbenchService(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
        uow=uow,
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        )
    )
    application.dependency_overrides[workbench_service] = lambda: service
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
    service = WorkbenchService(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        )
    )
    application.dependency_overrides[workbench_service] = lambda: service
    with TestClient(application) as client:
        yield client
