import asyncio
import os
import signal
import sys
import tempfile
from typing import Annotated, Literal, Self, cast, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from grafy_core.operators.tables import Table

from grafy_plugin_sql.models import SqlStatement


MAX_RELATIONS = 32
MAX_STATEMENTS = 32
MAX_INPUT_BYTES = 64 * 1_024 * 1_024
MAX_OUTPUT_BYTES = 64 * 1_024 * 1_024
MAX_RESULT_ROWS = 100_000
DEFAULT_WALL_TIME_SECONDS = 35.0


class ArtifactQueryRelationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: StrictStr = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    table: Table


class ArtifactQueryWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statements: list[SqlStatement] = Field(
        min_length=1,
        max_length=MAX_STATEMENTS,
    )
    relations: list[ArtifactQueryRelationPayload] = Field(
        min_length=1,
        max_length=MAX_RELATIONS,
    )

    @model_validator(mode="after")
    def validate_unique_aliases(self) -> Self:
        aliases = [relation.alias.casefold() for relation in self.relations]
        if len(aliases) != len(set(aliases)):
            raise ValueError("relation aliases must be unique ignoring case")
        return self


class ArtifactQueryWorkerError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: StrictStr = Field(min_length=1, max_length=128)
    message: StrictStr = Field(min_length=1, max_length=2_000)
    statement_index: StrictInt | None = Field(default=None, ge=0)


class ArtifactQueryWorkerSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    tables: list[Table] = Field(max_length=MAX_STATEMENTS)


class ArtifactQueryWorkerFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: ArtifactQueryWorkerError


type ArtifactQueryWorkerResponse = Annotated[
    ArtifactQueryWorkerSuccess | ArtifactQueryWorkerFailure,
    Field(discriminator="status"),
]


class ArtifactQueryExecutorError(RuntimeError):
    pass


async def _kill_worker(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


@final
class IsolatedDuckDbArtifactTableExecutor:
    def __init__(self, *, wall_time_seconds: float = DEFAULT_WALL_TIME_SECONDS) -> None:
        if wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive")
        self._wall_time_seconds = wall_time_seconds

    async def execute(
        self,
        statements: list[SqlStatement],
        relation_aliases: list[str],
        relations: list[Table],
        /,
    ) -> list[Table]:
        if len(relation_aliases) != len(relations):
            raise ArtifactQueryExecutorError(
                f"Received {len(relation_aliases)} relation aliases for "
                f"{len(relations)} tables"
            )
        request = ArtifactQueryWorkerRequest(
            statements=statements,
            relations=[
                ArtifactQueryRelationPayload(alias=alias, table=table)
                for alias, table in zip(relation_aliases, relations, strict=True)
            ],
        )
        payload = request.model_dump_json().encode("utf-8")
        if len(payload) > MAX_INPUT_BYTES:
            raise ArtifactQueryExecutorError(
                f"Artifact query input contains {len(payload)} bytes, exceeding "
                f"the {MAX_INPUT_BYTES}-byte limit"
            )

        with tempfile.TemporaryDirectory(prefix="grafy-artifact-sql-") as workdir:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-E",
                "-s",
                "-P",
                "-B",
                "-X",
                "utf8",
                "-m",
                "grafy_plugin_sql.artifact_query_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={
                    "LANG": "C.UTF-8",
                    "PATH": os.defpath,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONUTF8": "1",
                    "TMPDIR": workdir,
                },
                close_fds=True,
                start_new_session=True,
            )
            try:
                async with asyncio.timeout(self._wall_time_seconds):
                    stdout, stderr = await process.communicate(payload)
            except TimeoutError as exc:
                await _kill_worker(process)
                raise ArtifactQueryExecutorError(
                    "Artifact query worker exceeded the "
                    f"{self._wall_time_seconds:g}-second wall-time limit"
                ) from exc
            except asyncio.CancelledError:
                await _kill_worker(process)
                raise

        if process.returncode != 0:
            raise ArtifactQueryExecutorError(
                f"Artifact query worker exited with code {process.returncode}; "
                f"captured {len(stderr)} bytes of diagnostic output"
            )
        if len(stdout) > MAX_OUTPUT_BYTES:
            raise ArtifactQueryExecutorError(
                f"Artifact query worker returned {len(stdout)} bytes, exceeding "
                f"the {MAX_OUTPUT_BYTES}-byte limit"
            )
        try:
            response = cast(
                ArtifactQueryWorkerResponse,
                TypeAdapter(ArtifactQueryWorkerResponse).validate_json(stdout),
            )
        except Exception as exc:
            raise ArtifactQueryExecutorError(
                "Artifact query worker returned an invalid response"
            ) from exc
        if isinstance(response, ArtifactQueryWorkerFailure):
            statement_context = ""
            if response.error.statement_index is not None:
                statement_context = (
                    f" at statement index {response.error.statement_index}"
                )
            raise ArtifactQueryExecutorError(
                f"Artifact query worker rejected the batch{statement_context}: "
                f"{response.error.message} ({response.error.kind})"
            )
        if len(response.tables) != len(statements):
            raise ArtifactQueryExecutorError(
                f"Artifact query worker returned {len(response.tables)} tables for "
                f"{len(statements)} statements"
            )
        return response.tables


__all__ = [
    "ArtifactQueryExecutorError",
    "ArtifactQueryRelationPayload",
    "ArtifactQueryWorkerError",
    "ArtifactQueryWorkerFailure",
    "ArtifactQueryWorkerRequest",
    "ArtifactQueryWorkerSuccess",
    "IsolatedDuckDbArtifactTableExecutor",
    "MAX_INPUT_BYTES",
    "MAX_OUTPUT_BYTES",
    "MAX_RESULT_ROWS",
]
