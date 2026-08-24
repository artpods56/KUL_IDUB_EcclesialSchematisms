"""Hardened local Docker owner for Workspace Plugin guest invocations."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import tarfile
from tempfile import TemporaryDirectory
from typing import Protocol, cast, final, override
from uuid import UUID

from grafy_core.domain.plugin_releases import PluginRelease, PluginRuntimeArtifact
from grafy_core.ports.storage import FileStoragePort
from grafy_core.runtime.plugin_invocation import (
    PluginInvocationError,
    PluginInvocationRequest,
)
from grafy_core.runtime.plugin_protocol import (
    PluginFailureCode,
    PluginInvocationEnvelope,
    PluginInvocationLimits,
)

from grafy_api.plugin_oci import PluginRuntimeProfile

from .plugin_artifacts import (
    PluginGuestRunError,
    PluginGuestRunner,
    PluginInvocationScratch,
)
from .plugin_sandbox import (
    PluginSandboxLifecycle,
    PluginSandboxScopeId,
    current_plugin_sandbox_scope,
)


_SANDBOX_LABEL = "io.grafy.plugin.sandbox=1"
_CONTAINER_INVOCATIONS = PurePosixPath("/run/grafy/invocations")
_RESULT_TAR_LIMIT = 2 * 1_024 * 1_024


class DockerPluginRuntimeError(RuntimeError):
    """The local Docker sandbox owner could not satisfy an operation."""


class _DockerCommandError(RuntimeError):
    pass


class _DockerOutputLimitError(RuntimeError):
    pass


class PluginRuntimeReleaseLookup(Protocol):
    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginRelease | None: ...

    async def list_runtime_artifacts(self) -> list[PluginRuntimeArtifact]: ...


@dataclass(frozen=True, slots=True)
class _SandboxKey:
    scope_id: PluginSandboxScopeId
    workspace_id: UUID
    release_slug: str
    release_revision: int
    source_digest: str


@dataclass(frozen=True, slots=True)
class _Sandbox:
    key: _SandboxKey
    container_id: str
    scratch_root: Path


@dataclass(frozen=True, slots=True)
class PluginSandboxCapacityDiagnostics:
    max_live_sandboxes: int
    live_sandboxes: int
    waiting_sandbox_requests: int


async def _read_bounded(
    stream: asyncio.StreamReader,
    max_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    byte_count = 0
    while chunk := await stream.read(64 * 1_024):
        byte_count += len(chunk)
        if byte_count > max_bytes:
            raise _DockerOutputLimitError(
                f"Docker command output exceeded {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _kill_command(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


@final
class DockerPluginRuntime(
    PluginGuestRunner,
    PluginInvocationScratch,
    PluginSandboxLifecycle,
):
    """Own scope/release containers and execute isolated guest children.

    The Docker socket remains in the API process only. Containers receive no
    socket, host credentials, working copy, database, or object-store access.
    """

    def __init__(
        self,
        *,
        releases: PluginRuntimeReleaseLookup,
        storage: FileStoragePort,
        bucket: str,
        profile: PluginRuntimeProfile,
        scratch_root: Path,
        docker_binary: str = "docker",
        seccomp_profile: Path | None = None,
        max_live_sandboxes: int = 4,
        max_distinct_releases_per_scope: int = 4,
    ) -> None:
        if max_live_sandboxes < 1:
            raise ValueError("Maximum live Plugin sandboxes must be positive")
        if max_distinct_releases_per_scope < 1:
            raise ValueError(
                "Maximum distinct Plugin releases per execution must be positive"
            )
        if max_distinct_releases_per_scope > max_live_sandboxes:
            raise ValueError(
                "Distinct Plugin releases per execution cannot exceed live sandboxes"
            )
        self._releases = releases
        self._storage = storage
        self._bucket = bucket
        self._profile = profile
        self._scratch_root = scratch_root.resolve()
        self._docker_binary = docker_binary
        self._seccomp_profile = seccomp_profile
        self._max_distinct_releases_per_scope = max_distinct_releases_per_scope
        self._max_live_sandboxes = max_live_sandboxes
        self._live_sandbox_capacity = asyncio.BoundedSemaphore(max_live_sandboxes)
        self._waiting_sandbox_requests = 0
        self._sandboxes: dict[_SandboxKey, _Sandbox] = {}
        self._lock = asyncio.Lock()

    @override
    def root_for(self, request: PluginInvocationRequest, /) -> Path:
        scope = current_plugin_sandbox_scope()
        if scope is None:
            raise PluginInvocationError(
                "Workspace Plugin invocation has no top-level sandbox scope"
            )
        release_root = (
            self._scratch_root
            / str(scope.value)
            / (
                f"{request.release.slug}-r{request.release.revision}-"
                f"{request.release.source_digest[:16]}"
            )
        )
        release_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return release_root

    async def recover_orphans(self) -> None:
        completed = await self._docker(
            (
                "ps",
                "-aq",
                "--filter",
                f"label={_SANDBOX_LABEL}",
            ),
            timeout=30,
            max_stdout=1 * 1_024 * 1_024,
            check=True,
        )
        container_ids = completed.stdout.decode("utf-8").split()
        if container_ids:
            await self._docker(
                ("rm", "-f", *container_ids),
                timeout=60,
                max_stdout=1 * 1_024 * 1_024,
                check=True,
            )
        self._scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for child in self._scratch_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise DockerPluginRuntimeError(
                    "Plugin runtime scratch root contains an unexpected entry"
                )
            shutil.rmtree(child)
        await self._remove_unreferenced_images()

    async def check_ready(self) -> None:
        await self._docker(
            ("version", "--format", "{{.Server.Version}}"),
            timeout=3,
            max_stdout=64 * 1_024,
            check=True,
        )

    async def diagnostics(self) -> PluginSandboxCapacityDiagnostics:
        async with self._lock:
            return PluginSandboxCapacityDiagnostics(
                max_live_sandboxes=self._max_live_sandboxes,
                live_sandboxes=len(self._sandboxes),
                waiting_sandbox_requests=self._waiting_sandbox_requests,
            )

    @override
    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        request = PluginInvocationEnvelope.from_json_bytes(
            (invocation_root / "invocation.json").read_bytes()
        )
        scope = current_plugin_sandbox_scope()
        if scope is None:
            raise PluginGuestRunError(
                PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                "Plugin guest has no top-level sandbox scope",
            )
        release = await self._releases.get_by_revision(
            request.workspace_id,
            request.release.slug,
            request.release.revision,
        )
        if release is None or not _release_matches_request(release, request):
            raise PluginGuestRunError(
                PluginFailureCode.CONTRACT_FAILURE,
                "Exact Plugin runtime release is unavailable",
            )
        artifact = release.runtime_artifact
        if artifact is None:
            raise PluginGuestRunError(
                PluginFailureCode.CONTRACT_FAILURE,
                "Plugin release is catalog-only and has no runtime artifact",
            )
        key = _SandboxKey(
            scope_id=scope,
            workspace_id=request.workspace_id,
            release_slug=release.slug,
            release_revision=release.revision,
            source_digest=release.source_digest,
        )
        try:
            sandbox = await self._sandbox_for(
                key,
                release,
                artifact,
                invocation_root.parent,
            )
        except DockerPluginRuntimeError as exc:
            raise PluginGuestRunError(
                PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                str(exc),
            ) from exc
        container_root = _CONTAINER_INVOCATIONS / str(request.invocation_id)
        uncertain_cleanup = False
        try:
            archive = _invocation_tar(invocation_root, str(request.invocation_id))
            await self._docker(
                (
                    "exec",
                    "-i",
                    sandbox.container_id,
                    "/opt/grafy/plugin/.venv/bin/python",
                    "-I",
                    "-c",
                    (
                        "import sys,tarfile; "
                        "archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|'); "
                        "archive.extractall('/run/grafy/invocations',filter='data')"
                    ),
                ),
                input_bytes=archive,
                timeout=30,
                max_stdout=1 * 1_024 * 1_024,
                check=True,
            )
            try:
                await self._docker(
                    (
                        "exec",
                        "--workdir",
                        "/opt/grafy/plugin",
                        "--env",
                        f"TMPDIR={container_root / 'tmp'}",
                        sandbox.container_id,
                        "/opt/grafy/plugin/.venv/bin/python",
                        "-I",
                        "-m",
                        "grafy_core.runtime.plugin_guest",
                        str(container_root),
                    ),
                    timeout=limits.wall_time_seconds,
                    max_stdout=limits.max_log_bytes,
                    max_stderr=limits.max_log_bytes,
                    check=True,
                )
            except asyncio.TimeoutError as exc:
                uncertain_cleanup = True
                raise PluginGuestRunError(
                    PluginFailureCode.TIMEOUT,
                    f"Plugin guest exceeded {limits.wall_time_seconds} seconds",
                ) from exc
            result = await self._docker(
                (
                    "exec",
                    sandbox.container_id,
                    "/opt/grafy/plugin/.venv/bin/python",
                    "-I",
                    "-c",
                    (
                        "import pathlib,sys; "
                        "sys.stdout.buffer.write("
                        "pathlib.Path(sys.argv[1]).joinpath('result.json').read_bytes()"
                        ")"
                    ),
                    str(container_root),
                ),
                timeout=10,
                max_stdout=_RESULT_TAR_LIMIT,
                check=True,
            )
            (invocation_root / "result.json").write_bytes(result.stdout)
            output_tar_limit = (
                limits.max_output_bytes + limits.max_files * 2_048 + 1_024 * 1_024
            )
            output_archive = await self._docker(
                (
                    "exec",
                    sandbox.container_id,
                    "/opt/grafy/plugin/.venv/bin/python",
                    "-I",
                    "-c",
                    (
                        "import sys,tarfile; "
                        "archive=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); "
                        "archive.add(sys.argv[1],arcname='.',recursive=True); "
                        "archive.close()"
                    ),
                    str(container_root / "outputs"),
                ),
                timeout=30,
                max_stdout=output_tar_limit,
                check=True,
            )
            _restore_tar_files(
                output_archive.stdout,
                invocation_root / "outputs",
            )
        except asyncio.CancelledError:
            uncertain_cleanup = True
            raise
        except PluginGuestRunError:
            raise
        except (
            _DockerCommandError,
            _DockerOutputLimitError,
            OSError,
            tarfile.TarError,
        ) as exc:
            uncertain_cleanup = True
            raise PluginGuestRunError(
                PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                "Plugin Docker invocation failed",
            ) from exc
        finally:
            cleanup_task = asyncio.create_task(
                self._cleanup_invocation(
                    sandbox,
                    container_root,
                    destroy_sandbox=uncertain_cleanup,
                )
            )
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
                raise

    @override
    async def close_scope(self, scope_id: PluginSandboxScopeId, /) -> None:
        async with self._lock:
            sandboxes = [
                sandbox
                for key, sandbox in self._sandboxes.items()
                if key.scope_id == scope_id
            ]
            for sandbox in sandboxes:
                self._sandboxes.pop(sandbox.key, None)
        failures: list[Exception] = []
        for sandbox in sandboxes:
            try:
                await self._remove_container(sandbox.container_id)
            except Exception as exc:
                failures.append(exc)
            finally:
                self._live_sandbox_capacity.release()
        scope_root = self._scratch_root / str(scope_id.value)
        if scope_root.exists():
            shutil.rmtree(scope_root)
        if failures:
            raise DockerPluginRuntimeError(
                f"Could not remove {len(failures)} Plugin scope containers"
            ) from failures[0]

    async def shutdown(self) -> None:
        async with self._lock:
            sandboxes = list(self._sandboxes.values())
            self._sandboxes.clear()
        for sandbox in sandboxes:
            try:
                await self._remove_container(sandbox.container_id)
            finally:
                self._live_sandbox_capacity.release()

    async def _sandbox_for(
        self,
        key: _SandboxKey,
        release: PluginRelease,
        artifact: PluginRuntimeArtifact,
        scratch_root: Path,
    ) -> _Sandbox:
        async with self._lock:
            existing = self._sandboxes.get(key)
            if existing is not None:
                return existing
            scope_release_count = sum(
                existing_key.scope_id == key.scope_id
                for existing_key in self._sandboxes
            )
            if scope_release_count >= self._max_distinct_releases_per_scope:
                raise DockerPluginRuntimeError(
                    "Graph execution exceeds its distinct Plugin release limit"
                )
        self._waiting_sandbox_requests += 1
        try:
            await self._live_sandbox_capacity.acquire()
        finally:
            self._waiting_sandbox_requests -= 1
        try:
            async with self._lock:
                existing = self._sandboxes.get(key)
                if existing is not None:
                    self._live_sandbox_capacity.release()
                    return existing
                scope_release_count = sum(
                    existing_key.scope_id == key.scope_id
                    for existing_key in self._sandboxes
                )
                if scope_release_count >= self._max_distinct_releases_per_scope:
                    raise DockerPluginRuntimeError(
                        "Graph execution exceeds its distinct Plugin release limit"
                    )
                await self._ensure_image(release, artifact)
                container_id = await self._create_container(
                    key,
                    artifact,
                    scratch_root,
                )
                sandbox = _Sandbox(
                    key=key,
                    container_id=container_id,
                    scratch_root=scratch_root,
                )
                self._sandboxes[key] = sandbox
                return sandbox
        except BaseException:
            self._live_sandbox_capacity.release()
            raise

    async def _ensure_image(
        self,
        release: PluginRelease,
        artifact: PluginRuntimeArtifact,
    ) -> None:
        image_reference = f"sha256:{artifact.manifest_digest}"
        inspected = await self._docker(
            ("image", "inspect", image_reference),
            timeout=30,
            max_stdout=8 * 1_024 * 1_024,
            check=False,
        )
        if inspected.returncode != 0:
            with TemporaryDirectory(prefix="grafy-plugin-oci-load-") as temporary:
                archive_path = Path(temporary) / "image.oci.tar"
                stream = await self._storage.load(self._bucket, artifact.object_key)
                digest = sha256()
                try:
                    with archive_path.open("wb") as destination:
                        while chunk := stream.read(1 * 1_024 * 1_024):
                            digest.update(chunk)
                            destination.write(chunk)
                finally:
                    stream.close()
                if digest.hexdigest() != artifact.archive_digest:
                    raise DockerPluginRuntimeError(
                        "Stored Plugin OCI archive failed digest validation"
                    )
                await self._docker(
                    ("load", "--input", str(archive_path)),
                    timeout=300,
                    max_stdout=4 * 1_024 * 1_024,
                    check=True,
                )
        labels = await self._docker(
            (
                "image",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                image_reference,
            ),
            timeout=30,
            max_stdout=1 * 1_024 * 1_024,
            check=True,
        )
        parsed = json.loads(labels.stdout)
        if not isinstance(parsed, dict):
            raise DockerPluginRuntimeError("Plugin image labels are unavailable")
        image_labels = cast(dict[str, object], parsed)
        expected = {
            "org.opencontainers.image.source.digest": f"sha256:{release.source_digest}",
            "io.grafy.plugin.runtime": "1",
            "io.grafy.plugin.contract.digest": f"sha256:{release.contract_digest}",
            "io.grafy.plugin.profile.digest": f"sha256:{release.profile_digest}",
            "io.grafy.plugin.base.digest": (
                f"sha256:{self._profile.base_image_digest}"
            ),
            "io.grafy.plugin.protocol.digest": f"sha256:{release.protocol_digest}",
        }
        if any(image_labels.get(name) != value for name, value in expected.items()):
            raise DockerPluginRuntimeError(
                "Plugin image labels do not match the exact release descriptor"
            )

    async def _remove_unreferenced_images(self) -> None:
        retained_artifacts = await self._releases.list_runtime_artifacts()
        retained = {
            f"sha256:{artifact.manifest_digest}" for artifact in retained_artifacts
        }
        listed = await self._docker(
            (
                "image",
                "ls",
                "--quiet",
                "--filter",
                "label=io.grafy.plugin.runtime=1",
            ),
            timeout=30,
            max_stdout=4 * 1_024 * 1_024,
            check=True,
        )
        stale = sorted(set(listed.stdout.decode("utf-8").split()) - retained)
        if stale:
            await self._docker(
                ("image", "rm", "--force", *stale),
                timeout=120,
                max_stdout=4 * 1_024 * 1_024,
                check=True,
            )

    async def _create_container(
        self,
        key: _SandboxKey,
        artifact: PluginRuntimeArtifact,
        scratch_root: Path,
    ) -> str:
        del scratch_root
        name = (
            f"grafy-plugin-{str(key.scope_id.value)[:8]}-"
            f"{key.release_slug[:24]}-r{key.release_revision}-"
            f"{key.source_digest[:8]}"
        )
        seccomp = (
            "seccomp=builtin"
            if self._seccomp_profile is None
            else f"seccomp={self._seccomp_profile}"
        )
        created_at = datetime.now(UTC).isoformat()
        command = (
            "create",
            "--name",
            name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--security-opt",
            seccomp,
            "--cpus",
            str(self._profile.cpu_count),
            "--memory",
            str(self._profile.memory_bytes),
            "--pids-limit",
            str(self._profile.pid_limit),
            "--ulimit",
            (f"nofile={self._profile.open_file_limit}:{self._profile.open_file_limit}"),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777",
            "--tmpfs",
            (
                "/run/grafy/invocations:rw,noexec,nosuid,nodev,"
                f"size={self._profile.scratch_bytes},mode=700,uid=65532,gid=65532"
            ),
            "--label",
            _SANDBOX_LABEL,
            "--label",
            f"io.grafy.plugin.scope={key.scope_id.value}",
            "--label",
            f"io.grafy.plugin.release={key.release_slug}@{key.release_revision}",
            "--label",
            f"io.grafy.plugin.created_at={created_at}",
            f"sha256:{artifact.manifest_digest}",
            "-c",
            "import pathlib,time; pathlib.Path('/tmp/home').mkdir(); time.sleep(10**9)",
        )
        created = await self._docker(
            command,
            timeout=60,
            max_stdout=1 * 1_024 * 1_024,
            check=True,
        )
        container_id = created.stdout.decode("utf-8").strip()
        if not container_id:
            raise DockerPluginRuntimeError("Docker returned no Plugin container ID")
        try:
            await self._docker(
                ("start", container_id),
                timeout=60,
                max_stdout=1 * 1_024 * 1_024,
                check=True,
            )
        except Exception:
            await self._remove_container(container_id)
            raise
        return container_id

    async def _destroy_sandbox(self, sandbox: _Sandbox) -> None:
        async with self._lock:
            owned = self._sandboxes.pop(sandbox.key, None)
        if owned is None:
            return
        try:
            await self._remove_container(sandbox.container_id)
        finally:
            self._live_sandbox_capacity.release()

    async def _cleanup_invocation(
        self,
        sandbox: _Sandbox,
        container_root: PurePosixPath,
        *,
        destroy_sandbox: bool,
    ) -> None:
        if destroy_sandbox:
            await self._destroy_sandbox(sandbox)
            return
        try:
            await self._docker(
                (
                    "exec",
                    sandbox.container_id,
                    "/opt/grafy/plugin/.venv/bin/python",
                    "-I",
                    "-c",
                    "import shutil,sys; shutil.rmtree(sys.argv[1])",
                    str(container_root),
                ),
                timeout=10,
                max_stdout=64 * 1_024,
                check=True,
            )
        except Exception:
            await self._destroy_sandbox(sandbox)

    async def _remove_container(self, container_id: str) -> None:
        completed = await self._docker(
            ("rm", "-f", container_id),
            timeout=60,
            max_stdout=1 * 1_024 * 1_024,
            check=False,
        )
        if completed.returncode == 0:
            return
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if "No such container" not in detail:
            raise _DockerCommandError(
                f"Docker container removal failed with status "
                f"{completed.returncode}: {detail[-2_000:]}"
            )

    @dataclass(frozen=True, slots=True)
    class _Completed:
        returncode: int
        stdout: bytes
        stderr: bytes

    async def _docker(
        self,
        arguments: tuple[str, ...],
        *,
        timeout: int,
        max_stdout: int,
        max_stderr: int = 1 * 1_024 * 1_024,
        input_bytes: bytes | None = None,
        check: bool,
    ) -> _Completed:
        process = await asyncio.create_subprocess_exec(
            self._docker_binary,
            *arguments,
            stdin=(
                asyncio.subprocess.PIPE
                if input_bytes is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            await _kill_command(process)
            raise _DockerCommandError("Docker command did not expose output streams")
        stdout_task = asyncio.create_task(_read_bounded(process.stdout, max_stdout))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr, max_stderr))
        try:
            if input_bytes is not None:
                if process.stdin is None:
                    raise _DockerCommandError("Docker command did not expose stdin")
                process.stdin.write(input_bytes)
                await process.stdin.drain()
                process.stdin.close()
            await asyncio.wait_for(process.wait(), timeout=timeout)
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        except BaseException:
            await _kill_command(process)
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        completed = self._Completed(
            returncode=process.returncode or 0,
            stdout=stdout,
            stderr=stderr,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()[-2_000:]
            raise _DockerCommandError(
                f"Docker command failed with status {completed.returncode}: {detail}"
            )
        return completed


def _release_matches_request(
    release: PluginRelease,
    request: PluginInvocationEnvelope,
) -> bool:
    return (
        release.slug == request.release.slug
        and release.revision == request.release.revision
        and release.source_digest == request.release.source_digest
        and release.contract_digest == request.release.contract_digest
        and release.protocol_digest == request.release.protocol_digest
    )


def _invocation_tar(invocation_root: Path, directory_name: str) -> bytes:
    destination = BytesIO()
    with tarfile.open(fileobj=destination, mode="w") as archive:
        root_info = tarfile.TarInfo(directory_name)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o700
        root_info.uid = 65_532
        root_info.gid = 65_532
        archive.addfile(root_info)
        tmp_info = tarfile.TarInfo(f"{directory_name}/tmp")
        tmp_info.type = tarfile.DIRTYPE
        tmp_info.mode = 0o700
        tmp_info.uid = 65_532
        tmp_info.gid = 65_532
        archive.addfile(tmp_info)
        for path in sorted(invocation_root.rglob("*")):
            if path.is_symlink():
                raise DockerPluginRuntimeError(
                    "Plugin invocation scratch must not contain symlinks"
                )
            relative = path.relative_to(invocation_root).as_posix()
            info = tarfile.TarInfo(f"{directory_name}/{relative}")
            info.uid = 65_532
            info.gid = 65_532
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                archive.addfile(info)
                continue
            if not path.is_file():
                raise DockerPluginRuntimeError(
                    "Plugin invocation scratch contains a non-regular file"
                )
            content = path.read_bytes()
            info.size = len(content)
            info.mode = 0o400 if relative.startswith("inputs/") else 0o600
            archive.addfile(info, BytesIO(content))
    return destination.getvalue()


def _restore_tar_files(
    content: bytes,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(content), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise DockerPluginRuntimeError(
                    "Plugin output archive contains a non-regular file"
                )
            path = PurePosixPath(member.name)
            parts = tuple(part for part in path.parts if part not in {"", "."})
            if not parts or any(part == ".." for part in parts):
                raise DockerPluginRuntimeError(
                    "Plugin output archive contains an unsafe path"
                )
            relative = PurePosixPath(*parts)
            stream = archive.extractfile(member)
            if stream is None:
                raise DockerPluginRuntimeError("Plugin output archive is unreadable")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read())


__all__ = [
    "DockerPluginRuntime",
    "DockerPluginRuntimeError",
    "PluginSandboxCapacityDiagnostics",
]
