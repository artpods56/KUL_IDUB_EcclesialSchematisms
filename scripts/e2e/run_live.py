#!/usr/bin/env python3
"""Run the disposable Grafy HTTP E2E deployment and its black-box test."""

import ipaddress
import os
from pathlib import Path
from secrets import token_hex
import socket
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class LiveE2EError(RuntimeError):
    """The disposable live E2E deployment could not complete."""


class DockerNetworkSubnet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subnet: str | None = Field(default=None, alias="Subnet")


class DockerNetworkIpam(BaseModel):
    model_config = ConfigDict(extra="ignore")

    configurations: tuple[DockerNetworkSubnet, ...] | None = Field(
        default=None,
        alias="Config",
    )


class DockerNetworkDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ipam: DockerNetworkIpam = Field(alias="IPAM")


def run_checked(
    command: list[str],
    *,
    repository: Path,
    environment: dict[str, str],
    operation: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=repository,
            env=environment,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except OSError as exc:
        raise LiveE2EError(f"Could not {operation}") from exc
    except subprocess.CalledProcessError as exc:
        detail = ""
        if capture_output:
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = "" if not detail else f": {detail[-4_000:]}"
        raise LiveE2EError(
            f"Failed to {operation} with exit code {exc.returncode}{suffix}"
        ) from exc


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    docker_host = os.environ.get("DOCKER_HOST")
    if sys.platform == "darwin":
        docker_socket = Path("/var/run/docker.sock")
    elif docker_host is None:
        docker_socket = Path("/var/run/docker.sock")
    elif docker_host.startswith("unix://"):
        docker_socket = Path(docker_host.removeprefix("unix://"))
    else:
        raise LiveE2EError(
            "The live E2E test requires a local Unix Docker daemon socket"
        )
    if not docker_socket.is_absolute():
        raise LiveE2EError("The Docker daemon socket path must be absolute")
    try:
        docker_socket_metadata = docker_socket.stat()
    except OSError as exc:
        raise LiveE2EError(
            f"The live E2E test cannot access Docker socket {docker_socket}"
        ) from exc
    if not stat.S_ISSOCK(docker_socket_metadata.st_mode):
        raise LiveE2EError(f"Docker endpoint {docker_socket} is not a Unix socket")

    for address, port in (("127.0.0.1", 18080), ("0.0.0.0", 18443)):
        try:
            with socket.socket() as listener:
                listener.bind((address, port))
        except OSError as exc:
            raise LiveE2EError(
                f"The live E2E test requires unused host port {port}"
            ) from exc

    environment = dict(os.environ)
    broker_tag = "grafy-plugin-egress-broker-e2e:local"
    run_checked(
        [
            "docker",
            "buildx",
            "build",
            "--load",
            "--provenance=false",
            "--sbom=false",
            "--tag",
            broker_tag,
            "--file",
            "infra/docker/plugin-egress-broker.Dockerfile",
            ".",
        ],
        repository=repository,
        environment=environment,
        operation="build the pinned E2E egress broker image",
    )
    inspected = run_checked(
        [
            "docker",
            "image",
            "inspect",
            broker_tag,
            "--format",
            "{{json .RepoDigests}}",
        ],
        repository=repository,
        environment=environment,
        operation="inspect the pinned E2E egress broker image",
        capture_output=True,
    )
    try:
        repo_digests = TypeAdapter(list[str]).validate_json(inspected.stdout)
    except ValidationError as exc:
        raise LiveE2EError(
            "Docker returned invalid repository digests for the E2E broker image"
        ) from exc
    if (
        len(repo_digests) != 1
        or "@sha256:" not in repo_digests[0]
    ):
        raise LiveE2EError(
            "The locally built E2E broker image has no unique immutable digest"
        )
    broker_image = repo_digests[0]
    inspected_socket = run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--volume",
            f"{docker_socket}:/var/run/docker.sock:ro",
            broker_image,
            "python",
            "-c",
            "import os; print(os.stat('/var/run/docker.sock').st_gid)",
        ],
        repository=repository,
        environment=environment,
        operation="inspect the container-visible Docker socket group",
        capture_output=True,
    )
    try:
        docker_socket_gid = int(inspected_socket.stdout.strip())
    except ValueError as exc:
        raise LiveE2EError(
            "Docker returned an invalid container-visible socket group"
        ) from exc
    if docker_socket_gid < 0:
        raise LiveE2EError("Docker returned a negative socket group identifier")

    networks = run_checked(
        ["docker", "network", "ls", "--quiet"],
        repository=repository,
        environment=environment,
        operation="list Docker networks before E2E allocation",
        capture_output=True,
    )
    occupied_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    network_ids = networks.stdout.split()
    if network_ids:
        inspected_networks = run_checked(
            ["docker", "network", "inspect", *network_ids],
            repository=repository,
            environment=environment,
            operation="inspect Docker networks before E2E allocation",
            capture_output=True,
        )
        try:
            network_documents = TypeAdapter(
                list[DockerNetworkDocument]
            ).validate_json(inspected_networks.stdout)
            for document in network_documents:
                configurations = document.ipam.configurations or ()
                for configuration in configurations:
                    if configuration.subnet is None:
                        continue
                    occupied_networks.append(
                        ipaddress.ip_network(configuration.subnet, strict=False)
                    )
        except (ValueError, ValidationError) as exc:
            raise LiveE2EError(
                "Docker returned invalid network allocation metadata"
            ) from exc

    docker_subnet: ipaddress.IPv4Network | None = None
    for third_octet in range(240, 256):
        candidate = ipaddress.IPv4Network(f"10.250.{third_octet}.0/24")
        if not any(candidate.overlaps(occupied) for occupied in occupied_networks):
            docker_subnet = candidate
            break
    if docker_subnet is None:
        raise LiveE2EError("Could not allocate an unused RFC1918 /24 for live E2E")

    project = f"grafy-e2e-{os.getpid()}-{token_hex(4)}"
    compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        "infra/e2e/compose.env",
        "--file",
        "infra/docker/compose.yaml",
        "--file",
        "infra/docker/compose.plugin-runtime.yaml",
        "--file",
        "infra/e2e/compose.yaml",
    ]
    scratch_parent = repository / ".grafy-artifacts/e2e"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(
            prefix="grafy-e2e-workspace-",
            dir=scratch_parent,
        ) as workspace_value,
        TemporaryDirectory(
            prefix="grafy-e2e-control-",
            dir=scratch_parent,
        ) as control_value,
    ):
        workspace = Path(workspace_value).resolve(strict=True)
        control = Path(control_value).resolve(strict=True)
        stack_environment = dict(environment)
        stack_environment.update(
            {
                "GRAFY_E2E_WORKSPACE_ROOT": str(workspace),
                "GRAFY_E2E_CONTROL_ROOT": str(control),
                "GRAFY_E2E_BROKER_IMAGE": broker_image,
                "GRAFY_E2E_HOST_UID": str(os.getuid()),
                "GRAFY_E2E_HOST_GID": str(os.getgid()),
                "GRAFY_E2E_DOCKER_GID": str(docker_socket_gid),
                "GRAFY_E2E_DOCKER_SOCKET": str(docker_socket),
                "GRAFY_DATA_VOLUME": f"{project}-data",
                "GRAFY_DOCKER_NETWORK": f"{project}-internal",
                "GRAFY_DOCKER_SUBNET": str(docker_subnet),
                "GRAFY_DOCKER_GATEWAY": str(docker_subnet.network_address + 1),
            }
        )
        try:
            run_checked(
                [*compose, "config", "--quiet"],
                repository=repository,
                environment=stack_environment,
                operation="validate the disposable E2E Compose deployment",
            )
            run_checked(
                [*compose, "up", "--detach", "--build", "--wait", "api"],
                repository=repository,
                environment=stack_environment,
                operation="start the disposable E2E deployment",
            )
            token_path = control / "workspace.pat"
            try:
                token_metadata = token_path.lstat()
            except OSError as exc:
                raise LiveE2EError(
                    "E2E bootstrap did not produce the Workspace personal token"
                ) from exc
            if (
                not stat.S_ISREG(token_metadata.st_mode)
                or stat.S_IMODE(token_metadata.st_mode) != 0o600
                or token_metadata.st_size > 4_096
            ):
                raise LiveE2EError(
                    "E2E bootstrap produced an unsafe Workspace token file"
                )
            token = token_path.read_text(encoding="utf-8").strip()
            if not token.startswith("nrt_") or "." not in token:
                raise LiveE2EError(
                    "E2E bootstrap produced an invalid Workspace personal token"
                )

            test_environment = dict(environment)
            test_environment.update(
                {
                    "GRAFY_E2E_BASE_URL": "http://127.0.0.1:18080",
                    "GRAFY_E2E_TOKEN": token,
                }
            )
            run_checked(
                [
                    "uv",
                    "run",
                    "--isolated",
                    "--all-extras",
                    "pytest",
                    "-q",
                    "tests/e2e/live/test_llm_image_graph.py",
                ],
                repository=repository,
                environment=test_environment,
                operation="run the live HTTP multimodal graph test",
            )
        except BaseException:
            subprocess.run(
                [
                    *compose,
                    "logs",
                    "--no-color",
                    "--tail",
                    "400",
                    "migrate",
                    "bootstrap",
                    "openai-e2e",
                    "api",
                ],
                cwd=repository,
                env=stack_environment,
                check=False,
                text=True,
            )
            raise
        finally:
            active_error = sys.exception()
            try:
                cleanup = subprocess.run(
                    [
                        *compose,
                        "down",
                        "--volumes",
                        "--remove-orphans",
                        "--rmi",
                        "local",
                    ],
                    cwd=repository,
                    env=stack_environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                if active_error is None:
                    raise LiveE2EError(
                        f"Could not clean up disposable Compose project {project}"
                    ) from exc
                detail = str(exc).strip()[-4_000:]
                print(
                    "Live E2E cleanup warning: could not clean up disposable "
                    f"Compose project {project}: {detail}",
                    file=sys.stderr,
                )
            else:
                if cleanup.returncode != 0:
                    detail = (cleanup.stderr or cleanup.stdout).strip()[-4_000:]
                    message = (
                        "Disposable Compose project cleanup failed for "
                        f"{project} with exit code {cleanup.returncode}: {detail}"
                    )
                    if active_error is None:
                        raise LiveE2EError(message)
                    print(f"Live E2E cleanup warning: {message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LiveE2EError as error:
        print(f"Live E2E failed: {error}", file=sys.stderr)
        sys.exit(1)
