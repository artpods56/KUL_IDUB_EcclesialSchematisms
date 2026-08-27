"""Grafy operator CLI."""

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_core.domain.plugin_releases import PlatformPluginActor
from grafy_core.domain.plugin_revocations import PluginReleaseRevocationReason
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork

from grafy_api.plugin_admission import isolated_release_admission
from grafy_api.plugin_authoring import PluginAuthoringService
from grafy_api.plugin_publication import (
    PluginPublicationWorkflow,
    SystemPluginPublicationWorkflow,
    SystemPluginRevocationWorkflow,
)
from grafy_api.plugin_publisher_sandbox import DockerPluginDirectoryPublisher
from grafy_api.plugin_publishing import (
    PluginDirectoryPublisher,
    PluginPublishingError,
)
from grafy_api.plugin_oci import PluginOciImageBuilder, runtime_profile
from grafy_api.network_policy import load_network_policy_manifest
from grafy_api.settings import get_settings
from grafy_api.storage import configured_file_storage
from grafy_api.system_cutover import (
    SystemBaselineCutoverService,
    SystemCutoverCommand,
)
from grafy_api.system_cutover_operations import (
    create_rollback_unit_manifest,
    generate_system_baseline_file,
    load_rollback_unit,
    load_system_baseline_manifest,
    verify_rollback_unit_manifest,
)
from grafy_api.system_host_bindings import SystemHostPluginBinding
from grafy_api.system_plugin_deployment import SystemPluginDeploymentManifestBuilder
from grafy_api.system_plugin_inventory import (
    CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH,
    SystemBaselineManifestGenerator,
    load_system_plugin_inventory,
)
from grafy_api.system_plugin_loader import load_system_plugin_deployment_file
from grafy_core.runtime.plugin_loader import WORKSPACE_PLUGIN_LOADER_TARGET


class PluginCheckReport(BaseModel):
    """Read-only CLI summary of one verified Plugin working copy."""

    status: Literal["valid"] = "valid"
    slug: str
    loader_target: str
    node_count: int = Field(ge=0)
    artifact_type_count: int = Field(ge=0)
    capabilities: tuple[str, ...]
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_profile: str


async def _run_system_cutover(args: argparse.Namespace) -> None:
    if args.command == "create-rollback-unit":
        result = create_rollback_unit_manifest(
            rollback_unit_id=args.rollback_unit_id,
            database_backup=args.database_backup,
            release_objects=args.release_objects,
            artifact_storage=args.artifact_storage,
            migration_manifest=args.migration_manifest,
            output=args.output,
        )
        print(result.model_dump_json(indent=2))
        return
    if args.command == "verify-rollback-unit":
        result = verify_rollback_unit_manifest(
            manifest=args.manifest,
            database_backup=args.database_backup,
            release_objects=args.release_objects,
            artifact_storage=args.artifact_storage,
            migration_manifest=args.migration_manifest,
        )
        print(result.model_dump_json(indent=2))
        return

    settings = get_settings()
    database = create_database(settings.resolved_database_url)
    try:
        if args.command == "generate-baseline":
            result = await generate_system_baseline_file(
                SystemBaselineManifestGenerator(database.sessions),
                inventory_path=args.inventory,
                output=args.output,
                deployment_manifest_path=args.deployment_manifest,
            )
            print(result.model_dump_json(indent=2))
            return

        baseline = load_system_baseline_manifest(args.baseline)
        rollback_unit = load_rollback_unit(args.rollback_unit)
        command = SystemCutoverCommand(
            mode="dry-run" if args.command == "audit" else "apply",
            baseline=baseline,
            rollback_unit=rollback_unit,
            expected_precondition_token=(
                None if args.command == "audit" else args.precondition_token
            ),
        )
        report = await SystemBaselineCutoverService(database.sessions).execute(command)
        print(report.model_dump_json(indent=2))
    finally:
        await database.dispose()


async def _run_network_policy(args: argparse.Namespace) -> None:
    """Validate the deployment network policy without mutating anything."""

    settings = get_settings()
    manifest = args.manifest or settings.resolved_network_policy_manifest
    if manifest is None:
        policy = settings.resolved_network_policy
        source = "legacy GRAFY_PLUGIN_*_EGRESS_DESTINATIONS translation"
    else:
        policy = load_network_policy_manifest(manifest)
        source = str(manifest)
    print(f"network policy source: {source}")
    for profile in policy.profiles:
        print(
            "profile plane="
            f"{profile.plane.value} name={profile.name} "
            f"mode={profile.mode.value} origins={len(profile.allowed_origins)} "
            f"digest={profile.policy_digest}"
        )
    for assignment in policy.assignments:
        target = "plane-default"
        if assignment.scope is not None:
            target = f"scope={assignment.scope.value}"
            if assignment.workspace_id is not None:
                target += f" workspace={assignment.workspace_id}"
        if assignment.slug is not None:
            target += f" slug={assignment.slug}"
        if assignment.revision is not None:
            target += f" revision={assignment.revision}"
        print(
            f"assignment plane={assignment.plane.value} {target} -> "
            f"{assignment.profile}"
        )
    print("network policy OK")


async def _run(args: argparse.Namespace) -> None:
    if args.group == "network-policy":
        await _run_network_policy(args)
        return
    if args.group == "system-cutover":
        await _run_system_cutover(args)
        return
    if args.group != "plugin":
        raise ValueError("Unsupported Grafy command")
    settings = get_settings()
    if args.command == "check":
        directory = Path(args.directory)
        loader_target = WORKSPACE_PLUGIN_LOADER_TARGET
        expected_slug: str | None = None
        system_inventory = load_system_plugin_inventory(
            CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH
        )
        repository_root = CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH.parents[1]
        resolved_directory = directory.expanduser().resolve()
        for entry in system_inventory.plugins:
            if resolved_directory == (repository_root / entry.project).resolve():
                loader_target = entry.loader_target
                expected_slug = entry.slug
                break
        publisher = PluginDirectoryPublisher(
            settings.resolved_plugin_roots,
            runtime_profile=settings.plugin_runtime_profile,
            wheelhouse=settings.resolved_plugin_wheelhouse,
        )
        try:
            verified = await asyncio.to_thread(
                publisher.verify,
                directory,
                expected_slug=expected_slug,
                loader_target=loader_target,
            )
        except PluginPublishingError as exc:
            print(f"Plugin check failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        report = PluginCheckReport(
            slug=verified.catalog.slug,
            loader_target=verified.loader_target,
            node_count=len(verified.catalog.nodes),
            artifact_type_count=len(verified.catalog.artifact_types),
            capabilities=tuple(
                capability.value for capability in verified.capabilities.capabilities
            ),
            source_digest=verified.source_digest,
            lock_digest=verified.lock_digest,
            runtime_profile=verified.runtime_profile,
        )
        print(report.model_dump_json(indent=2))
        return
    database = create_database(settings.resolved_database_url)
    try:
        storage = configured_file_storage(settings)
        releases = PluginReleaseService(
            lambda: SqlAlchemyUnitOfWork(database.sessions),
            storage,
            bucket=settings.storage_bucket,
        )
        if args.command == "revoke-system":
            revocation = await SystemPluginRevocationWorkflow(
                database.sessions,
                releases,
            ).revoke(
                slug=args.slug,
                revision=args.revision,
                reason=PluginReleaseRevocationReason(args.reason),
                platform_actor=PlatformPluginActor(args.platform_actor),
            )
            print(
                f"Revoked System Plugin {revocation.slug} release "
                f"{revocation.revision} for {revocation.reason.value}"
            )
            return
        profile = runtime_profile(
            settings.plugin_runtime_profile,
            native_base_image=settings.plugin_runtime_native_base_image,
            native_base_image_digest=(settings.plugin_runtime_native_base_image_digest),
        )
        image_builder = PluginOciImageBuilder(
            storage,
            bucket=settings.storage_bucket,
            profile=profile,
            docker_binary=settings.plugin_docker_binary,
        )
        system_inventory = load_system_plugin_inventory(
            CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH
        )
        if args.command == "build-system-deployment":
            manifest = await SystemPluginDeploymentManifestBuilder(
                database.sessions
            ).build(
                system_inventory,
                repository_root=CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH.parents[1],
                output=args.output,
                slug=args.slug,
                revision=args.revision,
            )
            print(manifest.model_dump_json(indent=2))
            return
        if args.command == "promote" or (
            args.command == "publish" and args.global_scope
        ):
            host_bindings: tuple[SystemHostPluginBinding, ...] = ()
            if args.command == "promote" and args.deployment_manifest is not None:
                deployment = load_system_plugin_deployment_file(
                    args.deployment_manifest
                )
                host_bindings = deployment.bindings
            system_publication = SystemPluginPublicationWorkflow(
                image_builder,
                releases,
                isolated_release_admission(
                    profile=profile,
                    egress_policy=settings.resolved_plugin_egress_policy,
                    network_policy=settings.resolved_network_policy,
                    system_host_bindings=host_bindings,
                ),
                system_inventory,
            )
            platform_actor = PlatformPluginActor(args.actor)
            if args.command == "publish":
                if args.sandbox_image is None:
                    raise SystemExit(
                        "--sandbox-image is required for global Plugin publication"
                    )
                inventory_entry = system_inventory.entry_for(args.slug)
                if settings.resolved_plugin_wheelhouse is not None:
                    raise SystemExit(
                        "System sandbox publication requires dependencies to be "
                        "present in the frozen source or fetched from configured "
                        "package indexes; host wheelhouse mounts are not supported"
                    )
                publisher = DockerPluginDirectoryPublisher(
                    settings.resolved_plugin_roots,
                    runtime_profile=settings.plugin_runtime_profile,
                    image=args.sandbox_image,
                    docker_binary=settings.plugin_docker_binary,
                    scratch_root=(
                        None
                        if args.sandbox_scratch_root is None
                        else Path(args.sandbox_scratch_root)
                    ),
                )
                verified = await asyncio.to_thread(
                    publisher.verify,
                    Path(args.directory),
                    expected_slug=args.slug,
                    loader_target=inventory_entry.loader_target,
                )
                release = await system_publication.publish_verified(
                    verified,
                    platform_actor=platform_actor,
                )
                print(
                    f"Published global Plugin {release.slug} release "
                    f"{release.revision}; promote it explicitly to activate it"
                )
                return
            selection = await system_publication.promote(
                slug=args.slug,
                revision=args.revision,
                platform_actor=platform_actor,
                expected_generation=args.expected_generation,
            )
            print(
                f"Promoted System Plugin {selection.slug} release "
                f"{selection.selected_revision} at generation {selection.generation}"
            )
            return
        if args.command == "publish" and args.publisher_sandbox:
            if args.sandbox_image is None:
                raise SystemExit("--sandbox-image is required with --publisher-sandbox")
            if settings.resolved_plugin_wheelhouse is not None:
                raise SystemExit(
                    "Docker publisher mode does not support host wheelhouse mounts"
                )
            publisher: PluginDirectoryPublisher = DockerPluginDirectoryPublisher(
                settings.resolved_plugin_roots,
                runtime_profile=settings.plugin_runtime_profile,
                image=args.sandbox_image,
                docker_binary=settings.plugin_docker_binary,
                scratch_root=(
                    None
                    if args.sandbox_scratch_root is None
                    else Path(args.sandbox_scratch_root)
                ),
            )
        else:
            publisher = PluginDirectoryPublisher(
                settings.resolved_plugin_roots,
                runtime_profile=settings.plugin_runtime_profile,
                wheelhouse=settings.resolved_plugin_wheelhouse,
            )
        publication = PluginPublicationWorkflow(
            publisher,
            image_builder,
            releases,
            system_inventory,
        )
        if args.command == "publish":
            workspace_id = UUID(args.workspace)
            release = await publication.publish(
                workspace_id=workspace_id,
                directory=Path(args.directory),
                expected_slug=args.slug,
                published_by_user_id=UUID(args.actor),
            )
            print(
                f"Published Plugin {release.slug} release {release.revision} "
                f"for Workspace {release.workspace_id}"
            )
            return
        workspace_id = UUID(args.workspace)
        authoring = PluginAuthoringService(
            authoring_root=settings.resolved_plugin_authoring_root,
            allowed_roots=settings.resolved_plugin_roots,
            sdk_project=settings.resolved_plugin_sdk_project,
            publication=publication,
            releases=releases,
            storage=storage,
            bucket=settings.storage_bucket,
        )
        actor_user_id = UUID(args.actor)
        if args.command == "scaffold":
            reservation = await asyncio.to_thread(
                authoring.scaffold,
                workspace_id=workspace_id,
                slug=args.slug,
                operator_slug=args.operator,
                title=args.title,
                actor_user_id=actor_user_id,
            )
            print(
                f"Scaffolded {reservation.project_directory} with authoring "
                f"session {reservation.session_id} at "
                f"sha256:{reservation.source_digest}"
            )
            return
        if args.command == "reserve":
            reservation = authoring.reserve(
                workspace_id=workspace_id,
                slug=args.slug,
                actor_user_id=actor_user_id,
            )
            print(
                f"Reserved {reservation.project_directory} with authoring "
                f"session {reservation.session_id} at "
                f"sha256:{reservation.source_digest}"
            )
            return
        session_id = UUID(args.session)
        if args.command == "review":
            review = await authoring.review(
                workspace_id=workspace_id,
                slug=args.slug,
                actor_user_id=actor_user_id,
                session_id=session_id,
            )
            print(review.model_dump_json(indent=2))
            return
        if args.command == "publish-reviewed":
            release = await authoring.publish(
                workspace_id=workspace_id,
                slug=args.slug,
                actor_user_id=actor_user_id,
                session_id=session_id,
            )
            print(
                f"Published reviewed Plugin {release.slug} release "
                f"{release.revision} for Workspace {release.workspace_id}"
            )
            return
        if args.command == "release-reservation":
            authoring.release(
                workspace_id=workspace_id,
                slug=args.slug,
                actor_user_id=actor_user_id,
                session_id=session_id,
            )
            print(f"Released Plugin {args.slug!r} authoring session {session_id}")
            return
        raise ValueError("Unsupported Grafy Plugin command")
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="grafy")
    groups = parser.add_subparsers(dest="group", required=True)
    plugin = groups.add_parser("plugin")
    commands = plugin.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check",
        help="verify a Plugin directory without publishing it",
    )
    check.add_argument("directory")
    publish = commands.add_parser(
        "publish",
        help="verify and publish a Plugin to a Workspace or globally",
    )
    publish.add_argument("directory")
    publish_target = publish.add_mutually_exclusive_group(required=True)
    publish_target.add_argument("--workspace")
    publish_target.add_argument(
        "--global",
        dest="global_scope",
        action="store_true",
        help="publish a global Plugin without activating it",
    )
    publish.add_argument(
        "--slug",
        required=True,
        help="expected Plugin slug; must match the inspected declaration",
    )
    publish.add_argument(
        "--actor",
        required=True,
        help="Workspace user UUID or global platform actor reference",
    )
    publish.add_argument(
        "--publisher-sandbox",
        action="store_true",
        help="verify in the Docker publisher boundary instead of host subprocesses",
    )
    publish.add_argument("--sandbox-image")
    publish.add_argument("--sandbox-scratch-root")

    build_system_deployment = commands.add_parser("build-system-deployment")
    build_system_deployment.add_argument("--output", required=True, type=Path)
    build_system_deployment.add_argument("--slug")
    build_system_deployment.add_argument("--revision", type=int)

    promote = commands.add_parser(
        "promote",
        help="activate one published global Plugin release",
    )
    promote.add_argument("--slug", required=True)
    promote.add_argument("--revision", required=True, type=int)
    promote.add_argument("--actor", required=True)
    promote.add_argument("--expected-generation", type=int)
    promote.add_argument(
        "--deployment-manifest",
        type=Path,
        help="Exact host bindings; required only for host-eligible System Plugins",
    )

    revoke_system = commands.add_parser("revoke-system")
    revoke_system.add_argument("--slug", required=True)
    revoke_system.add_argument("--revision", required=True, type=int)
    revoke_system.add_argument(
        "--reason",
        required=True,
        choices=tuple(reason.value for reason in PluginReleaseRevocationReason),
    )
    revoke_system.add_argument("--platform-actor", required=True)
    for name in (
        "scaffold",
        "reserve",
        "review",
        "publish-reviewed",
        "release-reservation",
    ):
        command = commands.add_parser(name)
        command.add_argument("--workspace", required=True)
        command.add_argument("--slug", required=True)
        command.add_argument("--actor", required=True)
        if name not in {"scaffold", "reserve"}:
            command.add_argument("--session", required=True)
        if name == "scaffold":
            command.add_argument("--operator", required=True)
            command.add_argument("--title", required=True)

    network_policy_group = groups.add_parser("network-policy")
    network_policy_commands = network_policy_group.add_subparsers(
        dest="command",
        required=True,
    )
    validate_policy = network_policy_commands.add_parser(
        "validate",
        help="Validate the deployment network policy (non-mutating)",
    )
    validate_policy.add_argument(
        "--manifest",
        type=Path,
        help="Manifest to validate; defaults to GRAFY_NETWORK_POLICY_MANIFEST",
    )

    system_cutover = groups.add_parser("system-cutover")
    cutover_commands = system_cutover.add_subparsers(
        dest="command",
        required=True,
    )
    generate_baseline = cutover_commands.add_parser("generate-baseline")
    generate_baseline.add_argument("--inventory", required=True, type=Path)
    generate_baseline.add_argument("--output", required=True, type=Path)
    generate_baseline.add_argument("--deployment-manifest", type=Path)

    create_rollback = cutover_commands.add_parser("create-rollback-unit")
    create_rollback.add_argument("--rollback-unit-id", required=True)
    create_rollback.add_argument("--database-backup", required=True, type=Path)
    create_rollback.add_argument("--release-objects", required=True, type=Path)
    create_rollback.add_argument("--artifact-storage", required=True, type=Path)
    create_rollback.add_argument("--migration-manifest", required=True, type=Path)
    create_rollback.add_argument("--output", required=True, type=Path)

    verify_rollback = cutover_commands.add_parser("verify-rollback-unit")
    verify_rollback.add_argument("--manifest", required=True, type=Path)
    verify_rollback.add_argument("--database-backup", required=True, type=Path)
    verify_rollback.add_argument("--release-objects", required=True, type=Path)
    verify_rollback.add_argument("--artifact-storage", required=True, type=Path)
    verify_rollback.add_argument("--migration-manifest", required=True, type=Path)

    audit_cutover = cutover_commands.add_parser("audit")
    audit_cutover.add_argument("--baseline", required=True, type=Path)
    audit_cutover.add_argument("--rollback-unit", required=True, type=Path)

    apply_cutover = cutover_commands.add_parser("apply")
    apply_cutover.add_argument("--baseline", required=True, type=Path)
    apply_cutover.add_argument("--rollback-unit", required=True, type=Path)
    apply_cutover.add_argument("--precondition-token", required=True)
    args = parser.parse_args()
    if (
        args.group == "plugin"
        and args.command == "publish"
        and args.global_scope
        and args.sandbox_image is None
    ):
        parser.error("--sandbox-image is required with plugin publish --global")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
