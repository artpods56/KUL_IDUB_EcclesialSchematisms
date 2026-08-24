"""Grafy operator CLI."""

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork

from grafy_api.plugin_authoring import PluginAuthoringService
from grafy_api.plugin_publication import PluginPublicationWorkflow
from grafy_api.plugin_publishing import PluginDirectoryPublisher
from grafy_api.plugin_oci import PluginOciImageBuilder, runtime_profile
from grafy_api.settings import get_settings
from grafy_api.storage import configured_file_storage


async def _run(args: argparse.Namespace) -> None:
    if args.group != "plugin":
        raise ValueError("Unsupported Grafy command")
    settings = get_settings()
    database = create_database(settings.resolved_database_url)
    try:
        storage = configured_file_storage(settings)
        releases = PluginReleaseService(
            lambda: SqlAlchemyUnitOfWork(database.sessions),
            storage,
            bucket=settings.storage_bucket,
        )
        publication = PluginPublicationWorkflow(
            PluginDirectoryPublisher(
                settings.resolved_plugin_roots,
                runtime_profile=settings.plugin_runtime_profile,
                wheelhouse=settings.resolved_plugin_wheelhouse,
            ),
            PluginOciImageBuilder(
                storage,
                bucket=settings.storage_bucket,
                profile=runtime_profile(settings.plugin_runtime_profile),
                docker_binary=settings.plugin_docker_binary,
            ),
            releases,
        )
        workspace_id = UUID(args.workspace)
        if args.command == "publish":
            release = await publication.publish(
                workspace_id=workspace_id,
                directory=Path(args.directory),
                expected_slug=args.slug,
                published_by_user_id=UUID(args.published_by),
            )
            print(
                f"Published Plugin {release.slug} release {release.revision} "
                f"for Workspace {release.workspace_id}"
            )
            return
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
    publish = commands.add_parser("publish")
    publish.add_argument("directory")
    publish.add_argument("--workspace", required=True)
    publish.add_argument(
        "--slug",
        required=True,
        help="Expected Plugin slug; must match the inspected grafy_plugin.PLUGIN "
        "declaration and the established Workspace Plugin identity",
    )
    publish.add_argument("--published-by", required=True)
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
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
