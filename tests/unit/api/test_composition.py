from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, StrictInt

from notarius_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec, JsonObject
from notarius_core.plugins import Plugin, PluginRuntimeContext
from notarius_core.runtime.resolvers import InlineModelResolver

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.composition import build_workbench_components


class CompositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictInt


COMPOSITION_ARTIFACT = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.composition", 1),
    title="Composition test",
    payload_schema=cast(JsonObject, CompositionPayload.model_json_schema()),
)


def test_plugin_factories_receive_an_existing_upload_directory(tmp_path: Path) -> None:
    observed_upload_directories: list[Path] = []
    plugin = Plugin(slug="test.composition", title="Composition test")
    plugin.register_artifact_type(COMPOSITION_ARTIFACT)

    def resolver_factory(
        context: PluginRuntimeContext,
    ) -> InlineModelResolver[CompositionPayload]:
        assert context.uploads_dir.is_dir()
        observed_upload_directories.append(context.uploads_dir)
        return InlineModelResolver(
            source=COMPOSITION_ARTIFACT.key,
            target=CompositionPayload,
            uow=context.uow,
        )

    plugin.register_resolver(resolver_factory)
    registry = build_plugin_registry(
        (*builtin_plugins(), plugin),
        external_plugins=(),
    )

    build_workbench_components(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
        execution_backend="inline",
    )

    assert observed_upload_directories == [
        (tmp_path / "workbench" / "uploads").resolve()
    ]
