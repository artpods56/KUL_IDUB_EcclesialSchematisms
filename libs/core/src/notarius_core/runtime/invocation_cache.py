import json
from collections.abc import Mapping, Sequence
from enum import Enum
from hashlib import sha256
from typing import Any, Protocol, cast, final
from uuid import UUID

from pydantic import BaseModel

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence, ArtifactTypeKey
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.nodes import Node, NodeExecutionContext
from notarius_core.runtime.invocation import NodeInvocation


INVOCATION_CACHE_FINGERPRINT_VERSION = 1


class InvocationCachePort(Protocol):
    async def get(self, key_sha256: str) -> InvocationCacheEntry | None: ...

    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool: ...

    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool: ...


@final
class DisabledInvocationCache(InvocationCachePort):
    async def get(self, key_sha256: str) -> InvocationCacheEntry | None:
        del key_sha256
        return None

    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool:
        del entry
        return False

    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        del key_sha256, generation
        return False


class _UncacheableInvocation(ValueError):
    pass


def invocation_cache_key(
    *,
    node: Node[Any, Any, Any],
    context: NodeExecutionContext,
    inputs: Mapping[str, object],
    config: BaseModel,
    invocation: NodeInvocation,
    artifact_type_bindings: Mapping[str, ArtifactTypeKey],
    opaque_secret_revisions: Mapping[str, str],
) -> str | None:
    """Return the versioned exact-invocation SHA-256, or None when unsafe."""

    node_id = context.node_id
    if node_id is None or node_id.strip() == "":
        return None
    if any(item.strip() == "" for item in context.module_path):
        return None

    try:
        canonical_inputs = {
            name: _canonical_value(value) for name, value in sorted(inputs.items())
        }
        canonical_bindings = [
            {
                "variable": variable,
                "artifact_type": {
                    "id": artifact_type.id,
                    "schema_version": artifact_type.schema_version,
                },
            }
            for variable, artifact_type in sorted(artifact_type_bindings.items())
        ]
        canonical_secret_revisions: list[dict[str, str]] = []
        for name, revision in sorted(opaque_secret_revisions.items()):
            if name.strip() == "" or revision.strip() == "":
                raise _UncacheableInvocation(
                    "Opaque secret revision names and values must not be blank"
                )
            canonical_secret_revisions.append(
                {
                    "name": name,
                    "revision": revision,
                }
            )
        document = {
            "fingerprint_version": INVOCATION_CACHE_FINGERPRINT_VERSION,
            "operator": {
                "id": node.operator_id,
                "version": node.operator_version,
            },
            "node": {
                "id": node_id,
                "module_path": list(context.module_path),
            },
            "config": config.model_dump(mode="json", by_alias=True),
            "invocation": {
                "mode": invocation.mode.value,
                "map_input": invocation.map_input,
                "item_index": context.invocation_index,
            },
            "artifact_type_bindings": canonical_bindings,
            "inputs": canonical_inputs,
            "opaque_secret_revisions": canonical_secret_revisions,
        }
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (_UncacheableInvocation, TypeError, ValueError):
        return None
    return sha256(payload).hexdigest()


def _canonical_value(value: object) -> object:
    if isinstance(value, ArtifactRef):
        return _canonical_ref(value)
    if isinstance(value, ArtifactRefSequence):
        return {
            "kind": "artifact_ref_sequence",
            "sequence_id": str(value.sequence_id),
            "artifact_type": value.artifact_type,
            "schema_version": value.schema_version,
            "item_refs": [_canonical_ref(ref) for ref in value.item_refs],
            "ordered": value.ordered,
            "index_key": value.index_key,
            "metadata": _canonical_value(value.metadata),
        }
    if isinstance(value, BaseModel):
        return {
            "kind": "model",
            "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "value": _canonical_value(value.model_dump(mode="json", by_alias=True)),
        }
    if isinstance(value, Mapping):
        raw_mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in raw_mapping):
            raise _UncacheableInvocation("Cacheable mappings require string keys")
        return {
            "kind": "mapping",
            "items": [
                [key, _canonical_value(raw_mapping[key])]
                for key in sorted(cast(Sequence[str], tuple(raw_mapping)))
            ],
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return {
            "kind": "list",
            "items": [_canonical_value(item) for item in items],
        }
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return {
            "kind": "tuple",
            "items": [_canonical_value(item) for item in items],
        }
    if isinstance(value, UUID):
        return {"kind": "uuid", "value": str(value)}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise _UncacheableInvocation(
        f"Unsupported cache fingerprint value {type(value).__name__}"
    )


def _canonical_ref(ref: ArtifactRef) -> dict[str, object]:
    content_hash = ref.content_hash
    if (
        content_hash is None
        or len(content_hash) != 64
        or any(character not in "0123456789abcdef" for character in content_hash)
    ):
        raise _UncacheableInvocation(
            f"Artifact {ref.artifact_id} does not carry a canonical SHA-256"
        )
    return {
        "kind": "artifact_ref",
        "artifact_id": str(ref.artifact_id),
        "artifact_type": ref.artifact_type,
        "schema_version": ref.schema_version,
        "content_hash": content_hash,
    }


__all__ = [
    "INVOCATION_CACHE_FINGERPRINT_VERSION",
    "DisabledInvocationCache",
    "InvocationCachePort",
    "invocation_cache_key",
]
