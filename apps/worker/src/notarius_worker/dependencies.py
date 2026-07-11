import os
from collections.abc import Callable

from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_persistence.adapters.in_memory import InMemoryDataStore, InMemoryUnitOfWork
from notarius_persistence.unit_of_work import create_sqlite_uow_factory
from notarius_storage import ArtifactPayloadStoragePort, LocalArtifactPayloadStorage

_STORE = InMemoryDataStore()
_DATABASE_URL = os.getenv("NOTARIUS_DATABASE_URL")
_SQL_UOW_FACTORY = create_sqlite_uow_factory(_DATABASE_URL) if _DATABASE_URL else None


def get_store() -> InMemoryDataStore:
    return _STORE


def create_uow_factory(
    store: InMemoryDataStore | None = None,
) -> Callable[[], StudioUnitOfWorkPort]:
    if _SQL_UOW_FACTORY is not None and store is None:
        return _SQL_UOW_FACTORY
    selected_store = store or get_store()
    return lambda: InMemoryUnitOfWork(selected_store)


def get_artifact_payload_storage() -> ArtifactPayloadStoragePort:
    storage_root = os.getenv(
        "NOTARIUS_ARTIFACT_PAYLOAD_DIR",
        os.getenv("NOTARIUS_OBJECT_STORAGE_DIR", ".notarius-artifacts"),
    )
    return LocalArtifactPayloadStorage(storage_root)
