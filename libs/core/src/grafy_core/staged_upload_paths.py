from pathlib import Path
from uuid import UUID

from grafy_core.ports.staged_uploads import StagedUploadUnitOfWorkPort


def resolve_staged_upload_path(
    uploads_dir: Path,
    *,
    workspace_id: UUID,
    upload_key: str,
) -> Path:
    """Resolve a staged upload under ``uploads_dir/{workspace_id}/{upload_key}``.

    ``upload_key`` must be one opaque relative name. Path traversal and
    cross-workspace resolution fail closed.
    """
    relative_path = Path(upload_key)
    if (
        relative_path.is_absolute()
        or relative_path.parts != (upload_key,)
        or upload_key in {".", ".."}
        or "\\" in upload_key
        or "\x00" in upload_key
    ):
        raise ValueError(
            f"Upload key {upload_key!r} must be one opaque relative name"
        )

    workspace_dir = (uploads_dir / str(workspace_id)).resolve()
    path = (workspace_dir / relative_path).resolve()
    if path.parent != workspace_dir:
        raise ValueError(
            f"Upload key {upload_key!r} resolves outside workspace "
            f"{workspace_id} uploads directory"
        )
    return path


async def resolve_persisted_staged_upload_path(
    uploads_dir: Path,
    unit_of_work: StagedUploadUnitOfWorkPort,
    *,
    workspace_id: UUID,
    upload_key: str,
) -> Path:
    """Resolve a staged upload that has a live workspace ``StagedUpload`` row.

    A file on disk is not authorization. Missing or foreign-workspace rows
    fail closed even when a file exists at the derived path.
    """
    resolve_staged_upload_path(
        uploads_dir,
        workspace_id=workspace_id,
        upload_key=upload_key,
    )
    async with unit_of_work as entered:
        record = await entered.staged_uploads.get(workspace_id, upload_key)
    if record is None:
        raise FileNotFoundError(
            f"Staged upload {upload_key!r} was not found in workspace "
            f"{workspace_id}"
        )
    return resolve_staged_upload_path(
        uploads_dir,
        workspace_id=record.workspace_id,
        upload_key=record.upload_key,
    )


__all__ = [
    "resolve_persisted_staged_upload_path",
    "resolve_staged_upload_path",
]
