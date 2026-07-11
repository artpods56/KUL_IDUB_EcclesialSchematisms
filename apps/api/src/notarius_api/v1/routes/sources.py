import re
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import ArtifactResponse, ArtifactSequenceResponse
from notarius_api.schemas.studio import (
    ImageSourceUploadResponse,
    PdfSourceUploadResponse,
    SourceCreate,
    SourceItemResponse,
    SourceResponse,
)
from notarius_api.services.pdf_sources import PdfSourceIngestor
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import Artifact, ArtifactSequence, Source, SourceItem
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_storage import (
    ArtifactPayloadStoragePort,
    SaveArtifactPayloadCommand,
    artifact_payload_ref,
)

router = APIRouter(tags=["sources"])


@router.post(
    "/projects/{project_id}/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_source(
    project_id: UUID,
    body: SourceCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    name_validator: deps.NameRequiredValidatorDependency,
) -> SourceResponse:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        await name_validator.validate(body)
        source = Source(
            project_id=project_id,
            name=body.name,
            description=body.description,
        )
        await uow.sources.add(source)
        await uow.source_items.add_batch(
            [
                SourceItem(
                    source_id=source.id,
                    order=item.order,
                    text=item.text,
                    image_path=item.image_path,
                    metadata=item.metadata,
                )
                for item in body.items
            ]
        )
        await uow.commit()
        return SourceResponse.from_source(source)


@router.post(
    "/projects/{project_id}/sources/pdf",
    response_model=PdfSourceUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf_source(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    ingestor: Annotated[PdfSourceIngestor, Depends(deps.get_pdf_source_ingestor)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
    file: UploadFile = File(...),
    name: str | None = Form(None),
    description: str | None = Form(None),
) -> PdfSourceUploadResponse:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        content = await file.read()
        ingested = ingestor.ingest(
            project_id=project_id,
            filename=file.filename or "upload.pdf",
            content=content,
            name=name,
            description=description,
        )
        source = ingested.source
        safe_filename = _safe_filename(file.filename or "upload.pdf")
        document_stored = storage.save(
            SaveArtifactPayloadCommand(
                bucket="source-documents",
                key=f"projects/{project_id}/sources/{source.id}/{safe_filename}",
                payload=content,
            )
        )
        document_artifact = Artifact(
            artifact_type="source.document",
            schema_version=1,
            workflow_run_id=None,
            producer_node_run_id=None,
            payload_ref=artifact_payload_ref(
                bucket=document_stored.bucket,
                key=document_stored.key,
            ),
            content_hash=document_stored.sha256,
            metadata={
                "project_id": str(project_id),
                "source_id": str(source.id),
                "filename": safe_filename,
                "content_type": file.content_type or "application/pdf",
                "byte_size": document_stored.byte_size,
                "document_uri": ingested.document_uri,
            },
        )
        artifacts: list[Artifact] = []
        for page_number, page in enumerate(ingested.pages, start=1):
            page_filename = f"{page_number:04d}.png"
            page_stored = storage.save(
                SaveArtifactPayloadCommand(
                    bucket="source-page-images",
                    key=(
                        f"projects/{project_id}/sources/{source.id}/pages/"
                        f"{page_filename}"
                    ),
                    payload=page.image_payload,
                )
            )
            payload_ref = artifact_payload_ref(
                bucket=page_stored.bucket,
                key=page_stored.key,
            )
            page.source_item.image_path = payload_ref
            artifact = Artifact(
                artifact_type="source.page_image",
                schema_version=1,
                workflow_run_id=None,
                producer_node_run_id=None,
                payload_ref=payload_ref,
                input_artifact_ids=[document_artifact.id],
                content_hash=page_stored.sha256,
                metadata={
                    "project_id": str(project_id),
                    "source_id": str(source.id),
                    "source_item_id": str(page.source_item.id),
                    "document_artifact_id": str(document_artifact.id),
                    "page_number": page_number,
                    "filename": page_filename,
                    "content_type": "image/png",
                    "byte_size": page_stored.byte_size,
                    "width": page.image_width,
                    "height": page.image_height,
                },
            )
            artifacts.append(artifact)

        sequence = ArtifactSequence(
            artifact_type="source.page_image",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in artifacts],
            index_key="page_number",
            metadata={
                "project_id": str(project_id),
                "source_id": str(source.id),
                "document_artifact_id": str(document_artifact.id),
                "page_count": len(artifacts),
            },
        )
        for artifact, page in zip(artifacts, ingested.pages, strict=True):
            page.source_item.metadata["artifact_id"] = str(artifact.id)
            page.source_item.metadata["artifact_sequence_id"] = str(sequence.id)
            page.source_item.metadata["document_artifact_id"] = str(
                document_artifact.id
            )
            page.source_item.metadata["image_payload_ref"] = artifact.payload_ref

        await uow.sources.add(ingested.source)
        await uow.source_items.add_batch(ingested.items)
        await uow.artifacts.add(document_artifact)
        for artifact in artifacts:
            await uow.artifacts.add(artifact)
        await uow.artifact_sequences.add(sequence)
        await uow.commit()
        return PdfSourceUploadResponse(
            source=SourceResponse.from_source(ingested.source),
            items=[
                SourceItemResponse.from_source_item(item)
                for item in ingested.items
            ],
            document_artifact=ArtifactResponse.from_domain(document_artifact),
            artifacts=[ArtifactResponse.from_domain(artifact) for artifact in artifacts],
            sequence=ArtifactSequenceResponse.from_domain(sequence),
        )


@router.post(
    "/projects/{project_id}/sources/images",
    response_model=ImageSourceUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image_source(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
    files: list[UploadFile] = File(...),
    name: str | None = Form(None),
    description: str | None = Form(None),
) -> ImageSourceUploadResponse:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        if not files:
            raise ValidationError("At least one page image is required")

        source = Source(
            project_id=project_id,
            name=(name or "Image source").strip(),
            description=description,
        )
        source_items: list[SourceItem] = []
        artifacts: list[Artifact] = []
        for page_number, file in enumerate(files, start=1):
            content = await file.read()
            if not content:
                raise ValidationError(f"Uploaded page image {page_number} is empty")
            if file.content_type and not file.content_type.startswith("image/"):
                raise ValidationError(
                    f"Uploaded file {file.filename or page_number!r} is not an image"
                )

            filename = _safe_filename(file.filename or f"page-{page_number}.bin")
            key = (
                f"projects/{project_id}/sources/{source.id}/pages/"
                f"{page_number:04d}-{filename}"
            )
            stored = storage.save(
                SaveArtifactPayloadCommand(
                    bucket="source-page-images",
                    key=key,
                    payload=content,
                )
            )
            payload_ref = artifact_payload_ref(bucket=stored.bucket, key=stored.key)
            source_item = SourceItem(
                source_id=source.id,
                order=page_number,
                image_path=payload_ref,
                metadata={
                    "filename": filename,
                    "content_type": file.content_type or "application/octet-stream",
                    "byte_size": stored.byte_size,
                    "sha256": stored.sha256,
                    "page_number": page_number,
                },
            )
            artifact = Artifact(
                artifact_type="source.page_image",
                schema_version=1,
                workflow_run_id=None,
                producer_node_run_id=None,
                payload_ref=payload_ref,
                content_hash=stored.sha256,
                metadata={
                    "project_id": str(project_id),
                    "source_id": str(source.id),
                    "source_item_id": str(source_item.id),
                    "page_number": page_number,
                    "filename": filename,
                    "content_type": file.content_type or "application/octet-stream",
                    "byte_size": stored.byte_size,
                },
            )
            source_items.append(source_item)
            artifacts.append(artifact)

        sequence = ArtifactSequence(
            artifact_type="source.page_image",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in artifacts],
            index_key="page_number",
            metadata={
                "project_id": str(project_id),
                "source_id": str(source.id),
                "page_count": len(artifacts),
            },
        )
        for source_item, artifact in zip(source_items, artifacts, strict=True):
            source_item.metadata["artifact_id"] = str(artifact.id)
            source_item.metadata["artifact_sequence_id"] = str(sequence.id)

        await uow.sources.add(source)
        await uow.source_items.add_batch(source_items)
        for artifact in artifacts:
            await uow.artifacts.add(artifact)
        await uow.artifact_sequences.add(sequence)
        await uow.commit()
        return ImageSourceUploadResponse(
            source=SourceResponse.from_source(source),
            items=[
                SourceItemResponse.from_source_item(item)
                for item in source_items
            ],
            artifacts=[ArtifactResponse.from_domain(artifact) for artifact in artifacts],
            sequence=ArtifactSequenceResponse.from_domain(sequence),
        )


@router.get("/projects/{project_id}/sources", response_model=list[SourceResponse])
async def list_project_sources(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[SourceResponse]:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        return [
            SourceResponse.from_source(source)
            for source in await uow.sources.list_for_project(project_id)
        ]


@router.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> SourceResponse:
    async with uow:
        source = await deps.get_source_or_404(uow, source_id)
        return SourceResponse.from_source(source)


@router.get("/sources/{source_id}/items", response_model=list[SourceItemResponse])
async def list_source_items(
    source_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[SourceItemResponse]:
    async with uow:
        await deps.get_source_or_404(uow, source_id)
        return [
            SourceItemResponse.from_source_item(item)
            for item in await uow.source_items.list_for_source(source_id)
        ]


@router.get("/sources/{source_id}/artifacts", response_model=list[ArtifactResponse])
async def list_source_artifacts(
    source_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ArtifactResponse]:
    async with uow:
        await deps.get_source_or_404(uow, source_id)
        return [
            ArtifactResponse.from_domain(artifact)
            for artifact in await uow.artifacts.list_for_source(source_id)
        ]


@router.get(
    "/sources/{source_id}/artifact-sequences",
    response_model=list[ArtifactSequenceResponse],
)
async def list_source_artifact_sequences(
    source_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ArtifactSequenceResponse]:
    async with uow:
        await deps.get_source_or_404(uow, source_id)
        return [
            ArtifactSequenceResponse.from_domain(sequence)
            for sequence in await uow.artifact_sequences.list_for_source(source_id)
        ]


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name).strip(".-")
    return cleaned or "page.bin"
