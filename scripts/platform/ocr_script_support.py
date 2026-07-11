import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont


JsonObject = dict[str, object]
PAGE_IMAGE_ARTIFACT_TYPE = "source.page_image"
PAGE_IMAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PageSequenceSelection:
    project_id: object | None
    source_id: object | None
    source_kind: str
    source_page_count: int
    sequence: JsonObject


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request_json(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
        query: JsonObject | None = None,
    ) -> object:
        data = None
        headers = {"Accept": "application/json"}
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                decoded = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"{method} {url} failed before an HTTP response: {exc.reason}"
            ) from exc

        return json.loads(decoded)

    def request_object(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
        query: JsonObject | None = None,
    ) -> JsonObject:
        parsed = self.request_json(method, path, payload, query)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{method} {path} returned a non-object JSON response")
        return parsed

    def request_multipart(
        self,
        path: str,
        fields: JsonObject,
        files: list[tuple[str, Path, str]],
    ) -> JsonObject:
        boundary = f"----notarius-{uuid4().hex}"
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                )
            )
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        for field_name, file_path, content_type in files:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                (
                    "Content-Disposition: form-data; "
                    f'name="{field_name}"; filename="{file_path.name}"\r\n'
                ).encode("utf-8")
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body.extend(file_path.read_bytes())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        request = Request(
            f"{self.base_url}{path}",
            data=bytes(body),
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                decoded = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"POST {path} failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"POST {self.base_url}{path} failed before an HTTP response: "
                f"{exc.reason}"
            ) from exc

        parsed = json.loads(decoded)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"POST {path} returned a non-object JSON response")
        return parsed


def api_base_url_default() -> str:
    return os.getenv("NOTARIUS_API_BASE_URL", "http://127.0.0.1:8000")


def add_page_sequence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-id",
        help=(
            "Existing project ID for uploading --image/--pdf sources. "
            "If omitted, upload mode creates a project."
        ),
    )
    parser.add_argument(
        "--source-id",
        help=(
            "Existing source ID whose source.page_image sequence should be used. "
            "Cannot be combined with --image, --pdf, or --sequence-id."
        ),
    )
    parser.add_argument(
        "--sequence-id",
        help=(
            "Existing source.page_image artifact sequence ID to use directly. "
            "Cannot be combined with --image, --pdf, or --source-id."
        ),
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Page image path. Repeat for multiple ordered pages.",
    )
    parser.add_argument(
        "--pdf",
        help="Scanned PDF path. Mutually exclusive with --image.",
    )
    parser.add_argument("--project-name", default="Script OCR project")
    parser.add_argument("--source-name", default="Script OCR source")


def validate_page_sequence_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.pdf is not None and args.image:
        parser.error("--pdf cannot be combined with --image")
    if args.source_id is not None and (args.pdf is not None or args.image):
        parser.error("--source-id cannot be combined with --image or --pdf")
    if args.sequence_id is not None and (args.pdf is not None or args.image):
        parser.error("--sequence-id cannot be combined with --image or --pdf")
    if args.source_id is not None and args.sequence_id is not None:
        parser.error("--source-id cannot be combined with --sequence-id")


def resolve_page_sequence(
    client: ApiClient,
    args: argparse.Namespace,
    temporary_directory: Path,
) -> PageSequenceSelection:
    if args.source_id is not None:
        return _resolve_existing_source_sequence(client, args)
    if args.sequence_id is not None:
        return _resolve_existing_sequence(client, args)

    project = _upload_project(client, args)
    if args.pdf is None:
        image_paths = (
            [Path(path) for path in args.image]
            if args.image
            else _create_sample_page_images(temporary_directory)
        )
        _require_files(image_paths)
        uploaded = client.request_multipart(
            f"/v1/projects/{project['id']}/sources/images",
            fields={"name": args.source_name},
            files=[
                ("files", image_path, _content_type(image_path))
                for image_path in image_paths
            ],
        )
        source_kind = "images"
    else:
        pdf_path = Path(args.pdf)
        _require_files([pdf_path])
        uploaded = client.request_multipart(
            f"/v1/projects/{project['id']}/sources/pdf",
            fields={"name": args.source_name},
            files=[("file", pdf_path, "application/pdf")],
        )
        source_kind = "pdf"

    sequence = object_field(uploaded, "sequence")
    require_page_image_sequence(sequence)
    source = object_field(uploaded, "source")
    return PageSequenceSelection(
        project_id=project["id"],
        source_id=source["id"],
        source_kind=source_kind,
        source_page_count=sequence_page_count(sequence),
        sequence=sequence,
    )


def json_object_config(raw_config: str, option_name: str) -> JsonObject:
    parsed = json.loads(raw_config)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{option_name} must decode to a JSON object")
    return parsed


def engine_config(raw_config: str) -> JsonObject:
    return json_object_config(raw_config, "--engine-config-json")


def object_field(value: object, field_name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object while reading field {field_name}")
    field_value = value[field_name]
    if not isinstance(field_value, dict):
        raise RuntimeError(f"Field {field_name} is not an object")
    return field_value


def object_list_field(value: object, field_name: str) -> list[JsonObject]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object while reading field {field_name}")
    field_value = value[field_name]
    return object_list(field_value, f"field {field_name}")


def object_list(value: object, description: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise RuntimeError(f"{description} is not a list")
    objects: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError(f"{description} contains a non-object item")
        objects.append(item)
    return objects


def require_page_image_sequence(sequence: JsonObject) -> None:
    artifact_type = sequence.get("artifact_type")
    schema_version = sequence.get("schema_version")
    if artifact_type != PAGE_IMAGE_ARTIFACT_TYPE:
        raise RuntimeError(
            f"Expected sequence artifact_type {PAGE_IMAGE_ARTIFACT_TYPE!r}, "
            f"got {artifact_type!r}"
        )
    if schema_version != PAGE_IMAGE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Expected sequence schema_version {PAGE_IMAGE_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )


def sequence_page_count(sequence: JsonObject) -> int:
    return len(object_list_field(sequence, "item_refs"))


def image_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def metadata_string(value: JsonObject, field_name: str) -> str | None:
    metadata = object_field(value, "metadata")
    field_value = metadata.get(field_name)
    return field_value if isinstance(field_value, str) else None


def _upload_project(client: ApiClient, args: argparse.Namespace) -> JsonObject:
    if args.project_id is not None:
        return client.request_object("GET", f"/v1/projects/{args.project_id}")
    return client.request_object(
        "POST",
        "/v1/projects",
        {"name": args.project_name},
    )


def _resolve_existing_source_sequence(
    client: ApiClient,
    args: argparse.Namespace,
) -> PageSequenceSelection:
    source = client.request_object("GET", f"/v1/sources/{args.source_id}")
    project_id = source["project_id"]
    if args.project_id is not None and args.project_id != project_id:
        raise RuntimeError(
            f"Source {args.source_id} belongs to project {project_id}, "
            f"not {args.project_id}"
        )

    response = client.request_json(
        "GET",
        f"/v1/sources/{args.source_id}/artifact-sequences",
    )
    sequence = _select_page_sequence(
        object_list(response, "source artifact sequences"),
        str(args.source_id),
    )
    return PageSequenceSelection(
        project_id=project_id,
        source_id=source["id"],
        source_kind="existing_source",
        source_page_count=sequence_page_count(sequence),
        sequence=sequence,
    )


def _resolve_existing_sequence(
    client: ApiClient,
    args: argparse.Namespace,
) -> PageSequenceSelection:
    sequence = client.request_object("GET", f"/v1/artifact-sequences/{args.sequence_id}")
    require_page_image_sequence(sequence)
    project_id = metadata_string(sequence, "project_id")
    if args.project_id is not None:
        if project_id is None:
            raise RuntimeError(
                f"Sequence {args.sequence_id} does not carry metadata.project_id"
            )
        if args.project_id != project_id:
            raise RuntimeError(
                f"Sequence {args.sequence_id} belongs to project {project_id}, "
                f"not {args.project_id}"
            )
    return PageSequenceSelection(
        project_id=project_id,
        source_id=metadata_string(sequence, "source_id"),
        source_kind="existing_sequence",
        source_page_count=sequence_page_count(sequence),
        sequence=sequence,
    )


def _select_page_sequence(sequences: list[JsonObject], source_id: str) -> JsonObject:
    candidates = []
    for sequence in sequences:
        if (
            sequence.get("artifact_type") == PAGE_IMAGE_ARTIFACT_TYPE
            and sequence.get("schema_version") == PAGE_IMAGE_SCHEMA_VERSION
        ):
            candidates.append(sequence)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"Source {source_id} has no source.page_image sequence")
    raise RuntimeError(
        f"Source {source_id} has {len(candidates)} source.page_image sequences; "
        "pass --sequence-id"
    )


def _create_sample_page_images(directory: Path) -> list[Path]:
    pages = [
        ("page-1.png", "Alpha parish register page"),
        ("page-2.png", "Beta deanery summary page"),
    ]
    paths: list[Path] = []
    for filename, text in pages:
        path = directory / filename
        image = Image.new("RGB", (1400, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.text((72, 130), text, fill="black", font=_sample_font())
        image.save(path, format="PNG")
        paths.append(path)
    return paths


def _sample_font() -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), 64)
    try:
        return ImageFont.load_default(size=48)
    except TypeError:
        return ImageFont.load_default()


def _require_files(paths: list[Path]) -> None:
    if not paths:
        raise RuntimeError("At least one image is required")
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Image path does not exist: {path}")


def _content_type(path: Path) -> str:
    return image_content_type(path)
