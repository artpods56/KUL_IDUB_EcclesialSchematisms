from dataclasses import dataclass
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, final, override

from notarius.application.ports import FileStorage
from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.schemas.data.pipeline import BaseDataset


@dataclass
class JsonExportRequest(BaseRequest):
    """Request for exporting a dataset to JSON."""

    dataset: BaseDataset[Any]
    output_dir: Path
    group_by_schematism: bool = True
    pretty_print: bool = True
    filename_prefix: str = ""


@dataclass
class JsonExportResponse(BaseResponse):
    """Response containing export results."""

    output_files: dict[str, Path]


@final
class JsonExportUseCase(BaseUseCase[JsonExportRequest, JsonExportResponse]):
    """Export dataset to JSON file(s)."""

    def __init__(self, storage: FileStorage):
        self.storage = storage

    @override
    def execute(self, request: JsonExportRequest) -> JsonExportResponse:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{request.filename_prefix}_" if request.filename_prefix else ""
        output_files: dict[str, Path] = {}

        groups = (
            request.dataset.group_by_schematism()
            if request.group_by_schematism
            else [("all", request.dataset)]
        )

        for name, group_dataset in groups:
            records = [item.model_dump() for item in group_dataset.items]
            content = {
                "generated_at": timestamp,
                "total_records": len(records),
                "records": records,
            }
            if request.group_by_schematism:
                content["schematism_name"] = name

            filename = f"{timestamp}_{prefix}{name}.json"
            file_path = request.output_dir / filename

            json_str = json.dumps(
                content,
                ensure_ascii=False,
                indent=2 if request.pretty_print else None,
                default=str,
            )
            saved_path = self.storage.save(io.BytesIO(json_str.encode("utf-8")), file_path)

            output_files[name] = self.storage.storage_root / saved_path

        return JsonExportResponse(output_files=output_files)
