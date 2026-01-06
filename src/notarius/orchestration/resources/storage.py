from pathlib import Path
from typing import override

import dagster as dg
from PIL import Image

from notarius.application import ports
from notarius.infrastructure.persistence import storage
from notarius.infrastructure.persistence.storage.local import LocalFileStorage


class LocalStorageResource(dg.ConfigurableResource[ports.FileStorage]):
    storage_root: str

    @override
    def create_resource(self, context: dg.InitResourceContext) -> ports.FileStorage:
        return LocalFileStorage(storage_root=Path(self.storage_root))


class ImageRepositoryResource(dg.ConfigurableResource[storage.ImageRepository]):
    storage_resource: dg.ResourceDependency[ports.FileStorage]

    @override
    def create_resource(
        self, context: dg.InitResourceContext
    ) -> storage.ImageRepository:
        return storage.ImageRepository(storage=self.storage_resource)
