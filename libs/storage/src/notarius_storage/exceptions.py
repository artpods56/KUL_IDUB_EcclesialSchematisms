class StorageError(Exception):
    pass


class FileUploadError(StorageError):
    pass


class FileDownloadError(StorageError):
    pass

