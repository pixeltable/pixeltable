from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.utils.media_path import ObjectPath, StorageObjectAddress, StorageTarget

# from pixeltable.utils.media_store import LocalStore, TempStore
from pixeltable.utils.media_store_base import ObjectStoreBase

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class ObjectOps:
    @classmethod
    def get_store(cls, dest: Optional[str], may_contain_object_name: bool, col_name: Optional[str] = None) -> Any:
        from pixeltable.env import Env
        from pixeltable.utils.gcs_store import GCSStore
        from pixeltable.utils.media_store import LocalStore
        from pixeltable.utils.s3_store import S3Store

        soa = (
            Env.get().object_soa
            if dest is None
            else ObjectPath.parse_object_storage_addr(dest, may_contain_object_name=may_contain_object_name)
        )
        if soa.storage_target == StorageTarget.OS:
            return LocalStore.from_soa(soa)
        if soa.storage_target == StorageTarget.S3 and soa.scheme == 's3':
            return S3Store(soa)
        if soa.storage_target == StorageTarget.R2:
            return S3Store(soa)
        if soa.storage_target == StorageTarget.GS and soa.scheme == 'gs':
            return GCSStore(soa)
        if soa.storage_target == StorageTarget.HTTP and soa.is_http_readable:
            return HTTPStore(soa)
        error_col_name = f'Column {col_name!r}: ' if col_name is not None else ''
        raise excs.Error(
            f'{error_col_name}`destination` must be a valid reference to a supported destination, got {dest!r}'
        )

    @classmethod
    def validate_destination(cls, dest: str | Path | None, col_name: Optional[str]) -> str:
        """Convert a Column destination parameter to a URI, else raise errors.
        Args:
            dest: The requested destination
            col_name: Used to raise error messages
        Returns:
            URI of destination, or raises an error
        """
        error_col_name = f'Column {col_name!r}: ' if col_name is not None else ''

        # General checks on any destination
        if isinstance(dest, Path):
            dest = str(dest)
        if dest is not None and not isinstance(dest, str):
            raise excs.Error(f'{error_col_name}`destination` must be a string or path, got {dest!r}')

        # Specific checks for storage backends
        store = cls.get_store(dest, False, col_name)
        dest2 = store.validate(error_col_name)
        if dest2 is None:
            raise excs.Error(f'{error_col_name}`destination` must be a supported destination, got {dest!r}')
        return dest2

    @classmethod
    def copy_object_to_local_file(cls, src_uri: str, dest_path: Path) -> None:
        """Copy an object from a URL to a local Path. Thread safe.
        Raises an exception if the download fails or the scheme is not supported
        """
        soa = ObjectPath.parse_object_storage_addr(src_uri, may_contain_object_name=True)
        store = cls.get_store(src_uri, True)
        store.copy_object_to_local_file(soa.object_name, dest_path)

    @classmethod
    def put_file(cls, col: Column, src_path: Path, relocate_or_delete: bool) -> str:
        """Move or copy a file to the destination, returning the file's URL within the destination.
        If relocate_or_delete is True and the file is in the TempStore, the file will be deleted after the operation.
        """
        from pixeltable.utils.media_store import TempStore

        if relocate_or_delete:
            # File is temporary, used only once, so we can delete it after copy if it can't be moved
            assert TempStore.contains_path(src_path)
        dest = col.destination
        store = cls.get_store(dest, False, col.name)
        # Attempt to move
        if relocate_or_delete:
            moved_file_url = store.move_local_file(col, src_path)
            if moved_file_url is not None:
                return moved_file_url
        new_file_url = store.copy_local_file(col, src_path)
        if relocate_or_delete:
            TempStore.delete_media_file(src_path)
        return new_file_url

    @classmethod
    def move_local_file(cls, col: Column, src_path: Path) -> str:
        """Move a file to the destination specified by the Column, returning the file's URL within the destination."""
        store = cls.get_store(col.destination, False, col.name)
        return store.move_local_file(col, src_path)

    @classmethod
    def copy_local_file(cls, col: Column, src_path: Path) -> str:
        """Copy a file to the destination specified by the Column, returning the file's URL within the destination."""
        store = cls.get_store(col.destination, False, col.name)
        return store.copy_local_file(col, src_path)

    @classmethod
    def delete(cls, dest: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> Optional[int]:
        """Delete objects in the destination for a given table ID, table version.
        Returns:
            Number of objects deleted or None
        """
        store = cls.get_store(dest, False)
        return store.delete(tbl_id, tbl_version)

    @classmethod
    def count(cls, dest: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the count of objects in the destination for a given table ID"""
        store = cls.get_store(dest, False)
        return store.count(tbl_id, tbl_version)

    @classmethod
    def list_objects(cls, dest: Optional[str], return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found in the specified destination bucket.
        The dest specification string must not contain an object name.
        Each returned object includes the full set of prefixes.
        if return_uri is True, full URI's are returned; otherwise, just the object keys.
        """
        store = cls.get_store(dest, False)
        return store.list_objects(return_uri, n_max)

    @classmethod
    def list_uris(cls, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of URIs found within the specified uri"""
        return cls.list_objects(source_uri, True, n_max)


class HTTPStore(ObjectStoreBase):
    base_url: str

    def __init__(self, soa: StorageObjectAddress):
        self.base_url = f'{soa.scheme}://{soa.account_extension}/{soa.prefix}'
        if not self.base_url.endswith('/'):
            self.base_url += '/'

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        with urllib.request.urlopen(self.base_url + src_path) as resp, open(dest_path, 'wb') as f:
            data = resp.read()
            f.write(data)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage
