from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import UUID

from pixeltable import exceptions as excs
from pixeltable.utils.media_path import MediaPath, StorageObjectAddress
from pixeltable.utils.media_store import MediaStore, TempStore
from pixeltable.utils.media_store_base import MediaStoreBase

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class HTTPStore(MediaStoreBase):
    base_url: str

    def __init__(self, soa: StorageObjectAddress):
        self.base_url = f'{soa.scheme}://{soa.account_extension}/{soa.prefix}'
        if not self.base_url.endswith('/'):
            self.base_url += '/'

    def download_media_object(self, src_path: str, dest_path: Path) -> None:
        with urllib.request.urlopen(self.base_url + src_path) as resp, open(dest_path, 'wb') as f:
            data = resp.read()
            f.write(data)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage


class MediaDestination:
    @classmethod
    def get_store(cls, dest: Optional[str], soa: Optional[StorageObjectAddress], col_name: Optional[str] = None) -> Any:
        from pixeltable.utils.gcs_store import GCSStore
        from pixeltable.utils.s3_store import S3Store

        if soa is None or soa.storage_target == 'os':
            return MediaStore.from_soa(soa)
        if soa.storage_target == 's3' and soa.scheme == 's3':
            return S3Store(soa)
        if soa.storage_target == 'r2':
            return S3Store(soa)
        if soa.storage_target == 'gs' and soa.scheme == 'gs':
            return GCSStore(soa)
        if soa.storage_target == 'http' and soa.is_http_readable:
            return HTTPStore(soa)
        if col_name is not None:
            raise excs.Error(
                f'Column {col_name}: "destination" must be a valid URI to a supported destination, got {dest!r}'
            )
        else:
            raise excs.Error(f'"destination" must be a valid URI to a supported destination, got {dest!r}')

    @classmethod
    def validate_destination(cls, col_name: Optional[str], dest: Union[str, Path, None]) -> str:
        """Convert a Column destination parameter to a URI, else raise errors.
        Args:
            col_name: Used to raise error messages
            dest: The requested destination
        Returns:
            URI of destination, or raises an error
        """
        if dest is None or isinstance(dest, Path):
            return MediaStore.validate_destination(col_name, dest)
        if not isinstance(dest, str):
            raise excs.Error(f'Column {col_name}: "destination" must be a string or path, got {dest!r}')
        soa = MediaPath.parse_media_storage_addr(dest, may_contain_object_name=False)
        if soa.storage_target == 'os':
            return MediaStore.validate_destination(col_name, soa)
        store = cls.get_store(dest, soa)
        dest2 = store.validate_uri()
        if dest2 is not None:
            return dest2
        raise excs.Error(
            f'Column {col_name}: "destination" must be a valid URI to a supported destination, got {dest!r}'
        )

    @classmethod
    def download_media_object(cls, src_uri: str, dest_path: Path) -> None:
        """Copy an object from a URL to a local Path. Thread safe.
        Raises an exception if the download fails or the scheme is not supported
        """
        soa = MediaPath.parse_media_storage_addr(src_uri, may_contain_object_name=True)
        store = cls.get_store(src_uri, soa)
        store.download_media_object(soa.object_name, dest_path)

    @classmethod
    def put_file(cls, col: Column, src_path: Path, relocate_or_delete: bool) -> str:
        """Move or copy a file to the destination, returning the file's URL within the destination.
        If relocate_or_delete is True and the file is in the TempStore, the file will be deleted after the operation.
        """
        if relocate_or_delete:
            # File is temporary, used only once, so we can delete it after copy if it can't be moved
            assert TempStore.contains_path(src_path)
        dest = col.destination
        soa = None if dest is None else MediaPath.parse_media_storage_addr(dest, may_contain_object_name=False)

        store = cls.get_store(dest, soa)
        if soa is None or soa.storage_target == 'os':
            if relocate_or_delete:
                new_file_url = store.relocate_local_media_file(src_path, col)
            else:
                new_file_url = store.copy_local_media_file(col, src_path)
            return new_file_url
        new_file_url = store.copy_local_media_file(col, src_path)
        if relocate_or_delete:
            TempStore.delete_media_file(src_path)
        return new_file_url

    @classmethod
    def delete(cls, dest: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> None:
        """Delete media files in the destination for a given table ID"""
        soa = None if dest is None else MediaPath.parse_media_storage_addr(dest, may_contain_object_name=False)
        store = cls.get_store(dest, soa)
        store.delete(tbl_id, tbl_version)

    @classmethod
    def count(cls, dest: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the count of media files in the destination for a given table ID"""
        soa = None if dest is None else MediaPath.parse_media_storage_addr(dest, may_contain_object_name=False)
        store = cls.get_store(dest, soa)
        return store.count(tbl_id, tbl_version)

    @classmethod
    def list_objects(cls, dest: Optional[str], return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found with the specified dest
        The dest specification string must not contain an object name.
        Each returned object includes the full set of prefixes.
        if return_uri is True, the full S3 URI is returned; otherwise, just the object key.
        """
        soa = None if dest is None else MediaPath.parse_media_storage_addr(dest, may_contain_object_name=False)
        store = cls.get_store(dest, soa)
        return store.list_objects(return_uri, n_max)

    @classmethod
    def list_uris(cls, source_uri: str, n_max: int = 10) -> list[str]:
        """Return a list of URIs found within the specified uri"""
        return cls.list_objects(source_uri, True, n_max)
