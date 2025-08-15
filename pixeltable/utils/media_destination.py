from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import UUID

import PIL.Image

from pixeltable import exceptions as excs
from pixeltable.utils.media_store import MediaStore, TempStore

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class MediaDestination:
    @classmethod
    def validate_destination(cls, col_name: Optional[str], dest: Union[str, Path, None]) -> str:
        """Convert a Column destination parameter to a URI, else raise errors.
        Args:
            col_name: Used to raise error messages
            dest: The requested destination
        Returns:
            URI of destination, or raises an error
        """
        from pixeltable.utils.s3_store import S3Store

        if dest is None or isinstance(dest, Path):
            return MediaStore.validate_destination(col_name, dest)
        if not isinstance(dest, str):
            raise excs.Error(f'Column {col_name}: "destination" must be a string or path, got {dest!r}')
        if dest.startswith('s3://'):
            dest2 = S3Store(dest).validate_uri()
            if dest2 is None:
                raise excs.Error(f'Column {col_name}: invalid S3 destination {dest!r}')
            return dest2
        # Check for "gs://" and Azure variants here
        return MediaStore.validate_destination(col_name, dest)

    def save_media_object(self, val: Any, col: Column, to_temp: bool = False) -> tuple[Optional[Path], str]:
        """Save the media object in the column to the destination or TempStore"""
        assert col.col_type.is_media_type()
        format = None
        if isinstance(val, PIL.Image.Image):
            # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
            # In that case, use WebP instead.
            format = 'webp' if val.has_transparency_data else 'jpeg'
        if to_temp:
            filepath, url = TempStore.save_media_object(val, col, format=format)
        else:
            filepath, url = MediaStore.get(col.destination).save_media_object(val, col, format=format)
        return filepath, url

    @classmethod
    def put_file(cls, col: Column, src_path: Path, can_relocate: bool) -> str:
        """Move or copy a file to the destination, returning the file's URL within the destination."""
        from pixeltable.utils.s3_store import S3Store

        destination = col.destination
        if destination is not None and destination.startswith('s3'):
            # If the destination is 's3', we need to copy the file to S3
            new_file_url = S3Store(destination).copy_local_media_file(col, src_path)
            if can_relocate:
                # File is temporary, used only once, so we can delete it after copy
                assert TempStore.contains_path(src_path)
                TempStore.delete_media_file(src_path)
            return new_file_url
        if can_relocate:
            new_file_url = MediaStore.get(destination).relocate_local_media_file(src_path, col)
        else:
            new_file_url = MediaStore.get(destination).copy_local_media_file(src_path, col)
        return new_file_url

    @classmethod
    def delete(cls, destination: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> None:
        """Delete media files in the destination URI for a given table ID"""
        from pixeltable.utils.s3_store import S3Store

        if destination is not None and destination.startswith('s3://'):
            S3Store(destination).delete(tbl_id, tbl_version)
        else:
            MediaStore.get(destination).delete(tbl_id, tbl_version)

    @classmethod
    def count(cls, uri: Optional[str], tbl_id: UUID, tbl_version: Optional[int] = None) -> int:
        """Return the count of media files in the destination URI for a given table ID"""
        if uri is not None and uri.startswith('s3://'):
            from pixeltable.utils.s3_store import S3Store

            # If the URI is an S3 URI, use the S3Store to count media files
            return S3Store(uri).count(tbl_id, tbl_version)
        # Check for "gs://" and Azure variants here
        return MediaStore.get(uri).count(tbl_id, tbl_version)
