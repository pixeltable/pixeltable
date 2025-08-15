from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from pixeltable.utils.media_path import MediaPath
from pixeltable.utils.s3 import S3ClientContainer

if TYPE_CHECKING:
    from botocore.exceptions import ClientError

    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class S3Store:
    """Class to handle S3 storage operations."""

    # URI of the S3 bucket in the format s3://bucket_name/prefix/
    # Always ends with a slash
    __base_uri: str

    # bucket name extracted from the URI
    __bucket_name: str

    # prefix path within the bucket, either empty or ending with a slash
    __prefix_name: str

    def __init__(self, uri: str):
        assert uri.startswith('s3'), "URI must start with 's3'"
        parsed_uri = urllib.parse.urlparse(uri)
        assert parsed_uri.scheme == 's3', 'URI must be an S3 URI'
        self.__base_uri = uri.rstrip('/')
        self.__bucket_name = parsed_uri.netloc
        self.__prefix_name = parsed_uri.path.lstrip('/').rstrip('/')
        if len(self.__prefix_name) > 0:
            self.__prefix_name += '/'
        self.__base_uri += '/'
        if 0:
            print(
                f'Initialized S3Store with base URI: {self.__base_uri},',
                f'bucket: {self.__bucket_name}, prefix: {self.__prefix_name}',
            )

    def client(self, for_write: bool = False) -> Any:
        """Return the S3 client."""
        return S3ClientContainer.get().get_client(for_write=for_write)

    @property
    def bucket_name(self) -> str:
        """Return the bucket name from the base URI."""
        return self.__bucket_name

    @property
    def prefix(self) -> str:
        """Return the prefix from the base URI."""
        return self.__prefix_name

    def validate_uri(self) -> Optional[str]:
        """
        Checks if the URI exists.

        Returns:
            bool: True if the S3 URI exists and is accessible, False otherwise.
        """
        try:
            self.client().head_bucket(Bucket=self.bucket_name)
            return self.__base_uri
        except ClientError as e:
            S3ClientContainer.handle_s3_error(e, self.bucket_name, 'validate bucket')
        return None

    def _prepare_media_uri_raw(
        self, tbl_id: uuid.UUID, col_id: int, tbl_version: int, ext: Optional[str] = None
    ) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        prefix, filename = MediaPath.media_prefix_file_raw(tbl_id, col_id, tbl_version, ext)
        parent = f'{self.__base_uri}{prefix}'
        return f'{parent}/{filename}'

    def _prepare_media_uri(self, col: Column, ext: Optional[str] = None) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        assert col.tbl is not None, 'Column must be associated with a table'
        return self._prepare_media_uri_raw(col.tbl.id, col.id, col.tbl.version, ext=ext)

    def download_media_object(self, src_path: str, dest_path: Path) -> None:
        """Copies an object to a local file. Thread safe."""
        try:
            client = S3ClientContainer.get().get_client(for_write=False)
            client.download_file(self.bucket_name, self.prefix + src_path, str(dest_path))
        except ClientError as e:
            S3ClientContainer.handle_s3_error(e, self.bucket_name, f'download file {src_path}')
            raise

    def copy_local_media_file(self, col: Column, src_path: Path) -> str:
        """Copy a local file, and return its new URL"""
        new_file_uri = self._prepare_media_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        try:
            self.client(for_write=True).upload_file(
                Filename=str(src_path), Bucket=parsed.netloc, Key=parsed.path.lstrip('/')
            )
            _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
            return new_file_uri
        except ClientError as e:
            S3ClientContainer.handle_s3_error(e, self.bucket_name, f'setup iterator {self.prefix}')
            raise

    def _get_filtered_objects(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> tuple[Iterator, Any]:
        """Private method to get filtered objects for a table, optionally filtered by version.

        Args:
            tbl_id: Table UUID to filter by
            tbl_version: Optional table version to filter by

        Returns:
            Tuple of (iterator over S3 objects matching the criteria, bucket object)
        """
        # Use MediaPath to construct the prefix for this table

        table_prefix = MediaPath.media_table_prefix(tbl_id)
        prefix = f'{self.prefix}{table_prefix}/'

        try:
            # Use S3 resource interface for filtering
            s3_resource = S3ClientContainer.get().get_resource()
            bucket = s3_resource.Bucket(self.bucket_name)

            if tbl_version is None:
                # Return all objects with the table prefix
                object_iterator = bucket.objects.filter(Prefix=prefix)
            else:
                # Filter by both table_id and table_version using the MediaPath pattern
                # Pattern: tbl_id_col_id_version_uuid
                version_pattern = re.compile(
                    rf'{re.escape(table_prefix)}_\d+_{re.escape(str(tbl_version))}_[0-9a-fA-F]+.*'
                )
                # Return filtered collection - this still uses lazy loading
                object_iterator = (
                    obj for obj in bucket.objects.filter(Prefix=prefix) if version_pattern.match(obj.key.split('/')[-1])
                )

            return object_iterator, bucket

        except ClientError as e:
            S3ClientContainer.handle_s3_error(e, self.bucket_name, f'setup iterator {self.prefix}')
            raise

    def count(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> int:
        """Count the number of files belonging to tbl_id. If tbl_version is not None,
        count only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to count objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects matching the criteria
        """
        assert tbl_id is not None

        object_iterator, _ = self._get_filtered_objects(tbl_id, tbl_version)

        return sum(1 for _ in object_iterator)

    def delete(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> int:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to delete objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects deleted
        """
        assert tbl_id is not None

        # Use shared method to get filtered objects and bucket
        object_iterator, bucket = self._get_filtered_objects(tbl_id, tbl_version)

        total_deleted = 0

        try:
            objects_to_delete = []

            # Process objects in batches as we iterate (memory efficient)
            for obj in object_iterator:
                objects_to_delete.append({'Key': obj.key})

                # Delete in batches of 1000 (S3 limit)
                if len(objects_to_delete) >= 1000:
                    bucket.delete_objects(Delete={'Objects': objects_to_delete, 'Quiet': True})
                    total_deleted += len(objects_to_delete)
                    objects_to_delete = []

            # Delete any remaining objects in the final batch
            if len(objects_to_delete) > 0:
                bucket.delete_objects(Delete={'Objects': objects_to_delete, 'Quiet': True})
                total_deleted += len(objects_to_delete)

            print(f"Deleted {total_deleted} objects from bucket '{self.bucket_name}'.")
            return total_deleted

        except ClientError as e:
            S3ClientContainer.handle_s3_error(e, self.bucket_name, f'deleting with {self.prefix}')
            raise
