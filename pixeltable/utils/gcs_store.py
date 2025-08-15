from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from pixeltable.utils.gcs import GCSClientContainer
from pixeltable.utils.media_path import MediaPath

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class GCSStore:
    """Class to handle Google Cloud Storage operations."""

    # URI of the GCS bucket in the format gs://bucket_name/prefix/
    # Always ends with a slash
    __base_uri: str

    # bucket name extracted from the URI
    __bucket_name: str

    # prefix path within the bucket, either empty or ending with a slash
    __prefix_name: str

    def __init__(self, uri: str):
        assert uri.startswith('gs'), "URI must start with 'gs'"
        parsed_uri = urllib.parse.urlparse(uri)
        assert parsed_uri.scheme == 'gs', 'URI must be a GCS URI'
        self.__base_uri = uri.rstrip('/')
        self.__bucket_name = parsed_uri.netloc
        self.__prefix_name = parsed_uri.path.lstrip('/').rstrip('/')
        if len(self.__prefix_name) > 0:
            self.__prefix_name += '/'
        self.__base_uri += '/'
        if 0:
            print(
                f'Initialized GCSStore with base URI: {self.__base_uri},',
                f'bucket: {self.__bucket_name}, prefix: {self.__prefix_name}',
            )

    @classmethod
    def client(cls, for_write: bool = False) -> Any:
        """Return the GCS client."""
        return GCSClientContainer.get().get_client(for_write=for_write)

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
            str: The base URI if the GCS bucket exists and is accessible, None otherwise.
        """
        from google.api_core.exceptions import GoogleAPIError
        from google.cloud.exceptions import Forbidden, NotFound

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            bucket.reload()  # This will raise an exception if the bucket doesn't exist
            return self.__base_uri
        except (NotFound, Forbidden, GoogleAPIError) as e:
            GCSClientContainer.handle_gcs_error(e, self.bucket_name, 'validate bucket')
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

    def copy_local_media_file(self, col: Column, src_path: Path) -> str:
        """Copy a local file, and return its new URL"""
        from google.api_core.exceptions import GoogleAPIError

        new_file_uri = self._prepare_media_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        blob_name = parsed.path.lstrip('/')

        try:
            client = self.client(for_write=True)
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(src_path))
            _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
            return new_file_uri
        except GoogleAPIError as e:
            GCSClientContainer.handle_gcs_error(e, self.bucket_name, f'upload file {src_path}')
            raise

    def download_media_object(self, src_path: str, dest_path: Path) -> None:
        """Copies an object to a local file. Thread safe"""
        from google.api_core.exceptions import GoogleAPIError

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.prefix + src_path)
            blob.download_to_filename(str(dest_path))
        except GoogleAPIError as e:
            GCSClientContainer.handle_gcs_error(e, self.bucket_name, f'download file {src_path}')
            raise

    def _get_filtered_objects(self, bucket: Any, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> Iterator:
        """Private method to get filtered objects for a table, optionally filtered by version.

        Args:
            tbl_id: Table UUID to filter by
            tbl_version: Optional table version to filter by

        Returns:
            Tuple of (iterator over GCS objects matching the criteria, bucket object)
        """
        table_prefix = MediaPath.media_table_prefix(tbl_id)
        prefix = f'{self.prefix}{table_prefix}/'

        if tbl_version is None:
            # Return all blobs with the table prefix
            blob_iterator = bucket.list_blobs(prefix=prefix)
        else:
            # Filter by both table_id and table_version using the MediaPath pattern
            # Pattern: tbl_id_col_id_version_uuid
            version_pattern = re.compile(rf'{re.escape(table_prefix)}_\d+_{re.escape(str(tbl_version))}_[0-9a-fA-F]+.*')
            # Return filtered collection - this still uses lazy loading
            all_blobs = bucket.list_blobs(prefix=prefix)
            blob_iterator = (blob for blob in all_blobs if version_pattern.match(blob.name.split('/')[-1]))

        return blob_iterator

    def count(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> int:
        """Count the number of files belonging to tbl_id. If tbl_version is not None,
        count only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to count objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects matching the criteria
        """
        from google.api_core.exceptions import GoogleAPIError

        assert tbl_id is not None

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)

            blob_iterator = self._get_filtered_objects(bucket, tbl_id, tbl_version)

            return sum(1 for _ in blob_iterator)

        except GoogleAPIError as e:
            GCSClientContainer.handle_gcs_error(e, self.bucket_name, f'setup iterator {self.prefix}')
            raise

    def delete(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> int:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to delete objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects deleted
        """
        from google.api_core.exceptions import GoogleAPIError

        assert tbl_id is not None

        total_deleted = 0

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            blob_iterator = self._get_filtered_objects(bucket, tbl_id, tbl_version)

            # Collect blob names for batch deletion
            blobs_to_delete = []

            for blob in blob_iterator:
                blobs_to_delete.append(blob)

                # Process in batches for efficiency
                if len(blobs_to_delete) >= 100:
                    with client.batch():
                        for b in blobs_to_delete:
                            b.delete()
                    total_deleted += len(blobs_to_delete)
                    blobs_to_delete = []

            # Delete any remaining blobs in the final batch
            if len(blobs_to_delete) > 0:
                with client.batch():
                    for b in blobs_to_delete:
                        b.delete()
                total_deleted += len(blobs_to_delete)

            print(f"Deleted {total_deleted} objects from bucket '{self.bucket_name}'.")
            return total_deleted

        except GoogleAPIError as e:
            GCSClientContainer.handle_gcs_error(e, self.bucket_name, f'deleting with {self.prefix}')
            raise
