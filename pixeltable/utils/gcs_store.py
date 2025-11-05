from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage  # type: ignore[attr-defined]
from google.cloud.exceptions import Forbidden, NotFound
from google.cloud.storage.client import Client  # type: ignore[import-untyped]

from pixeltable import env, exceptions as excs
from pixeltable.utils.object_stores import ObjectPath, ObjectStoreBase, StorageObjectAddress, StorageTarget

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


@env.register_client('gcs_store')
def _() -> 'Client':
    """Create and return a GCS client, using default credentials if available,
    otherwise creating an anonymous client for public buckets.
    """
    try:
        # Create a client with default credentials
        # Note that if the default credentials have expired, gcloud will still create a client,
        # which will report the expiry error when it is used.
        # To create and use an anonymous client, expired credentials must be removed.
        # For application default credentials, delete the file in ~/.config/gcloud/, or
        #   gcloud auth application-default revoke
        # OR
        # For service account keys, you must delete the downloaded key file.
        client = storage.Client()
        return client
    except Exception:
        # If no credentials are found, create an anonymous client which can be used for public buckets.
        client = storage.Client.create_anonymous_client()
        return client


class GCSStore(ObjectStoreBase):
    """Class to handle Google Cloud Storage operations."""

    # URI of the GCS bucket in the format gs://bucket_name/prefix/
    # Always ends with a slash
    __base_uri: str

    # bucket name extracted from the URI
    __bucket_name: str

    # prefix path within the bucket, either empty or ending with a slash
    __prefix_name: str

    # The parsed form of the given destination address
    soa: StorageObjectAddress

    def __init__(self, soa: StorageObjectAddress):
        assert soa.storage_target == StorageTarget.GCS_STORE, f'Expected storage_target "gs", got {soa.storage_target}'
        self.soa = soa
        self.__base_uri = soa.prefix_free_uri + soa.prefix
        self.__bucket_name = soa.container
        self.__prefix_name = soa.prefix

    @classmethod
    def client(cls) -> 'Client':
        """Return the GCS client."""
        return env.Env.get().get_client('gcs_store')

    @property
    def bucket_name(self) -> str:
        """Return the bucket name from the base URI."""
        return self.__bucket_name

    @property
    def prefix(self) -> str:
        """Return the prefix from the base URI."""
        return self.__prefix_name

    def validate(self, error_col_name: str) -> str | None:
        """
        Checks if the URI exists.

        Returns:
            str: The base URI if the GCS bucket exists and is accessible, None otherwise.
        """
        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(max_results=1)
            # This will raise an exception if the destination doesn't exist or cannot be listed
            _ = list(blobs)  # Force evaluation to check access
            return self.__base_uri
        except (NotFound, Forbidden, GoogleAPIError) as e:
            self.handle_gcs_error(e, self.bucket_name, f'validate bucket {error_col_name}')
        return None

    def _prepare_uri_raw(self, tbl_id: uuid.UUID, col_id: int, tbl_version: int, ext: str | None = None) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        prefix, filename = ObjectPath.create_prefix_raw(tbl_id, col_id, tbl_version, ext)
        parent = f'{self.__base_uri}{prefix}'
        return f'{parent}/{filename}'

    def _prepare_uri(self, col: Column, ext: str | None = None) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        assert col.get_tbl() is not None, 'Column must be associated with a table'
        return self._prepare_uri_raw(col.get_tbl().id, col.id, col.get_tbl().version, ext=ext)

    def copy_local_file(self, col: Column, src_path: Path) -> str:
        """Copy a local file, and return its new URL"""
        new_file_uri = self._prepare_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        blob_name = parsed.path.lstrip('/')

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(src_path))
            _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
            return new_file_uri
        except GoogleAPIError as e:
            self.handle_gcs_error(e, self.bucket_name, f'upload file {src_path}')
            raise

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        """Copies an object to a local file. Thread safe"""
        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.prefix + src_path)
            blob.download_to_filename(str(dest_path))
        except GoogleAPIError as e:
            self.handle_gcs_error(e, self.bucket_name, f'download file {src_path}')
            raise

    def _get_filtered_objects(self, bucket: Any, tbl_id: uuid.UUID, tbl_version: int | None = None) -> Iterator:
        """Private method to get filtered objects for a table, optionally filtered by version.

        Args:
            tbl_id: Table UUID to filter by
            tbl_version: Optional table version to filter by

        Returns:
            Tuple of (iterator over GCS objects matching the criteria, bucket object)
        """
        table_prefix = ObjectPath.table_prefix(tbl_id)
        prefix = f'{self.prefix}{table_prefix}/'

        if tbl_version is None:
            # Return all blobs with the table prefix
            blob_iterator = bucket.list_blobs(prefix=prefix)
        else:
            # Filter by both table_id and table_version using the ObjectPath pattern
            # Pattern: tbl_id_col_id_version_uuid
            version_pattern = re.compile(rf'{re.escape(table_prefix)}_\d+_{re.escape(str(tbl_version))}_[0-9a-fA-F]+.*')
            # Return filtered collection - this still uses lazy loading
            all_blobs = bucket.list_blobs(prefix=prefix)
            blob_iterator = (blob for blob in all_blobs if version_pattern.match(blob.name.split('/')[-1]))

        return blob_iterator

    def count(self, tbl_id: uuid.UUID, tbl_version: int | None = None) -> int:
        """Count the number of files belonging to tbl_id. If tbl_version is not None,
        count only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to count objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects matching the criteria
        """
        assert tbl_id is not None

        try:
            client = self.client()
            bucket = client.bucket(self.bucket_name)

            blob_iterator = self._get_filtered_objects(bucket, tbl_id, tbl_version)

            return sum(1 for _ in blob_iterator)

        except GoogleAPIError as e:
            self.handle_gcs_error(e, self.bucket_name, f'setup iterator {self.prefix}')
            raise

    def delete(self, tbl_id: uuid.UUID, tbl_version: int | None = None) -> int:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to delete objects for
            tbl_version: Optional table version to filter by

        Returns:
            Number of objects deleted
        """
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

            return total_deleted

        except GoogleAPIError as e:
            self.handle_gcs_error(e, self.bucket_name, f'deleting with {self.prefix}')
            raise

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found in the specified destination bucket.
        Each returned object includes the full set of prefixes.
        if return_uri is True, full URI's are returned; otherwise, just the object keys.
        """
        p = self.soa.prefix_free_uri if return_uri else ''
        gcs_client = self.client()
        r: list[str] = []

        try:
            bucket = gcs_client.bucket(self.bucket_name)
            # List blobs with the given prefix, limiting to n_max
            blobs = bucket.list_blobs(prefix=self.prefix, max_results=n_max)

            for blob in blobs:
                r.append(f'{p}{blob.name}')
                if len(r) >= n_max:
                    break

        except GoogleAPIError as e:
            self.handle_gcs_error(e, self.bucket_name, f'list objects from {self.prefix}')
        return r

    @classmethod
    def handle_gcs_error(cls, e: Exception, bucket_name: str, operation: str = '', *, ignore_404: bool = False) -> None:
        """Handle GCS-specific errors and convert them to appropriate exceptions"""
        if isinstance(e, NotFound):
            if ignore_404:
                return
            raise excs.Error(f'Bucket or object {bucket_name} not found during {operation}: {str(e)!r}')
        elif isinstance(e, Forbidden):
            raise excs.Error(f'Access denied to bucket {bucket_name} during {operation}: {str(e)!r}')
        elif isinstance(e, GoogleAPIError):
            # Handle other Google API errors
            error_message = str(e)
            if 'Precondition' in error_message:
                raise excs.Error(f'Precondition failed for bucket {bucket_name} during {operation}: {error_message}')
            else:
                raise excs.Error(f'Error during {operation} in bucket {bucket_name}: {error_message}')
        else:
            # Generic error handling
            raise excs.Error(f'Unexpected error during {operation} in bucket {bucket_name}: {str(e)!r}')
