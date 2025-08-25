from __future__ import annotations

import logging
import os
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from botocore.exceptions import ClientError

from pixeltable import exceptions as excs
from pixeltable.utils.client_container import ClientContainer
from pixeltable.utils.media_path import MediaPath, StorageObjectAddress
from pixeltable.utils.media_store_base import MediaStoreBase

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


class S3Store(MediaStoreBase):
    """Class to handle S3 storage operations."""

    # URI of the S3 bucket in the format s3://bucket_name/prefix/
    # Always ends with a slash
    __base_uri: str

    # bucket name extracted from the URI
    __bucket_name: str

    # prefix path within the bucket, either empty or ending with a slash
    __prefix_name: str

    a_key: str
    s_key: str
    acct: str
    soa: StorageObjectAddress

    def __init__(self, soa: StorageObjectAddress):
        """Initialize the S3Store with a StorageObjectAddress."""
        self.soa = soa
        self.__bucket_name = self.soa.container
        self.__prefix_name = self.soa.prefix
        if soa.storage_target == 'r2':
            self.a_key = os.environ['R2_ACCESS_KEY']
            self.s_key = os.environ['R2_SECRET_KEY']
            self.acct = self.soa.account
            self.acct = self.soa.container
            self.__base_uri = self.soa.prefix_free_uri + self.soa.prefix
            print(self.a_key, self.s_key, self.acct)
        else:
            self.a_key = ''
            assert soa.storage_target == 's3', f'Expected storage_target "s3", got {soa.storage_target}'
            self.__base_uri = soa.prefix_free_uri + soa.prefix
        if 1:
            self.show()

    def show(self) -> None:
        print(
            f'S3Store with: base URI: {self.__base_uri},', f'bucket: {self.__bucket_name}, prefix: {self.__prefix_name}'
        )
        print(repr(self.soa))

    def client(self, for_write: bool = False) -> Any:
        """Return the S3 client."""
        return ClientContainer.get().get_client(
            for_write=for_write, storage_target=self.soa.storage_target, soa=self.soa
        )

    def get_resource(self) -> Any:
        return ClientContainer.get().get_resource(storage_target=self.soa.storage_target, soa=self.soa)

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
            self.handle_s3_error(e, self.bucket_name, 'validate bucket')
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
        # import time
        # time.sleep(3.0)
        try:
            print(
                '============= Download media object (S3)'
                + f'\nMedia Storage: downloading {src_path} to {dest_path}'
                + f'\nMedia Storage: downloading {self.bucket_name}, {self.prefix}, {src_path} to {dest_path}'
                + '\n'
                + repr(self.soa)
            )
            self.client(for_write=False).download_file(
                Bucket=self.bucket_name, Key=self.prefix + src_path, Filename=str(dest_path)
            )
        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'download file {src_path}')
            raise

    def copy_local_media_file(self, col: Column, src_path: Path) -> str:
        """Copy a local file, and return its new URL"""
        new_file_uri = self._prepare_media_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        key = parsed.path.lstrip('/')
        if self.soa.storage_target == 'r2':
            key = key.split('/', 1)[-1]  # Remove the bucket name from the key for R2
        try:
            _logger.debug(f'Media Storage: copying {src_path} to {new_file_uri} : Key: {key}')
            self.client(for_write=True).upload_file(Filename=str(src_path), Bucket=self.bucket_name, Key=key)
            _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
            return new_file_uri
        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'setup iterator {self.prefix}')
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
            s3_resource = self.get_resource()
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
            self.handle_s3_error(e, self.bucket_name, f'setup iterator {self.prefix}')
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
            self.handle_s3_error(e, self.bucket_name, f'deleting with {self.prefix}')
            raise

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found with the specified S3 uri
        Each returned object includes the full set of prefixes.
        if return_uri is True, the full S3 URI is returned; otherwise, just the object key.
        """
        # I think the n_max parameter should be passed into the list_objects_v2 call
        if self.soa.storage_target == 's3':
            p = f's3://{self.bucket_name}/' if return_uri else ''
        elif self.soa.storage_target == 'r2':
            p = f'https://{self.a_key}.r2.cloudflarestorage.com/{self.bucket_name}/' if return_uri else ''
        else:
            raise ValueError(f'Unsupported storage target: {self.soa.storage_target}')

        s3_client = self.client(for_write=False)
        r: list[str] = []
        try:
            # Use paginator to handle more than 1000 objects
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    if len(r) >= n_max:
                        return r
                    r.append(f'{p}{obj["Key"]}')
        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'list objects from {self.prefix}')
        return r

    @classmethod
    def handle_s3_error(
        cls, e: ClientError, bucket_name: str, operation: str = '', *, ignore_404: bool = False
    ) -> None:
        error_code = e.response.get('Error', {}).get('Code')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        if ignore_404 and error_code == '404':
            return
        if error_code == '404':
            raise excs.Error(f'Bucket {bucket_name} not found during {operation}: {error_message}')
        elif error_code == '403':
            raise excs.Error(f'Access denied to bucket {bucket_name} during {operation}: {error_message}')
        elif error_code == 'PreconditionFailed' or 'PreconditionFailed' in error_message:
            raise excs.Error(f'Precondition failed for bucket {bucket_name} during {operation}: {error_message}')
        else:
            raise excs.Error(f'Error during {operation} in bucket {bucket_name}: {error_code} - {error_message}')
