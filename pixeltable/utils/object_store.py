from __future__ import annotations

import logging
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import boto3
from botocore.exceptions import ClientError

from pixeltable.utils.media_path import MediaPath
from pixeltable.utils.s3 import S3ClientContainer

if TYPE_CHECKING:
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

    __client_source: S3ClientContainer

    def __init__(self, client_source: S3ClientContainer, uri: str):
        assert uri.startswith('s3'), "URI must start with 's3'"
        parsed_uri = urllib.parse.urlparse(uri)
        assert parsed_uri.scheme == 's3', 'URI must be an S3 URI'
        self.__base_uri = uri.rstrip('/')
        self.__bucket_name = parsed_uri.netloc
        self.__prefix_name = parsed_uri.path.lstrip('/').rstrip('/')
        if len(self.__prefix_name) > 0:
            self.__prefix_name += '/'
        self.__base_uri += '/'
        self.__client_source = client_source
        print(
            f'Initialized S3Store with base URI: {self.__base_uri},',
            f'bucket: {self.__bucket_name}, prefix: {self.__prefix_name}',
        )

    def client(self, for_write: bool = False) -> Any:
        """Return the S3 client."""
        return self.__client_source.get_client(for_write=for_write)

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
        except ClientError:
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
        new_file_uri = self._prepare_media_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        self.client(for_write=True).upload_file(
            Filename=str(src_path), Bucket=parsed.netloc, Key=parsed.path.lstrip('/')
        )
        #        _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
        return new_file_uri

    def delete_objects_with_prefix(self, prefix: str) -> int:
        """
        Deletes all objects in this S3 bucket with a specified prefix.

        Args:
            prefix (str): The prefix of the objects to delete.

        Returns:
            number of objects deleted (int)
        """
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.bucket_name)

        # List objects with the specified prefix
        objects_to_delete = []
        for obj in bucket.objects.filter(Prefix=prefix):
            objects_to_delete.append({'Key': obj.key})

        n = 0
        while len(objects_to_delete) > 0:
            # Delete objects in batches (max 1000 per request)
            bucket.delete_objects(
                Delete={
                    'Objects': objects_to_delete[:1000],
                    'Quiet': True,  # Set to False to get details about each deletion
                }
            )
            n += min(1000, len(objects_to_delete))
            del objects_to_delete[:1000]
        print(f"Deleted {len(objects_to_delete)} objects with prefix '{prefix}' from bucket '{self.bucket_name}'.")
        return n

    def delete(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> None:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version."""
        assert tbl_id is not None
        if tbl_version is None:
            # Remove the entire folder for this table id.
            prefix = f'{self.prefix}{MediaPath.media_table_prefix(tbl_id)}/'
            self.delete_objects_with_prefix(prefix)
        else:
            # Silently ignore deletion for specific table versions
            return

    #            raise NotImplementedError(
    #                f'Deleting S3 objects for a specific table version {tbl_version} is not implemented yet.'
    #            )

    def count_objects_with_prefix(self, prefix: str) -> int:
        """
        Count S3 objects with given prefix whose names match a regex pattern.

        Args:
            pattern: Regex pattern to match object names

        Returns:
            Number of matching objects
        """
        count = 0
        continuation_token = None
        try:
            while True:
                # Prepare parameters for list_objects_v2
                params = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix,
                    'MaxKeys': 1000,  # Maximum allowed per request
                }

                # Add continuation token if we have one (for pagination)
                if continuation_token:
                    params['ContinuationToken'] = continuation_token

                # List objects
                response = self.client().list_objects_v2(**params)

                # Count matching objects
                if 'Contents' in response:
                    for _ in response['Contents']:
                        count += 1

                # Check if there are more objects to retrieve
                if response.get('IsTruncated', False):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

        except Exception as e:
            raise Exception(f"Error accessing S3 bucket '{self.bucket_name}': {str(e)!r}") from e

        return count

    def count(self, tbl_id: uuid.UUID, tbl_version: Optional[int] = None) -> int:
        """Count the number of files belonging to tbl_id. If tbl_version is not None,
        count only those files belonging to the specified tbl_version."""
        assert tbl_id is not None
        if tbl_version is None:
            prefix = f'{self.prefix}{MediaPath.media_table_prefix(tbl_id)}/'
            return self.count_objects_with_prefix(prefix=prefix)
        else:
            raise NotImplementedError(
                f'Counting S3 objects for a specific table version {tbl_version} is not implemented yet.'
            )
