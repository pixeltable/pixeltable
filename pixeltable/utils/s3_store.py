import logging
import re
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Optional

import boto3
import botocore
from botocore.exceptions import ClientError

from pixeltable import env, exceptions as excs
from pixeltable.config import Config
from pixeltable.utils.object_stores import ObjectPath, ObjectStoreBase, StorageObjectAddress, StorageTarget

if TYPE_CHECKING:
    from botocore.exceptions import ClientError

    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')

client_lock = threading.Lock()


class R2ClientDict(NamedTuple):
    """Container for actual R2 access objects (clients, resources)
    Thread-safe, protected by the module lock 'client_lock'"""

    profile: Optional[str]  # profile used to find credentials
    clients: dict[str, Any]  # Dictionary of URI to client object attached to the URI


@env.register_client('r2')
def _() -> Any:
    profile_name = Config.get().get_string_value('r2_profile')
    return R2ClientDict(profile=profile_name, clients={})


@env.register_client('r2_resource')
def _() -> Any:
    profile_name = Config.get().get_string_value('r2_profile')
    return R2ClientDict(profile=profile_name, clients={})


@env.register_client('s3')
def _() -> Any:
    profile_name = Config.get().get_string_value('s3_profile')
    return S3Store.create_boto_client(profile_name=profile_name)


@env.register_client('s3_resource')
def _() -> Any:
    profile_name = Config.get().get_string_value('s3_profile')
    return S3Store.create_boto_resource(profile_name=profile_name)


class S3Store(ObjectStoreBase):
    """Wrapper for an s3 storage target with all needed methods."""

    # URI of the S3 bucket in the format s3://bucket_name/prefix/
    # Always ends with a slash
    __base_uri: str

    # bucket name extracted from the URI
    __bucket_name: str

    # prefix path within the bucket, either empty or ending with a slash
    __prefix_name: str

    soa: StorageObjectAddress

    def __init__(self, soa: StorageObjectAddress):
        self.soa = soa
        self.__bucket_name = self.soa.container
        self.__prefix_name = self.soa.prefix
        assert self.soa.storage_target in {StorageTarget.R2_STORE, StorageTarget.S3_STORE}, (
            f'Expected storage_target "s3" or "r2", got {self.soa.storage_target}'
        )
        self.__base_uri = self.soa.prefix_free_uri + self.soa.prefix

    def client(self) -> Any:
        """Return a client to access the store."""
        if self.soa.storage_target == StorageTarget.R2_STORE:
            cd = env.Env.get().get_client('r2')
            with client_lock:
                if self.soa.container_free_uri not in cd.clients:
                    cd.clients[self.soa.container_free_uri] = S3Store.create_boto_client(
                        profile_name=cd.profile,
                        extra_args={'endpoint_url': self.soa.container_free_uri, 'region_name': 'auto'},
                    )
                return cd.clients[self.soa.container_free_uri]
        if self.soa.storage_target == StorageTarget.S3_STORE:
            return env.Env.get().get_client('s3')
        raise AssertionError(f'Unexpected storage_target: {self.soa.storage_target}')

    def get_resource(self) -> Any:
        if self.soa.storage_target == StorageTarget.R2_STORE:
            cd = env.Env.get().get_client('r2_resource')
            with client_lock:
                if self.soa.container_free_uri not in cd.clients:
                    cd.clients[self.soa.container_free_uri] = S3Store.create_boto_resource(
                        profile_name=cd.profile,
                        extra_args={'endpoint_url': self.soa.container_free_uri, 'region_name': 'auto'},
                    )
                return cd.clients[self.soa.container_free_uri]
        if self.soa.storage_target == StorageTarget.S3_STORE:
            return env.Env.get().get_client('s3_resource')
        raise AssertionError(f'Unexpected storage_target: {self.soa.storage_target}')

    @property
    def bucket_name(self) -> str:
        """Return the bucket name from the base URI."""
        return self.__bucket_name

    @property
    def prefix(self) -> str:
        """Return the prefix from the base URI."""
        return self.__prefix_name

    def validate(self, error_col_name: str) -> Optional[str]:
        """
        Checks if the URI exists.

        Returns:
            bool: True if the S3 URI exists and is accessible, False otherwise.
        """
        try:
            self.client().head_bucket(Bucket=self.bucket_name)
            return self.__base_uri
        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'validate bucket {error_col_name}')
        return None

    def _prepare_uri_raw(self, tbl_id: uuid.UUID, col_id: int, tbl_version: int, ext: Optional[str] = None) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        prefix, filename = ObjectPath.create_prefix_raw(tbl_id, col_id, tbl_version, ext)
        parent = f'{self.__base_uri}{prefix}'
        return f'{parent}/{filename}'

    def _prepare_uri(self, col: 'Column', ext: Optional[str] = None) -> str:
        """
        Construct a new, unique URI for a persisted media file.
        """
        assert col.tbl is not None, 'Column must be associated with a table'
        return self._prepare_uri_raw(col.tbl.id, col.id, col.tbl.version, ext=ext)

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        """Copies an object to a local file. Thread safe."""
        try:
            self.client().download_file(Bucket=self.bucket_name, Key=self.prefix + src_path, Filename=str(dest_path))
        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'download file {src_path}')
            raise

    def copy_local_file(self, col: 'Column', src_path: Path) -> str:
        """Copy a local file, and return its new URL"""
        new_file_uri = self._prepare_uri(col, ext=src_path.suffix)
        parsed = urllib.parse.urlparse(new_file_uri)
        key = parsed.path.lstrip('/')
        if self.soa.storage_target == StorageTarget.R2_STORE:
            key = key.split('/', 1)[-1]  # Remove the bucket name from the key for R2
        try:
            _logger.debug(f'Media Storage: copying {src_path} to {new_file_uri} : Key: {key}')
            self.client().upload_file(Filename=str(src_path), Bucket=self.bucket_name, Key=key)
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
        # Use ObjectPath to construct the prefix for this table
        table_prefix = ObjectPath.table_prefix(tbl_id)
        prefix = f'{self.prefix}{table_prefix}/'

        try:
            # Use S3 resource interface for filtering
            s3_resource = self.get_resource()
            bucket = s3_resource.Bucket(self.bucket_name)

            if tbl_version is None:
                # Return all objects with the table prefix
                object_iterator = bucket.objects.filter(Prefix=prefix)
            else:
                # Filter by both table_id and table_version using the ObjectPath pattern
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

            return total_deleted

        except ClientError as e:
            self.handle_s3_error(e, self.bucket_name, f'deleting with {self.prefix}')
            raise

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found in the specified destination bucket.
        Each returned object includes the full set of prefixes.
        if return_uri is True, full URI's are returned; otherwise, just the object keys.
        """
        p = self.soa.prefix_free_uri if return_uri else ''

        s3_client = self.client()
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
        cls, e: 'ClientError', bucket_name: str, operation: str = '', *, ignore_404: bool = False
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

    @classmethod
    def create_boto_session(cls, profile_name: Optional[str] = None) -> Any:
        """Create a boto session using the defined profile"""
        if profile_name:
            try:
                _logger.info(f'Creating boto session with profile {profile_name}')
                session = boto3.Session(profile_name=profile_name)
                return session
            except Exception as e:
                _logger.info(f'Error occurred while creating boto session with profile {profile_name}: {e}')
        return boto3.Session()

    @classmethod
    def create_boto_client(cls, profile_name: Optional[str] = None, extra_args: Optional[dict[str, Any]] = None) -> Any:
        config_args: dict[str, Any] = {
            'max_pool_connections': 30,
            'connect_timeout': 15,
            'read_timeout': 30,
            'retries': {'max_attempts': 3, 'mode': 'adaptive'},
        }

        session = cls.create_boto_session(profile_name)

        try:
            # Check if credentials are available
            session.get_credentials().get_frozen_credentials()
            config = botocore.config.Config(**config_args)
            return session.client('s3', config=config, **(extra_args or {}))  # credentials are available
        except Exception as e:
            _logger.info(f'Error occurred while creating S3 client: {e}, fallback to unsigned mode')
            # No credentials available, use unsigned mode
            config_args = config_args.copy()
            config_args['signature_version'] = botocore.UNSIGNED
            config = botocore.config.Config(**config_args)
            return boto3.client('s3', config=config)

    @classmethod
    def create_boto_resource(
        cls, profile_name: Optional[str] = None, extra_args: Optional[dict[str, Any]] = None
    ) -> Any:
        # Create a session using the defined profile
        return cls.create_boto_session(profile_name).resource('s3', **(extra_args or {}))
