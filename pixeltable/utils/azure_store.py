import logging
import re
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from azure.core.exceptions import AzureError

from pixeltable import env, exceptions as excs
from pixeltable.config import Config
from pixeltable.utils.object_stores import ObjectPath, ObjectStoreBase, StorageObjectAddress

if TYPE_CHECKING:
    from azure.storage.blob import BlobProperties, BlobServiceClient

    from pixeltable.catalog import Column


_logger = logging.getLogger('pixeltable')


client_lock = threading.Lock()


@env.register_client('azure_blob')
def _() -> dict[str, 'BlobServiceClient']:
    return {}


class AzureBlobStore(ObjectStoreBase):
    """Class to handle Azure Blob Storage operations."""

    # TODO: This needs to be redesigned to use asyncio.

    # URI of the Azure Blob Storage container
    # Always ends with a slash
    __base_uri: str

    # Storage account name
    __account_name: str

    # Container name extracted from the URI
    __container_name: str

    # Prefix path within the container, either empty or ending with a slash
    __prefix_name: str

    # URI scheme (wasb, wasbs, abfs, abfss, https)
    __scheme: str

    soa: StorageObjectAddress

    def __init__(self, soa: StorageObjectAddress):
        self.soa = soa
        self.__scheme = soa.scheme
        self.__account_name = soa.account
        self.__container_name = soa.container
        self.__prefix_name = soa.prefix

        # Reconstruct base URI in normalized format
        self.__base_uri = self.soa.prefix_free_uri + self.__prefix_name
        _logger.info(
            f'Initialized AzureBlobStore with base URI: {self.__base_uri},',
            f'account: {self.__account_name}, container: {self.__container_name}, prefix: {self.__prefix_name}',
        )

    def client(self) -> 'BlobServiceClient':
        """Return the Azure Blob Storage client."""
        client_dict: dict[str, 'BlobServiceClient'] = env.Env.get().get_client('azure_blob')
        with client_lock:
            uri = self.soa.container_free_uri
            if uri not in client_dict:
                storage_account_name = Config.get().get_string_value('storage_account_name', section='azure')
                storage_account_key = Config.get().get_string_value('storage_account_key', section='azure')
                if (storage_account_name is None) != (storage_account_key is None):
                    raise excs.Error(
                        "Azure 'storage_account_name' and 'storage_account_key' must be specified together."
                    )
                if storage_account_name is None or storage_account_name != self.__account_name:
                    # Attempt a connection to a public resource, with no account key
                    client_dict[uri] = self.create_client(endpoint_url=uri)
                else:
                    client_dict[uri] = self.create_client(
                        endpoint_url=uri, account_name=self.__account_name, account_key=storage_account_key
                    )
            return client_dict[uri]

    @property
    def account_name(self) -> str:
        """Return the storage account name."""
        return self.__account_name

    @property
    def container_name(self) -> str:
        """Return the container name from the base URI."""
        return self.__container_name

    @property
    def prefix(self) -> str:
        """Return the prefix from the base URI."""
        return self.__prefix_name

    def validate(self, error_col_name: str) -> str | None:
        """
        Checks if the URI exists and is accessible.

        Returns:
            str: The base URI if the container exists and is accessible, None otherwise.
        """
        try:
            container_client = self.client().get_container_client(self.container_name)
            # Check if container exists by trying to get its properties
            container_client.get_container_properties()
            return self.__base_uri
        except AzureError as e:
            self.handle_azure_error(e, self.container_name, f'validate container {error_col_name}')
        return None

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        """Copies a blob to a local file. Thread safe."""
        try:
            blob_client = self.client().get_blob_client(container=self.container_name, blob=self.prefix + src_path)
            with open(dest_path, 'wb') as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
        except AzureError as e:
            self.handle_azure_error(e, self.container_name, f'download file {src_path}')
            raise

    # TODO: utils package should not include back-references to `Column`
    def copy_local_file(self, col: 'Column', src_path: Path) -> str:
        """Copy a local file to Azure Blob Storage, and return its new URL"""
        prefix, filename = ObjectPath.create_prefix_raw(
            col.get_tbl().id, col.id, col.get_tbl().version, ext=src_path.suffix
        )
        blob_name = f'{self.prefix}{prefix}/{filename}'
        new_file_uri = f'{self.__base_uri}{prefix}/{filename}'

        try:
            blob_client = self.client().get_blob_client(container=self.container_name, blob=blob_name)
            with open(src_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            _logger.debug(f'Media Storage: copied {src_path} to {new_file_uri}')
            return new_file_uri
        except AzureError as e:
            self.handle_azure_error(e, self.container_name, f'upload file {src_path}')
            raise

    def _get_filtered_blobs(
        self, tbl_id: uuid.UUID | None, tbl_version: int | None = None
    ) -> Iterator['BlobProperties']:
        """Private method to get filtered blobs for a table, optionally filtered by version.

        Args:
            tbl_id: Table UUID to filter by
            tbl_version: Optional table version to filter by

        Returns:
            Iterator over blob objects matching the criteria
        """
        # Use ObjectPath to construct the prefix for this table
        if tbl_id is None:
            prefix = self.prefix
            assert tbl_version is None, 'tbl_version must be None if tbl_id is None'
        else:
            table_prefix = ObjectPath.table_prefix(tbl_id)
            prefix = f'{self.prefix}{table_prefix}/'

        try:
            container_client = self.client().get_container_client(self.container_name)

            blob_iterator: Iterator['BlobProperties']
            if tbl_version is None:
                # Return all blobs with the table prefix
                blob_iterator = container_client.list_blobs(name_starts_with=prefix)
            else:
                # Filter by both table_id and table_version using the ObjectPath pattern
                # Pattern: tbl_id_col_id_version_uuid
                version_pattern = re.compile(
                    rf'{re.escape(table_prefix)}_\d+_{re.escape(str(tbl_version))}_[0-9a-fA-F]+.*'
                )
                # Get all blobs with the prefix and filter by version pattern
                all_blobs = container_client.list_blobs(name_starts_with=prefix)
                blob_iterator = (blob for blob in all_blobs if version_pattern.match(blob.name.split('/')[-1]))

            return blob_iterator

        except AzureError as e:
            self.handle_azure_error(e, self.container_name, f'setup iterator {self.prefix}')
            raise

    def count(self, tbl_id: uuid.UUID | None, tbl_version: int | None = None) -> int:
        """Count the number of files belonging to tbl_id. If tbl_version is not None,
        count only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to count blobs for
            tbl_version: Optional table version to filter by

        Returns:
            Number of blobs matching the criteria
        """
        blob_iterator = self._get_filtered_blobs(tbl_id, tbl_version)
        return sum(1 for _ in blob_iterator)

    def delete(self, tbl_id: uuid.UUID, tbl_version: int | None = None) -> int:
        """Delete all files belonging to tbl_id. If tbl_version is not None, delete
        only those files belonging to the specified tbl_version.

        Args:
            tbl_id: Table UUID to delete blobs for
            tbl_version: Optional table version to filter by

        Returns:
            Number of blobs deleted
        """
        assert tbl_id is not None
        blob_iterator = self._get_filtered_blobs(tbl_id, tbl_version)
        total_deleted = 0

        try:
            container_client = self.client().get_container_client(self.container_name)

            for blob in blob_iterator:
                # TODO: Figure out now to properly use batch method delete_blobs(), it doesn't seem to work properly
                container_client.delete_blob(blob.name)
                total_deleted += 1

            # print(f"Deleted {total_deleted} blobs from container '{self.container_name}'.")
            return total_deleted

        except AzureError as e:
            self.handle_azure_error(e, self.container_name, f'deleting with {self.prefix}')
            raise

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects found in the specified destination bucket.
        Each returned object includes the full set of prefixes.
        if return_uri is True, full URI's are returned; otherwise, just the object keys.
        """
        p = self.soa.prefix_free_uri if return_uri else ''
        r: list[str] = []
        try:
            blob_iterator = self._get_filtered_blobs(tbl_id=None, tbl_version=None)
            for blob in blob_iterator:
                r.append(f'{p}{blob.name}')
                if len(r) >= n_max:
                    return r

        except AzureError as e:
            self.handle_azure_error(e, self.__container_name, f'list objects from {self.__base_uri}')
        return r

    @classmethod
    def handle_azure_error(
        cls, e: 'AzureError', container_name: str, operation: str = '', *, ignore_404: bool = False
    ) -> None:
        from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceNotFoundError

        if ignore_404 and isinstance(e, ResourceNotFoundError):
            return

        if isinstance(e, ResourceNotFoundError):
            raise excs.Error(f'Container {container_name} or blob not found during {operation}: {str(e)!r}')
        elif isinstance(e, ClientAuthenticationError):
            raise excs.Error(f'Authentication failed for container {container_name} during {operation}: {str(e)!r}')
        elif isinstance(e, HttpResponseError):
            if e.status_code == 403:
                raise excs.Error(f'Access denied to container {container_name} during {operation}: {str(e)!r}')
            elif e.status_code == 412:
                raise excs.Error(f'Precondition failed for container {container_name} during {operation}: {str(e)!r}')
            else:
                raise excs.Error(
                    f'HTTP error during {operation} in container {container_name}: {e.status_code} - {str(e)!r}'
                )
        else:
            raise excs.Error(f'Error during {operation} in container {container_name}: {str(e)!r}')

    @classmethod
    def create_client(
        cls, endpoint_url: str, account_name: str | None = None, account_key: str | None = None
    ) -> 'BlobServiceClient':
        from azure.core.credentials import AzureNamedKeyCredential
        from azure.storage.blob import BlobServiceClient  # TODO: Use azure.storage.blob.aio instead

        assert (account_name is None) == (account_key is None)
        try:
            #  e.g. endpoint_url: str = f'https://{account_name}.blob.core.windows.net'
            assert endpoint_url is not None, 'No Azure Storage account information provided'

            # Use empty SAS token for anonymous authentication
            credential = None
            if account_name is not None:
                credential = AzureNamedKeyCredential(name=account_name, key=account_key)
            return BlobServiceClient(
                account_url=endpoint_url,
                credential=credential,
                max_single_get_size=(32 * 2**20),
                max_chunk_get_size=(4 * 2**20),
                connection_timeout=15,
                read_timeout=30,
            )
        except Exception as e:
            raise excs.Error(f'Failed to create Azure Blob Storage client: {str(e)!r}') from e
