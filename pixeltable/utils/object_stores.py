from __future__ import annotations

import enum
import os
import re
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from pixeltable import env, exceptions as excs

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class StorageTarget(enum.Enum):
    """Enumeration of storage kinds."""

    LOCAL_STORE = 'os'  # Local file system
    S3_STORE = 's3'  # Amazon S3
    R2_STORE = 'r2'  # Cloudflare R2
    B2_STORE = 'b2'  # Backblaze B2
    TIGRIS_STORE = 'tigris'  # Tigris
    GCS_STORE = 'gs'  # Google Cloud Storage
    AZURE_STORE = 'az'  # Azure Blob Storage
    HTTP_STORE = 'http'  # HTTP/HTTPS

    def __str__(self) -> str:
        return self.value


class StorageObjectAddress(NamedTuple):
    """Contains components of an object address.
    Unused components are empty strings.
    """

    storage_target: StorageTarget  # The kind of storage referenced. This is NOT the same as the scheme.
    scheme: str  # The scheme parsed from the source
    account: str = ''  # Account number parsed from the source when applicable
    account_extension: str = ''  # Account extension parsed from the source when applicable
    container: str = ''  # Container / bucket name parsed from the source
    key: str = ''  # Key parsed from the source (prefix + object_name)
    prefix: str = ''  # Prefix (within the bucket) parsed from the source
    object_name: str = ''  # Object name parsed from the source (if requested and applicable)
    path: Path | None = None

    @property
    def has_object(self) -> bool:
        return len(self.object_name) > 0

    @property
    def is_http_readable(self) -> bool:
        return self.scheme.startswith('http') and self.has_object

    @property
    def is_azure_scheme(self) -> bool:
        return self.scheme in ('wasb', 'wasbs', 'abfs', 'abfss')

    @property
    def has_valid_storage_target(self) -> bool:
        return self.storage_target in (
            StorageTarget.LOCAL_STORE,
            StorageTarget.S3_STORE,
            StorageTarget.R2_STORE,
            StorageTarget.B2_STORE,
            StorageTarget.TIGRIS_STORE,
            StorageTarget.GCS_STORE,
            StorageTarget.AZURE_STORE,
            StorageTarget.HTTP_STORE,
        )

    @property
    def prefix_free_uri(self) -> str:
        """Return the URI without any prefixes."""
        if self.is_azure_scheme:
            return f'{self.scheme}://{self.container}@{self.account}.{self.account_extension}/'
        if self.account and self.account_extension:
            return f'{self.scheme}://{self.account}.{self.account_extension}/{self.container}/'
        if self.account_extension:
            return f'{self.scheme}://{self.account_extension}/{self.container}/'
        return f'{self.scheme}://{self.container}/'

    @property
    def container_free_uri(self) -> str:
        """Return the URI without any prefixes."""
        assert not self.is_azure_scheme, 'Azure storage requires a container name'
        if self.account and self.account_extension:
            return f'{self.scheme}://{self.account}.{self.account_extension}/'
        if self.account_extension:
            return f'{self.scheme}://{self.account_extension}/'
        return f'{self.scheme}://'

    @property
    def to_path(self) -> Path:
        assert self.storage_target == StorageTarget.LOCAL_STORE
        assert self.path is not None
        return self.path

    def __str__(self) -> str:
        """A debug aid to override default str representation. Not to be used for any purpose."""
        return f'{self.storage_target}..{self.scheme}://{self.account}.{self.account_extension}/{self.container}/{self.prefix}{self.object_name}'

    def __repr__(self) -> str:
        """A debug aid to override default repr representation. Not to be used for any purpose."""
        return (
            f'SObjectAddress(client: {self.storage_target!r}, s: {self.scheme!r}, a: {self.account!r}, '
            f'ae: {self.account_extension!r}, c: {self.container!r}, '
            f'p: {self.prefix!r}, o: {self.object_name!r})'
        )


class ObjectPath:
    PATTERN = re.compile(r'([0-9a-fA-F]+)_(\d+)_(\d+)_([0-9a-fA-F]+)')  # tbl_id, col_id, version, uuid

    @classmethod
    def table_prefix(cls, tbl_id: UUID) -> str:
        """Construct a unique unix-style prefix for objects in a table (without leading/trailing slashes)."""
        assert isinstance(tbl_id, uuid.UUID)
        return tbl_id.hex

    @classmethod
    def create_prefix_raw(cls, tbl_id: UUID, col_id: int, tbl_version: int, ext: str | None = None) -> tuple[str, str]:
        """Construct a unique unix-style prefix and filename for a persisted file.
        The results are derived from table, col, and version specs.
        Returns:
            prefix: a unix-style prefix for the file without leading/trailing slashes
            filename: a unique filename for the file without leading slashes
        """
        table_prefix = cls.table_prefix(tbl_id)
        id_hex = uuid.uuid4().hex
        prefix = f'{table_prefix}/{id_hex[:2]}/{id_hex[:4]}'
        filename = f'{table_prefix}_{col_id}_{tbl_version}_{id_hex}{ext or ""}'
        return prefix, filename

    @classmethod
    def separate_prefix_object(cls, path_and_object: str, may_contain_object_name: bool) -> tuple[str, str]:
        path = path_and_object
        object_name = ''
        if not may_contain_object_name or path.endswith('/'):
            prefix = path.rstrip('/')
        elif '/' in path:
            # If there are slashes in the path, separate into prefix and object
            prefix, object_name = path.rsplit('/', 1)
            prefix = prefix.rstrip('/')
        else:
            # If no slashes, the entire path is the object name
            prefix = ''
            object_name = path
        if len(prefix) > 0 and not prefix.endswith('/'):
            prefix += '/'
        return prefix, object_name

    @classmethod
    def parse_object_storage_addr1(cls, src_addr: str) -> StorageObjectAddress:
        """
        Parses a cloud storage URI into its scheme, bucket, and key.

        Args:
            uri (str): The cloud storage URI (e.g., "gs://my-bucket/path/to/object.txt").

        Returns:
            StorageObjectAddress: A NamedTuple containing components of the address.

        Formats:
            s3://container/<optional prefix>/<optional object>
            gs://container/<optional prefix>/<optional object>
            wasb[s]://container@account.blob.core.windows.net/<optional prefix>/<optional object>
            abfs[s]://container@account.dfs.core.windows.net/<optional prefix>/<optional object>
            https://account.blob.core.windows.net/container/<optional prefix>/<optional object>
            https://account.r2.cloudflarestorage.com/container/<optional prefix>/<optional object>
            https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg
        """
        parsed = urllib.parse.urlparse(src_addr)
        scheme = parsed.scheme.lower()
        account_name = ''
        account_extension = ''
        container = ''
        key = ''
        path = None

        # len(parsed.scheme) == 1 occurs for Windows drive letters like C:\
        if not parsed.scheme or len(parsed.scheme) == 1:
            # If no scheme, treat as local file path; this will be further validated before use
            storage_target = StorageTarget.LOCAL_STORE
            scheme = 'file'
            path = Path(src_addr)

        elif scheme == 'file':
            storage_target = StorageTarget.LOCAL_STORE
            pth = parsed.path
            if parsed.netloc:
                # This is a UNC path, ie, file://host/share/path/to/file
                pth = f'\\\\{parsed.netloc}{pth}'
            path = Path(urllib.parse.unquote(urllib.request.url2pathname(pth)))
            key = str(parsed.path).lstrip('/')

        elif scheme in ('s3', 'gs'):
            storage_target = StorageTarget.S3_STORE if scheme == 's3' else StorageTarget.GCS_STORE
            container = parsed.netloc
            key = parsed.path.lstrip('/')

        elif scheme in ('wasb', 'wasbs', 'abfs', 'abfss'):
            # Azure-specific URI schemes
            # wasb[s]://container@account.blob.core.windows.net/<optional prefix>/<optional object>
            # abfs[s]://container@account.dfs.core.windows.net/<optional prefix>/<optional object>
            storage_target = StorageTarget.AZURE_STORE
            container_and_account = parsed.netloc
            if '@' in container_and_account:
                container, account_host = container_and_account.split('@', 1)
                account_name = account_host.split('.')[0]
                account_extension = account_host.split('.', 1)[1]
            else:
                raise ValueError(f'Invalid Azure URI format: {src_addr}')
            key = parsed.path.lstrip('/')

        elif scheme in ('http', 'https'):
            # Standard HTTP(S) URL format
            # https://account.blob.core.windows.net/container/<optional path>/<optional object>
            # https://account.r2.cloudflarestorage.com/container/<optional path>/<optional object>
            # https://s3.us-west-004.backblazeb2.com/container/<optional path>/<optional object>
            # https://t3.storage.dev/container/<optional path>/<optional object> (Tigris)
            # and possibly others
            key = parsed.path
            if 'cloudflare' in parsed.netloc:
                storage_target = StorageTarget.R2_STORE
            elif 'backblazeb2' in parsed.netloc:
                storage_target = StorageTarget.B2_STORE
            elif 'windows' in parsed.netloc:
                storage_target = StorageTarget.AZURE_STORE
            elif 't3.storage.dev' in parsed.netloc:
                storage_target = StorageTarget.TIGRIS_STORE
            else:
                storage_target = StorageTarget.HTTP_STORE
            if storage_target in (
                StorageTarget.S3_STORE,
                StorageTarget.AZURE_STORE,
                StorageTarget.R2_STORE,
                StorageTarget.B2_STORE,
                StorageTarget.TIGRIS_STORE,
            ):
                account_name = parsed.netloc.split('.', 1)[0]
                account_extension = parsed.netloc.split('.', 1)[1]
                path_parts = key.lstrip('/').split('/', 1)
                container = path_parts[0] if path_parts else ''
                key = path_parts[1] if len(path_parts) > 1 else ''
            else:
                account_extension = parsed.netloc
            key = key.lstrip('/')
        else:
            raise ValueError(f'Unsupported URI scheme: {parsed.scheme}')

        r = StorageObjectAddress(storage_target, scheme, account_name, account_extension, container, key, '', '', path)
        assert r.has_valid_storage_target
        return r

    @classmethod
    def parse_object_storage_addr(cls, src_addr: str, allow_obj_name: bool) -> StorageObjectAddress:
        """
        Parses a cloud storage URI into its scheme, bucket, prefix, and object name.

        Args:
            uri (str): The cloud storage URI (e.g., "gs://my-bucket/path/to/object.txt").

        Returns:
            StorageObjectAddress: A NamedTuple containing components of the address.

        Formats:
            s3://container/<optional prefix>/<optional object>
            gs://container/<optional prefix>/<optional object>
            wasb[s]://container@account.blob.core.windows.net/<optional prefix>/<optional object>
            abfs[s]://container@account.dfs.core.windows.net/<optional prefix>/<optional object>
            https://account.blob.core.windows.net/container/<optional prefix>/<optional object>
            https://account.r2.cloudflarestorage.com/container/<optional prefix>/<optional object>
            https://raw.github.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg
        """
        soa = cls.parse_object_storage_addr1(src_addr)
        prefix, object_name = cls.separate_prefix_object(soa.key, allow_obj_name)
        assert not object_name.endswith('/')
        r = soa._replace(prefix=prefix, object_name=object_name)
        return r


class ObjectStoreBase:
    def validate(self, error_prefix: str) -> str | None:
        """Check the store configuration. Returns base URI if store is accessible.

        Args:
            error_col_name: a string of the form 'Column {name}: ' used when raising errors

        Returns:
            Base URI for the store. This value is stored in any Column attached to the store.
        """
        raise AssertionError

    def copy_local_file(self, col: Column, src_path: Path) -> str:
        """Copy a file associated with a Column to the store, returning the file's URL within the destination.

        Args:
            col: The Column to which the file belongs, used to generate the URI of the stored object.
            src_path: The Path to the local file

        Returns:
            The URI of the object in the store
        """
        raise AssertionError

    def move_local_file(self, col: Column, src_path: Path) -> str | None:
        """Move a file associated with a Column to the store, returning the file's URL within the destination.

        Args:
            col: The Column to which the file belongs, used to generate the URI of the stored object.
            src_path: The Path to the local file

        Returns:
            The URI of the object in the store, None if the object cannot be moved to the store
        """
        return None

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        """Copies an object from the store to a local media file.

        Args:
            src_path: The URI of the object in the store
            dest_path: The desired Path to the local file
        """
        raise AssertionError

    def count(self, tbl_id: UUID, tbl_version: int | None = None) -> int:
        """Return the number of objects in the store associated with the given tbl_id

        Args:
            tbl_id: Only count objects associated with a given table
            tbl_version: Only count objects associated with a specific table version

        Returns:
            Number of objects found with the specified criteria
        """
        raise AssertionError

    def delete(self, tbl_id: UUID, tbl_version: int | None = None) -> int | None:
        """Delete objects in the destination for a given table ID, table version.

        Args:
            tbl_id: Only count objects associated with a given table
            tbl_version: Only count objects associated with a specific table version

        Returns:
            Number of objects deleted or None if the store does not count deletions.
        """
        raise AssertionError

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        """Return a list of objects in the store.

        Args:
            return_uri: If True, returns a full URI for each object, otherwise just the path to the object.
            n_max: Maximum number of objects to list
        """
        raise AssertionError

    def create_presigned_url(self, soa: StorageObjectAddress, expiration_seconds: int) -> str:
        """Create a presigned URL for downloading an object from the store.

        Args:
            soa: StorageObjectAddress containing the object location
            expiration_seconds: Time in seconds for the URL to remain valid

        Returns:
            A presigned HTTP URL that can be used to access the object
        """
        raise AssertionError


class ObjectOps:
    @classmethod
    def get_store(
        cls, dest: str | StorageObjectAddress | None, allow_obj_name: bool, col_name: str | None = None
    ) -> ObjectStoreBase:
        from pixeltable.env import Env
        from pixeltable.utils.local_store import LocalStore

        dest = dest or str(Env.get().media_dir)  # Use local media dir as fallback
        soa = (
            dest
            if isinstance(dest, StorageObjectAddress)
            else ObjectPath.parse_object_storage_addr(dest, allow_obj_name=allow_obj_name)
        )
        if soa.storage_target == StorageTarget.LOCAL_STORE:
            return LocalStore(soa)
        if soa.storage_target in (
            StorageTarget.S3_STORE,
            StorageTarget.R2_STORE,
            StorageTarget.B2_STORE,
            StorageTarget.TIGRIS_STORE,
        ):
            env.Env.get().require_package('boto3')
            from pixeltable.utils.s3_store import S3Store

            return S3Store(soa)
        if soa.storage_target == StorageTarget.GCS_STORE and soa.scheme == 'gs':
            env.Env.get().require_package('google.cloud.storage')
            from pixeltable.utils.gcs_store import GCSStore

            return GCSStore(soa)
        if soa.storage_target == StorageTarget.AZURE_STORE:
            env.Env.get().require_package('azure.storage.blob')
            from pixeltable.utils.azure_store import AzureBlobStore

            return AzureBlobStore(soa)
        if soa.storage_target == StorageTarget.HTTP_STORE and soa.is_http_readable:
            return HTTPStore(soa)
        error_col_name = f'Column {col_name!r}: ' if col_name is not None else ''
        raise excs.Error(
            f'{error_col_name}`destination` must be a valid reference to a supported destination, got {dest!r}'
        )

    @classmethod
    def validate_destination(cls, dest: str | Path | None, col_name: str | None = None) -> str:
        """Convert a Column destination parameter to a URI, else raise errors.
        Args:
            dest: The requested destination
            col_name: Used to raise error messages
        Returns:
            URI of destination, or raises an error
        """
        error_col_str = f'column {col_name!r}' if col_name is not None else ''

        # General checks on any destination
        if isinstance(dest, Path):
            dest = str(dest)
        if dest is not None and not isinstance(dest, str):
            raise excs.Error(f'{error_col_str}: `destination` must be a string or path; got {dest!r}')

        # Specific checks for storage backends
        store = cls.get_store(dest, False, col_name)
        dest2 = store.validate(error_col_str)
        if dest2 is None:
            raise excs.Error(f'{error_col_str}: `destination` must be a supported destination; got {dest!r}')
        return dest2

    @classmethod
    def copy_object_to_local_file(cls, src_uri: str, dest_path: Path) -> None:
        """Copy an object from a URL to a local Path. Thread safe.
        Raises an exception if the download fails or the scheme is not supported
        """
        soa = ObjectPath.parse_object_storage_addr(src_uri, allow_obj_name=True)
        store = cls.get_store(src_uri, True)
        store.copy_object_to_local_file(soa.object_name, dest_path)

    @classmethod
    def put_file(cls, col: Column, src_path: Path, relocate_or_delete: bool) -> str:
        """Move or copy a file to the destination, returning the file's URL within the destination.
        If relocate_or_delete is True and the file is in the TempStore, the file will be deleted after the operation.
        """
        from pixeltable.utils.local_store import TempStore

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
    def delete(cls, dest: str | None, tbl_id: UUID, tbl_version: int | None = None) -> int | None:
        """Delete objects in the destination for a given table ID, table version.
        Returns:
            Number of objects deleted or None
        """
        store = cls.get_store(dest, False)
        return store.delete(tbl_id, tbl_version)

    @classmethod
    def count(
        cls,
        tbl_id: UUID,
        tbl_version: int | None = None,
        dest: str | None = None,
        default_input_dest: bool = False,
        default_output_dest: bool = False,
    ) -> int:
        """
        Return the count of objects in the destination for a given table ID.

        At most one of dest, default_input, default_output may be specified. If none are specified, the fallback is the
        local media directory.

        Args:
            tbl_id: Table ID for which to count objects
            tbl_version: If specified, only counts objects for a specific table version
            dest: The destination to count objects in
            default_input_dest: If `True`, use the default input media destination
            default_output_dest: If `True`, use the default output media destination
        """
        assert sum((dest is not None, default_input_dest, default_output_dest)) <= 1, (
            'At most one of dest, default_input, default_output may be specified'
        )
        if default_input_dest:
            dest = env.Env.get().default_input_media_dest
        if default_output_dest:
            dest = env.Env.get().default_output_media_dest
        store = cls.get_store(dest, False)
        return store.count(tbl_id, tbl_version)

    @classmethod
    def list_objects(cls, dest: str | None, return_uri: bool, n_max: int = 10) -> list[str]:
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
        url = self.base_url + src_path
        req = urllib.request.Request(url, headers={'User-Agent': 'Pixeltable/1.0 (https://pixeltable.com)'})
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
            data = resp.read()
            f.write(data)
            f.flush()  # Ensures Python buffers are written to OS
            os.fsync(f.fileno())  # Forces OS to write to physical storage

    def create_presigned_url(self, soa: StorageObjectAddress, expiration_seconds: int) -> str:
        """Create a presigned URL for HTTP storage (returns the HTTP URL as-is).

        Args:
            soa: StorageObjectAddress containing the object location
            expiration_seconds: Time in seconds for the URL to remain valid (ignored for HTTP)

        Returns:
            The HTTP URL as-is since it's already servable
        """
        if not soa.has_object:
            raise excs.Error(f'StorageObjectAddress does not contain an object name: {soa}')

        # Construct the full HTTP URL from the StorageObjectAddress
        return f'{soa.scheme}://{soa.account_extension}/{soa.key}'
