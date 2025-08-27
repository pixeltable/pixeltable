from __future__ import annotations

import re
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional
from uuid import UUID

if TYPE_CHECKING:
    from pixeltable.catalog import Column


class StorageObjectAddress(NamedTuple):
    """Contains components of a media address.
    Unused components are empty strings.
    """

    storage_target: str
    scheme: str
    account: str
    account_extension: str
    container: str
    key: str
    prefix: str = ''
    object_name: str = ''

    @property
    def has_object(self) -> bool:
        return len(self.object_name) > 0

    @property
    def is_http_readable(self) -> bool:
        return self.scheme.startswith('http') and self.has_object

    @property
    def is_azure_scheme(self) -> bool:
        return self.scheme in ['wasb', 'wasbs', 'abfs', 'abfss']

    @property
    def has_valid_storage_target(self) -> bool:
        return self.storage_target in ['os', 's3', 'r2', 'gs', 'az', 'http']

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
        assert self.storage_target == 'os'
        assert self.prefix
        path_str = urllib.parse.unquote(urllib.request.url2pathname(self.prefix))
        return Path(path_str)

    def __str__(self) -> str:
        return (
            f'{self.scheme}://{self.account}.{self.account_extension}/{self.container}/{self.prefix}{self.object_name}'
        )

    def __repr__(self) -> str:
        return (
            f'SObjectAddress(client: {self.storage_target!r}, s: {self.scheme!r}, a: {self.account!r}, '
            f'ae: {self.account_extension!r}, c: {self.container!r}, '
            f'p: {self.prefix!r}, o: {self.object_name!r})'
        )


class MediaPath:
    PATTERN = re.compile(r'([0-9a-fA-F]+)_(\d+)_(\d+)_([0-9a-fA-F]+)')  # tbl_id, col_id, version, uuid

    @classmethod
    def media_table_prefix(cls, tbl_id: UUID) -> str:
        """Construct a unique unix-style prefix for a media table without leading/trailing slashes."""
        assert isinstance(tbl_id, uuid.UUID)
        return f'{tbl_id.hex}'

    @classmethod
    def media_prefix_file_raw(
        cls, tbl_id: UUID, col_id: int, tbl_version: int, ext: Optional[str] = None
    ) -> tuple[str, str]:
        """Construct a unique unix-style prefix and filename for a persisted media file.
        The results are derived from table, col, and version specs.
        Returns:
            prefix: a unix-style prefix for the media file without leading/trailing slashes
            filename: a unique filename for the media file without leading slashes
        """
        table_prefix = cls.media_table_prefix(tbl_id)
        media_id_hex = uuid.uuid4().hex
        prefix = f'{table_prefix}/{media_id_hex[:2]}/{media_id_hex[:4]}'
        filename = f'{table_prefix}_{col_id}_{tbl_version}_{media_id_hex}{ext or ""}'
        return prefix, filename

    @classmethod
    def media_prefix_file(cls, col: Column, ext: Optional[str] = None) -> tuple[str, str]:
        """Construct a unique unix-style prefix and filename for a persisted media file.
        The results are derived from a Column specs.
        Returns:
            prefix: a unix-style prefix for the media file without leading/trailing slashes
            filename: a unique filename for the media file without leading slashes
        """
        assert col.tbl is not None, 'Column must be associated with a table'
        return cls.media_prefix_file_raw(col.tbl.id, col.id, col.tbl.version, ext=ext)

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
    def parse_media_storage_addr1(cls, src_addr: str) -> StorageObjectAddress:
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

        account_name = ''
        account_extension = ''
        container = ''
        key = ''

        # If no scheme, treat as local file path; this will be further validated before use
        scheme = 'file' if not parsed.scheme else parsed.scheme.lower()

        # len(parsed.scheme) == 1 handles Windows drive letters like C:\
        if scheme == 'file' or len(scheme) == 1:
            storage_target = 'os'
            scheme = 'file'
            key = parsed.path

        elif scheme in ('s3', 'gs'):
            storage_target = scheme
            container = parsed.netloc
            key = parsed.path.lstrip('/')

        elif scheme in ['wasb', 'wasbs', 'abfs', 'abfss']:
            # Azure-specific URI schemes
            # wasb[s]://container@account.blob.core.windows.net/<optional prefix>/<optional object>
            # abfs[s]://container@account.dfs.core.windows.net/<optional prefix>/<optional object>
            storage_target = 'az'
            container_and_account = parsed.netloc
            if '@' in container_and_account:
                container, account_host = container_and_account.split('@', 1)
                account_name = account_host.split('.')[0]
                account_extension = account_host.split('.', 1)[1]
            else:
                raise ValueError(f'Invalid Azure URI format: {src_addr}')
            key = parsed.path.lstrip('/')

        elif scheme in ['http', 'https']:
            # Standard HTTP(S) URL format
            # https://account.blob.core.windows.net/container/<optional path>/<optional object>
            # https://account.r2.cloudflarestorage.com/container/<optional path>/<optional object>
            # and possibly others
            key = parsed.path
            if 'cloudflare' in parsed.netloc:
                storage_target = 'r2'
            elif 'windows' in parsed.netloc:
                storage_target = 'az'
            else:
                storage_target = 'http'
            if storage_target in ['s3', 'az', 'r2']:
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

        r = StorageObjectAddress(storage_target, scheme, account_name, account_extension, container, key)
        assert r.has_valid_storage_target
        return r

    @classmethod
    def parse_media_storage_addr(cls, src_addr: str, may_contain_object_name: bool) -> StorageObjectAddress:
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
        soa = cls.parse_media_storage_addr1(src_addr)
        prefix, object_name = cls.separate_prefix_object(soa.key, may_contain_object_name)
        assert not object_name.endswith('/')
        r = soa._replace(prefix=prefix, object_name=object_name)
        return r
