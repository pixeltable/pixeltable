"""Pixeltable Cloud home storage (``pxtfs://org:db/<bucket>/...``)"""

from __future__ import annotations

import logging
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import botocore
import botocore.credentials
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
from botocore.credentials import CredentialResolver, RefreshableCredentials
from botocore.session import get_session as get_botocore_session

from pixeltable import ErrorCode, env, exceptions as excs
from pixeltable.runtime import get_runtime
from pixeltable.utils.cloud_utils import get_bucket_credentials, get_presigned_url_from_cloud
from pixeltable.utils.object_stores import S3_COMPATIBLE_TARGETS, ObjectStoreBase, StorageObjectAddress, StorageTarget
from pixeltable.utils.s3_store import S3CompatClientDict, S3Store

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')

_PXTFS_URI_PATTERN = re.compile(r'^pxtfs://[^/]+/([^/?#]+)(.*)$')


@dataclass
class _PxtStoreCacheEntry:
    """Cached boto3 client/resource and quota state for a bucket."""

    client: BaseClient | None  # populated after boto3 session is built
    resource: ServiceResource | None  # populated after boto3 session is built
    physical_bucket_name: str
    endpoint_url: str
    storage_provider: str
    no_space_left: bool = False
    no_space_warned: bool = False  # tracks whether warning has been issued for no space left in pixeltable store


# pxt_store clients are thread-local (via Runtime._clients), consistent with r2/s3/b2/tigris.
@env.register_client('pxt_store')
def _() -> S3CompatClientDict:
    return S3CompatClientDict(profile=None, clients={})


class _PxtStoreCredentialProvider(botocore.credentials.CredentialProvider):
    """Supplies RefreshableCredentials to a botocore session via the CredentialResolver chain."""

    METHOD = 'pxt-store'
    CANONICAL_NAME = 'custom-pxt-store'

    def __init__(self, creds: RefreshableCredentials) -> None:
        self._creds = creds

    def load(self) -> RefreshableCredentials:
        return self._creds


def _handle_no_space_warning(no_space_left: bool, entry: _PxtStoreCacheEntry, org: str, db: str, bucket: str) -> None:
    if no_space_left:
        if entry.no_space_warned:
            return
        warnings.warn(
            f'Pixeltable store for pxtfs://{org}:{db}/{bucket} has no space left. '
            'Only read and delete operations are allowed.',
            category=excs.PixeltableWarning,
            stacklevel=3,
        )
        entry.no_space_warned = True
    else:
        entry.no_space_warned = False


def _refresh_credentials(org: str, db: str, bucket: str, prefix: str, entry: _PxtStoreCacheEntry) -> dict[str, str]:
    """Fetch fresh credentials and update the cache entry"""
    creds = get_bucket_credentials(org, db, bucket, prefix)
    expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)

    entry.no_space_left = creds.no_space_left
    if creds.resolved_bucket_name:
        entry.physical_bucket_name = creds.resolved_bucket_name

    _handle_no_space_warning(creds.no_space_left, entry, org, db, bucket)

    _logger.info(
        'Refreshed bucket credentials for %s:%s (ttl=%ds, no_space_left=%s)',
        org,
        db,
        creds.ttl_seconds,
        creds.no_space_left,
    )
    return {
        'access_key': creds.access_key_id,
        'secret_key': creds.secret_access_key,
        'token': creds.session_token,
        'expiry_time': expiry_time.isoformat(),
    }


def _build_pxt_store_entry(org: str, db: str, bucket: str, prefix: str) -> _PxtStoreCacheEntry:
    """Fetch credentials and build a boto3 session for the bucket."""
    creds = get_bucket_credentials(org, db, bucket, prefix)

    entry = _PxtStoreCacheEntry(
        client=None,
        resource=None,
        physical_bucket_name=creds.resolved_bucket_name,
        endpoint_url=creds.endpoint_url,
        no_space_left=creds.no_space_left,
        storage_provider=creds.storage_provider,
    )

    _handle_no_space_warning(creds.no_space_left, entry, org, db, bucket)

    # Build RefreshableCredentials from the initial fetch, reusing its result
    # to avoid a redundant control-plane call.
    expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)
    initial_metadata = {
        'access_key': creds.access_key_id,
        'secret_key': creds.secret_access_key,
        'token': creds.session_token,
        'expiry_time': expiry_time.isoformat(),
    }

    # keeps credentials fresh without triggering botocore's immediate-refresh behavior.
    refreshable_creds = RefreshableCredentials.create_from_metadata(
        metadata=initial_metadata,
        refresh_using=lambda: _refresh_credentials(org, db, bucket, prefix, entry),
        method='pxt-store',
        advisory_timeout=60,  # start refreshing 60s before expiry (non-blocking, best-effort)
        mandatory_timeout=30,  # block and force refresh if credentials expire within 30s
    )

    # Use a fresh botocore session with our credential provider as the sole entry in the
    # resolver chain, preventing fallthrough to env vars, config files, or instance metadata.
    botocore_session = get_botocore_session()
    resolver = CredentialResolver(providers=[_PxtStoreCredentialProvider(refreshable_creds)])
    botocore_session.register_component('credential_provider', resolver)
    boto3_session = boto3.Session(botocore_session=botocore_session)

    entry.client = boto3_session.client(
        's3',
        endpoint_url=creds.endpoint_url,
        region_name='auto',
        config=botocore.config.Config(
            max_pool_connections=30,
            connect_timeout=15,
            read_timeout=30,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            signature_version='s3v4',
            s3={'addressing_style': 'path'},
            user_agent_extra='pixeltable',
        ),
    )
    entry.resource = boto3_session.resource('s3', endpoint_url=creds.endpoint_url, region_name='auto')

    _logger.info(f'Initialized session for pxtfs://{org}:{db}/{bucket}')
    return entry


def _get_or_create_pxt_store_entry(org: str, db: str, bucket: str, prefix: str) -> _PxtStoreCacheEntry:
    """Return the cached entry for org:db:bucket:prefix"""
    cache_key = f'{org}:{db}:{bucket}:{prefix}'
    # pxt_store_client is thread-local (Runtime is thread-local), so no lock is needed here.
    pxt_store_client_dict = get_runtime().get_client('pxt_store')
    entry = pxt_store_client_dict.clients.get(cache_key)
    if entry is None:
        entry = _build_pxt_store_entry(org, db, bucket, prefix)
        pxt_store_client_dict.clients[cache_key] = entry
    return entry


class PxtStore(ObjectStoreBase):
    """Wraps a provider-specific store with control-plane credential refresh."""

    soa: StorageObjectAddress
    _pxt_store_entry: _PxtStoreCacheEntry
    _store: ObjectStoreBase  # underlying provider store (S3Store, AzureBlobStore, GCSStore, etc.)

    def __init__(self, soa: StorageObjectAddress) -> None:
        assert soa.storage_target == StorageTarget.PIXELTABLE_STORE

        self.soa = soa
        org, db, bucket, path = soa.account, soa.account_extension, soa.container, soa.prefix
        self._pxt_store_entry = _get_or_create_pxt_store_entry(org, db, bucket, path)
        physical_soa = soa._replace(container=self._pxt_store_entry.physical_bucket_name)
        self._store = self._build_store(physical_soa)

    def _build_store(self, soa: StorageObjectAddress) -> ObjectStoreBase:
        """Instantiate the correct underlying store based on the resolved provider."""
        assert self._pxt_store_entry.client is not None
        assert self._pxt_store_entry.resource is not None
        assert self._pxt_store_entry.storage_provider is not None

        try:
            storage_target = StorageTarget(self._pxt_store_entry.storage_provider)
        except ValueError:
            supported = ', '.join(t.name for t in StorageTarget)
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Invalid storage provider {self._pxt_store_entry.storage_provider!r}. Supported: {supported}',
            ) from None

        if storage_target not in S3_COMPATIBLE_TARGETS:
            supported = ', '.join(t.name for t in S3_COMPATIBLE_TARGETS)
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Storage target {storage_target.value!r} not supported. Supported: {supported}',
            )

        return S3Store(soa, client=self._pxt_store_entry.client, resource=self._pxt_store_entry.resource)

    def _to_logical_uri(self, physical_uri: str) -> str:
        """Create object uri with logical bucket name"""
        if not physical_uri.startswith('pxtfs://'):
            return physical_uri

        physical, logical = self._pxt_store_entry.physical_bucket_name, self.soa.container

        matched = _PXTFS_URI_PATTERN.match(physical_uri)
        assert matched, f'Unexpected pxtfs URI shape in {physical_uri!r}'

        bucket, path = matched.group(1), matched.group(2)
        if bucket == logical:
            return physical_uri
        assert bucket == physical, (
            f'Unexpected pxtfs bucket segment {bucket!r} in {physical_uri!r} (expected {physical!r} or {logical!r})'
        )
        org_db = physical_uri.split('/')[2]
        return f'pxtfs://{org_db}/{logical}{path}'

    def validate(self, error_col_name: str) -> str | None:
        """Probe the temp-credential-scoped prefix and return the logical base URI on success."""
        assert isinstance(self._store, S3Store)

        try:
            _ = self._store.list_objects(return_uri=False, n_max=1)
            return self._to_logical_uri(self._store.base_uri)
        except excs.ExternalServiceError:
            # Re-raise bucket specific errors (connection issues, permission denied, etc.)
            raise
        except excs.NotFoundError:
            # Handle not found gracefully
            _logger.warning(f'Bucket not found during validation for {error_col_name}')
            return None
        except Exception as e:
            # Catch any unexpected errors
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Unexpected error validating storage for {error_col_name}: {e}',
                provider=self._pxt_store_entry.storage_provider,
            ) from e

    def copy_local_file(self, col: Column, src_path: Path) -> str:
        if self._pxt_store_entry.no_space_left:
            raise excs.ServiceUnavailableError(
                ErrorCode.STORE_UNAVAILABLE,
                'No space left in Pixeltable store. Only read and delete operations are allowed.',
            )
        return self._to_logical_uri(self._store.copy_local_file(col, src_path))

    def copy_object_to_local_file(self, src_path: str, dest_path: Path) -> None:
        return self._store.copy_object_to_local_file(src_path, dest_path)

    def count(self, tbl_id: uuid.UUID, tbl_version: int | None = None) -> int:
        return self._store.count(tbl_id, tbl_version)

    def delete(self, tbl_id: uuid.UUID, tbl_version: int | None = None) -> int | None:
        return self._store.delete(tbl_id, tbl_version)

    def list_objects(self, return_uri: bool, n_max: int = 10) -> list[str]:
        results = self._store.list_objects(return_uri, n_max)
        if return_uri:
            return [self._to_logical_uri(r) for r in results]
        return results

    def create_presigned_url(self, soa: StorageObjectAddress, expiration_seconds: int) -> str:
        """Request a presigned GET URL from the control plane."""
        if not soa.has_object:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'StorageObjectAddress does not contain an object name: {soa}'
            )
        return get_presigned_url_from_cloud(
            org_slug=soa.account,
            db_slug=soa.account_extension or '',
            bucket=soa.container,
            key=soa.key,
            method='get',
            expiration=expiration_seconds,
        )
