"""Pixeltable Cloud home storage (``pxt://org:db/home/...``) backed by S3-compatible APIs."""

from __future__ import annotations

import logging
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

from pixeltable import env, exceptions as excs
from pixeltable.runtime import get_runtime
from pixeltable.utils.cloud_utils import get_home_bucket_credentials, get_presigned_url_from_cloud
from pixeltable.utils.object_stores import StorageObjectAddress, StorageTarget
from pixeltable.utils.s3_store import S3CompatClientDict, S3Store

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


@dataclass
class _PxtStoreCacheEntry:
    """Cached boto3 client/resource and quota state for a home bucket."""

    client: BaseClient | None  # populated after boto3 session is built
    resource: ServiceResource | None  # populated after boto3 session is built
    bucket_name: str
    endpoint_url: str
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


def _handle_no_space_warning(no_space_left: bool, entry: _PxtStoreCacheEntry, org: str, db: str) -> None:
    if no_space_left:
        if entry.no_space_warned:
            return
        warnings.warn(
            f'Pixeltable store for pxt://{org}:{db}/home has no space left. '
            'Only read and delete operations are allowed.',
            category=excs.PixeltableWarning,
            stacklevel=3,
        )
        entry.no_space_warned = True
    else:
        entry.no_space_warned = False


def _refresh_credentials(org: str, db: str, entry: _PxtStoreCacheEntry) -> dict[str, str]:
    """Fetch fresh credentials and update the cache entry"""
    creds = get_home_bucket_credentials(org, db)
    expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)

    entry.no_space_left = creds.no_space_left
    if creds.bucket_name:
        entry.bucket_name = creds.bucket_name

    _handle_no_space_warning(creds.no_space_left, entry, org, db)

    _logger.info(
        'Refreshed home bucket credentials for %s:%s (ttl=%ds, no_space_left=%s)',
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


def _build_pxt_store_entry(org: str, db: str) -> _PxtStoreCacheEntry:
    """Fetch credentials and build a boto3 session for the home bucket."""
    creds = get_home_bucket_credentials(org, db)

    entry = _PxtStoreCacheEntry(
        client=None,
        resource=None,
        bucket_name=creds.bucket_name,
        endpoint_url=creds.endpoint_url,
        no_space_left=creds.no_space_left,
    )

    _handle_no_space_warning(creds.no_space_left, entry, org, db)

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
        refresh_using=lambda: _refresh_credentials(org, db, entry),
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

    _logger.info('Initialized home bucket session for %s:%s', org, db)
    return entry


def _get_or_create_pxt_store_entry(org: str, db: str) -> _PxtStoreCacheEntry:
    """Return the home bucket entry for org:db, building it on first use per thread."""
    cache_key = f'{org}:{db}'
    # pxt_store_client is thread-local (Runtime is thread-local), so no lock is needed here.
    pxt_store_client_dict = get_runtime().get_client('pxt_store')
    entry = pxt_store_client_dict.clients.get(cache_key)
    if entry is None:
        entry = _build_pxt_store_entry(org, db)
        pxt_store_client_dict.clients[cache_key] = entry
    return entry


class PxtStore(S3Store):
    """Home bucket store via per-thread auto-refreshing credentials."""

    _pxt_store_entry: _PxtStoreCacheEntry

    def __init__(self, soa: StorageObjectAddress) -> None:
        if soa.storage_target != StorageTarget.PIXELTABLE_STORE:
            raise excs.Error(
                f'Invalid storage target for PxtStore: expected PIXELTABLE_STORE, got {soa.storage_target!s}.'
            )
        org, db = soa.account, soa.account_extension
        self._pxt_store_entry = _get_or_create_pxt_store_entry(org, db)
        super().__init__(soa._replace(container=self._pxt_store_entry.bucket_name))

    @property
    def bucket_name(self) -> str:
        """Physical bucket name (updated when credentials refresh)."""
        return self._pxt_store_entry.bucket_name

    def client(self) -> BaseClient:
        assert self._pxt_store_entry.client is not None
        return self._pxt_store_entry.client

    def get_resource(self) -> ServiceResource:
        assert self._pxt_store_entry.resource is not None
        return self._pxt_store_entry.resource

    def copy_local_file(self, col: 'Column', src_path: Path) -> str:
        if self._pxt_store_entry.no_space_left:
            raise excs.Error('No space left in Pixeltable store. Only read and delete operations are allowed.')
        return super().copy_local_file(col, src_path)

    def create_presigned_url(self, soa: StorageObjectAddress, expiration_seconds: int) -> str:
        """Request a presigned GET URL from the control plane (lifetime independent of temp credentials)."""
        if not soa.has_object:
            raise excs.Error(f'StorageObjectAddress does not contain an object name: {soa}')

        return get_presigned_url_from_cloud(
            org_slug=soa.account,
            db_slug=soa.account_extension or '',
            key=soa.key,
            method='get',
            expiration=expiration_seconds,
        )
