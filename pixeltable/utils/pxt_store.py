"""Pixeltable Cloud home storage (``pxt://org:db/home/...``) backed by S3-compatible APIs."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import boto3
import botocore
import botocore.credentials
import botocore.session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session as get_botocore_session

from pixeltable import env, exceptions as excs
from pixeltable.runtime import get_runtime
from pixeltable.utils.object_stores import StorageObjectAddress, StorageTarget
from pixeltable.utils.s3_store import S3CompatClientDict, S3Store

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


@dataclass
class _PxtHomeCacheEntry:
    """Cached boto3 client/resource and quota state for a home bucket."""

    client: Any
    resource: Any
    bucket_name: str
    endpoint_url: str
    no_space_left: bool = False


# pxt_home clients are thread-local via Runtime._clients, matching r2/s3/b2/tigris.
@env.register_client('pxt_home')
def _() -> Any:
    return S3CompatClientDict(profile=None, clients={})


class _PxtHomeCredentialProvider(botocore.credentials.CredentialProvider):
    """Supplies RefreshableCredentials to a botocore session via the CredentialResolver chain."""

    METHOD = 'pxt-home-bucket'
    CANONICAL_NAME = 'custom-pxt-home-bucket'

    def __init__(self, creds: RefreshableCredentials) -> None:
        self._creds = creds

    def load(self) -> RefreshableCredentials:
        return self._creds


def _warn_no_space_left(org: str, db: str) -> None:
    warnings.warn(
        f'Pixeltable store for {org}:{db} has no space left. '
        'Only read and delete operations are allowed. '
        'Please delete unused data or contact support to increase your storage limit.',
        category=excs.PixeltableWarning,
        stacklevel=3,
    )


def _make_credential_refresher(
    org: str, db: str, entry: _PxtHomeCacheEntry
) -> Callable[[], dict[str, str]]:
    """Return a credential refresher for use with RefreshableCredentials.

    Updates the cache entry's no_space_left and bucket_name on each refresh.
    """

    def _refresh() -> dict[str, str]:
        from pixeltable.utils.cloud_utils import get_home_bucket_credentials

        creds = get_home_bucket_credentials(org, db)
        expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)

        entry.no_space_left = creds.no_space_left
        if creds.bucket_name:
            entry.bucket_name = creds.bucket_name
        if creds.no_space_left:
            _warn_no_space_left(org, db)

        _logger.info(
            'Refreshed home bucket credentials for %s:%s (ttl=%ds, no_space_left=%s)',
            org, db, creds.ttl_seconds, creds.no_space_left,
        )
        return {
            'access_key': creds.access_key_id,
            'secret_key': creds.secret_access_key,
            'token': creds.session_token,
            'expiry_time': expiry_time.isoformat(),
        }

    return _refresh


def _build_pxt_home_entry(org: str, db: str) -> _PxtHomeCacheEntry:
    """Fetch credentials and build a boto3 session for the home bucket."""
    from pixeltable.utils.cloud_utils import get_home_bucket_credentials

    creds = get_home_bucket_credentials(org, db)
    if not creds.bucket_name:
        raise excs.Error(
            f'Pixeltable cloud returned an empty home bucket name for {org}:{db}. '
            'Ensure the control plane get_home_bucket_credentials response includes bucket_name.'
        )

    entry = _PxtHomeCacheEntry(
        client=None,
        resource=None,
        bucket_name=creds.bucket_name,
        endpoint_url=creds.endpoint_url,
        no_space_left=creds.no_space_left,
    )

    if creds.no_space_left:
        _warn_no_space_left(org, db)

    # Build RefreshableCredentials from the initial fetch, reusing its result
    # to avoid a redundant control-plane call.
    expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)
    initial_metadata = {
        'access_key': creds.access_key_id,
        'secret_key': creds.secret_access_key,
        'token': creds.session_token,
        'expiry_time': expiry_time.isoformat(),
    }
    refresher = _make_credential_refresher(org, db, entry)

    # Refresh well before expiry: control plane issues 900s TTLs, so advisory=60s / mandatory=30s
    # keeps credentials fresh without triggering botocore's immediate-refresh behaviour.
    refreshable_creds = RefreshableCredentials.create_from_metadata(
        metadata=initial_metadata,
        refresh_using=refresher,
        method='pxt-home-bucket',
        advisory_timeout=60,
        mandatory_timeout=30,
    )

    # Register the refreshable credentials as the first provider in the resolver chain,
    # so the boto3 session uses them without falling through to env vars or config files.
    botocore_session = get_botocore_session()
    provider = _PxtHomeCredentialProvider(refreshable_creds)
    botocore_session._components.get_component('credential_provider').insert_before('env', provider)
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
    entry.resource = boto3_session.resource(
        's3',
        endpoint_url=creds.endpoint_url,
        region_name='auto',
    )

    _logger.info('Created home bucket session for %s:%s', org, db)
    return entry


def _get_or_create_pxt_home_entry(org: str, db: str) -> _PxtHomeCacheEntry:
    """Return the home bucket entry for org:db, building it on first use per thread."""
    cache_key = f'{org}:{db}'
    # cd.clients is thread-local (Runtime is thread-local), so no lock is needed here.
    cd = get_runtime().get_client('pxt_home')
    entry = cd.clients.get(cache_key)
    if entry is None:
        entry = _build_pxt_home_entry(org, db)
        cd.clients[cache_key] = entry
    return entry


class PxtStore(S3Store):
    """Home bucket store via per-thread auto-refreshing credentials."""

    _pxt_home_entry: _PxtHomeCacheEntry

    def __init__(self, soa: StorageObjectAddress) -> None:
        if soa.storage_target != StorageTarget.PIXELTABLE_STORE:
            raise AssertionError(f'PxtStore requires PIXELTABLE_STORE, got {soa.storage_target}')
        org, db = soa.account, soa.account_extension
        self._pxt_home_entry = _get_or_create_pxt_home_entry(org, db)
        super().__init__(soa, resolved_physical_bucket_name=self._pxt_home_entry.bucket_name)

    @property
    def bucket_name(self) -> str:
        """Physical bucket name (updated when credentials refresh)."""
        return self._pxt_home_entry.bucket_name

    def client(self) -> Any:
        return self._pxt_home_entry.client

    def get_resource(self) -> Any:
        return self._pxt_home_entry.resource

    def copy_local_file(self, col: 'Column', src_path: Path) -> str:
        if self._pxt_home_entry.no_space_left:
            raise excs.Error('No space left in Pixeltable store. Only read and delete operations are allowed.')
        return super().copy_local_file(col, src_path)

    def create_presigned_url(self, soa: StorageObjectAddress, expiration_seconds: int) -> str:
        """Request a presigned GET URL from the control plane (lifetime independent of temp credentials)."""
        if not soa.has_object:
            raise excs.Error(f'StorageObjectAddress does not contain an object name: {soa}')
        from pixeltable.utils.cloud_utils import get_presigned_url_from_cloud

        return get_presigned_url_from_cloud(
            org_slug=soa.account,
            db_slug=soa.account_extension or '',
            key=soa.key,
            method='get',
            expiration=expiration_seconds,
        )
