"""Pixeltable Cloud home storage (``pxt://org:db/home/...``) backed by S3-compatible APIs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
import botocore
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session as get_botocore_session

from pixeltable import env, exceptions as excs
from pixeltable.runtime import get_runtime
from pixeltable.utils.object_stores import StorageObjectAddress, StorageTarget
from pixeltable.utils.s3_store import S3CompatClientDict, S3Store, client_lock

if TYPE_CHECKING:
    from pixeltable.catalog import Column

_logger = logging.getLogger('pixeltable')


@dataclass
class _PxtHomeCacheEntry:
    """Cached boto3 client/resource for a home bucket with auto-refreshing credentials."""

    client: Any
    resource: Any
    bucket_name: str
    endpoint_url: str
    no_space_left: bool = False


# Share cached home-bucket sessions across threads. `Runtime` is thread-local, so without this,
# tests that mutate `cd.clients[...]` (e.g. `entry.no_space_left = True`) wouldn't affect the
# store used by `insert()` if it runs in a different thread.
_PXT_HOME_CLIENTS: dict[str, _PxtHomeCacheEntry] = {}


@env.register_client('pxt_home')
def _() -> Any:
    return S3CompatClientDict(profile=None, clients=_PXT_HOME_CLIENTS)


def _create_pxt_credential_refresher(org: str, db: str, entry: _PxtHomeCacheEntry):
    """Return a closure that fetches fresh home bucket credentials in the format expected by RefreshableCredentials.

    Also updates the cache entry's no_space_left flag on each refresh, so the quota
    state stays current without requiring a new client.
    """

    def _refresh():
        import warnings

        from pixeltable.utils.cloud_utils import get_home_bucket_credentials

        creds = get_home_bucket_credentials(org, db)
        expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)

        entry.no_space_left = creds.no_space_left
        if creds.bucket_name:
            entry.bucket_name = creds.bucket_name
        if creds.no_space_left:
            warnings.warn(
                f'Pixeltable store for {org}:{db} has no space left. '
                f'Only read and delete operations are allowed. '
                f'Please delete unused data or contact support to increase your storage limit.',
                category=excs.PixeltableWarning,
                stacklevel=2,
            )

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

    return _refresh


def _get_or_create_pxt_home_entry(org: str, db: str) -> _PxtHomeCacheEntry:
    """Resolve home bucket credentials and boto session (cached per org:db)."""
    cache_key = f'{org}:{db}'
    cd = get_runtime().get_client('pxt_home')
    with client_lock:
        entry = cd.clients.get(cache_key)
        if entry is not None:
            return entry

    from pixeltable.utils.cloud_utils import get_home_bucket_credentials

    creds = get_home_bucket_credentials(org, db)
    if not creds.bucket_name:
        raise excs.Error(
            f'Pixeltable cloud returned an empty home bucket name for {org}:{db}. '
            'Ensure the control plane get_home_bucket_credentials response includes bucket_name.'
        )
    expiry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=creds.ttl_seconds)
    initial_metadata = {
        'access_key': creds.access_key_id,
        'secret_key': creds.secret_access_key,
        'token': creds.session_token,
        'expiry_time': expiry_time.isoformat(),
    }

    new_entry = _PxtHomeCacheEntry(
        client=None,
        resource=None,
        bucket_name=creds.bucket_name,
        endpoint_url=creds.endpoint_url,
        no_space_left=creds.no_space_left,
    )

    if creds.no_space_left:
        import warnings

        warnings.warn(
            f'Pixeltable store for {org}:{db} has no space left. Only read and delete operations are allowed. ',
            category=excs.PixeltableWarning,
            stacklevel=3,
        )

    refresher = _create_pxt_credential_refresher(org, db, new_entry)
    refreshable_creds = RefreshableCredentials.create_from_metadata(
        metadata=initial_metadata,
        refresh_using=refresher,
        method='pxt-home-bucket',
        # botocore defaults to advisory_timeout=900s, mandatory_timeout=600s.
        # Our control plane returns ttl_seconds=900s, which makes the credentials look
        # "within refresh window" immediately, causing frequent refreshes.
        advisory_timeout=60,
        mandatory_timeout=30,
    )

    botocore_session = get_botocore_session()
    botocore_session._credentials = refreshable_creds
    boto3_session = boto3.Session(botocore_session=botocore_session)

    new_entry.client = boto3_session.client(
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
    new_entry.resource = boto3_session.resource('s3', endpoint_url=creds.endpoint_url, region_name='auto')

    with client_lock:
        cd.clients[cache_key] = new_entry

    _logger.info('Created auto-refreshing home bucket session for %s:%s', org, db)
    return new_entry


class PxtStore(S3Store):
    """Home bucket via cloud temp credentials."""

    _pxt_home_entry: _PxtHomeCacheEntry

    def __init__(self, soa: StorageObjectAddress) -> None:
        if soa.storage_target != StorageTarget.PIXELTABLE_STORE:
            raise AssertionError(f'PxtStore requires PIXELTABLE_STORE, got {soa.storage_target}')
        org, db = soa.account, soa.account_extension
        self._pxt_home_entry = _get_or_create_pxt_home_entry(org, db)
        super().__init__(soa, resolved_physical_bucket_name=self._pxt_home_entry.bucket_name)

    @property
    def bucket_name(self) -> str:
        """Physical R2 bucket from the cloud (updated when temp credentials refresh)."""
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
        """Presigned GET URLs from the control plane (not tied to temp credential TTL)."""
        if not soa.has_object:
            raise excs.Error(f'StorageObjectAddress does not contain an object name: {soa}')
        from pixeltable.utils.cloud_utils import get_presigned_url_from_cloud

        org = soa.account
        db = soa.account_extension or ''
        return get_presigned_url_from_cloud(
            org_slug=org, db_slug=db, key=soa.key, method='get', expiration=expiration_seconds
        )
