"""Client for the Pixeltable cloud management API.

All management API assembly lives here. Callers pass plain Python values;
this module constructs the protocol request objects and returns raw response dicts.
"""

from __future__ import annotations

import http.cookiejar
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.service.management_protocol import (
    CreateDbRequest,
    CreateServiceRequest,
    DeleteDbRequest,
    DeleteServiceRequest,
    GetBundleUploadUrlRequest,
    GetDbRequest,
    GetServiceRequest,
    ListDbRequest,
    ListOrgsRequest,
    ListServicesRequest,
    StartDbRequest,
    StartServiceRequest,
    StopDbRequest,
    StopServiceRequest,
    UpdateDbRequest,
    UpdateRuntimeRequest,
    UpdateServiceRequest,
)

_DEFAULT_API_URL = 'https://internal-api.pixeltable.com'


def api_url() -> str:
    """URL of the Pixeltable cloud management API."""
    # api_url is deliberately not a registered config option: PIXELTABLE_API_URL is the only override
    url = Config.get().get_string_value('api_url')
    return _DEFAULT_API_URL if url is None else url


_LONG_OPS = frozenset({'create_db', 'update_runtime', 'delete_db'})

# operations that don't change server state; can be sent multiple times
_READ_OPS = frozenset({'list_orgs', 'list_dbs', 'get_db', 'list_services', 'get_service'})

# maximum number of connections kept open to the service, sized for concurrent calls from multiple threads
_POOL_MAXSIZE = 16


def _new_session() -> requests.Session:
    """Create a pooled session for management API requests."""
    session = requests.Session()
    # the management API sets no cookies; blocking the jar leaves the session without mutable state, so
    # concurrent calls can share it
    session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    # retry only failures to establish a connection: those never reached the server. Read and status
    # retries stay off because every operation is a POST, and replaying one that the server may have
    # already processed could apply it twice.
    retries = Retry(total=2, connect=2, read=0, status=0, other=0, allowed_methods=frozenset(), backoff_factor=0.2)
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=_POOL_MAXSIZE, max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


_SESSION = _new_session()


def _api_headers() -> dict[str, str]:
    api_key = Env.get().pxt_api_key
    if api_key is None:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            'A Pixeltable API key is required. '
            'Set it with `os.environ["PIXELTABLE_API_KEY"] = "your-key"`, '
            'or add `api_key = "your-key"` to the `[pixeltable]` section in your Pixeltable config file.',
        )
    return {'Content-Type': 'application/json', 'X-api-key': api_key}


def api_call(request: Any) -> dict[str, Any]:
    """Forward one request to the cloud management API and return the raw response dict."""
    op = getattr(request, 'operation_type', None)
    op_str = op.value if hasattr(op, 'value') else str(op) if op else ''
    timeout = 180 if op_str in _LONG_OPS else 30
    body = request.model_dump_json()
    try:
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    except requests.exceptions.ConnectionError:
        # a pooled connection closed by the peer while idle fails the call that next picks it up.
        # Retrying gets a new connection, but is only safe for operations that a second delivery
        # cannot change.
        if op_str not in _READ_OPS:
            raise
        resp = _SESSION.post(api_url(), data=body, headers=_api_headers(), timeout=timeout)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Management API error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()


def list_orgs() -> dict[str, Any]:
    return api_call(ListOrgsRequest())


def get_org(org: str) -> dict[str, Any] | None:
    resp = api_call(ListOrgsRequest())
    return next((o for o in resp.get('orgs', []) if o.get('org') == org), None)


def list_dbs(org: str) -> dict[str, Any]:
    return api_call(ListDbRequest(org=org))


def create_db(org: str, db: str, location: str = 'aws', region: str = 'us-east-1') -> dict[str, Any]:
    return api_call(CreateDbRequest(org=org, db=db, location=location, region=region))


def get_db(org: str, db: str) -> dict[str, Any]:
    return api_call(GetDbRequest(org=org, db=db))


def delete_db(org: str, db: str) -> dict[str, Any]:
    return api_call(DeleteDbRequest(org=org, db=db))


def start_db(org: str, db: str) -> dict[str, Any]:
    return api_call(StartDbRequest(org=org, db=db))


def stop_db(org: str, db: str) -> dict[str, Any]:
    return api_call(StopDbRequest(org=org, db=db))


def update_db(
    org: str,
    db: str,
    workers: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
) -> dict[str, Any]:
    return api_call(UpdateDbRequest(org=org, db=db, workers=workers, cpu=cpu, memory_mb=memory_mb, disk_gb=disk_gb))


def get_upload_url(org: str, db: str) -> dict[str, Any]:
    return api_call(GetBundleUploadUrlRequest(org=org, db=db))


def trigger_runtime_update(org: str, db: str, bundle_s3_key: str) -> dict[str, Any]:
    return api_call(UpdateRuntimeRequest(org=org, db=db, bundle_s3_key=bundle_s3_key))


def list_services(org: str, db: str) -> dict[str, Any]:
    return api_call(ListServicesRequest(org=org, db=db))


def create_service(
    org: str,
    db: str,
    service_name: str,
    base_path: str = '',
    workers: int = 1,
    cpu: float = 0.5,
    memory_mb: int = 512,
    disk_gb: int = 10,
    service_config: str | None = None,
) -> dict[str, Any]:
    return api_call(
        CreateServiceRequest(
            org=org,
            db=db,
            service_name=service_name,
            base_path=base_path,
            workers_min=workers,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )


def get_service(org: str, db: str, svc_name: str) -> dict[str, Any]:
    return api_call(GetServiceRequest(org=org, db=db, service_name=svc_name))


def delete_service(org: str, db: str, svc_name: str) -> dict[str, Any]:
    return api_call(DeleteServiceRequest(org=org, db=db, service_name=svc_name))


def start_service(org: str, db: str, svc_name: str) -> dict[str, Any]:
    return api_call(StartServiceRequest(org=org, db=db, service_name=svc_name))


def stop_service(org: str, db: str, svc_name: str) -> dict[str, Any]:
    return api_call(StopServiceRequest(org=org, db=db, service_name=svc_name))


def update_service(
    org: str,
    db: str,
    svc_name: str,
    workers: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
    service_config: str | None = None,
) -> dict[str, Any]:
    return api_call(
        UpdateServiceRequest(
            org=org,
            db=db,
            service_name=svc_name,
            workers_min=workers,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )
