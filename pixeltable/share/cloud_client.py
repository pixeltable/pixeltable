"""Pixeltable cloud control-plane API client.

All cloud API assembly lives here. Callers pass plain Python values;
this module constructs the protocol request objects and returns raw response dicts.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.share.protocol.api import (
    CreateDatabaseRequest,
    CreateServiceRequest,
    DeleteDatabaseRequest,
    DeleteServiceRequest,
    GetBundleUploadUrlRequest,
    GetDatabaseRequest,
    GetServiceRequest,
    ListDatabaseRequest,
    ListOrgsRequest,
    ListServicesRequest,
    StartDatabaseRequest,
    StartServiceRequest,
    StopDatabaseRequest,
    StopServiceRequest,
    UpdateDatabaseRequest,
    UpdateRuntimeRequest,
    UpdateServiceRequest,
)

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')

_LONG_OPS = frozenset({'create_database', 'update_runtime', 'delete_database'})


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
    """Forward one request to the cloud control-plane and return the raw response dict."""
    op = getattr(request, 'operation_type', None)
    op_str = op.value if hasattr(op, 'value') else str(op) if op else ''
    timeout = 180 if op_str in _LONG_OPS else 30
    resp = requests.post(PIXELTABLE_API_URL, data=request.model_dump_json(), headers=_api_headers(), timeout=timeout)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Control-plane error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()


def list_orgs() -> dict[str, Any]:
    return api_call(ListOrgsRequest())


def get_org(org_slug: str) -> dict[str, Any] | None:
    resp = api_call(ListOrgsRequest())
    return next((o for o in resp.get('orgs', []) if o.get('org_slug') == org_slug), None)


def list_dbs(org_slug: str) -> dict[str, Any]:
    return api_call(ListDatabaseRequest(org_slug=org_slug))


def create_db(org_slug: str, db_slug: str, location: str = 'aws', region: str = 'us-east-1') -> dict[str, Any]:
    return api_call(CreateDatabaseRequest(org_slug=org_slug, db_slug=db_slug, location=location, region=region))


def get_db(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(GetDatabaseRequest(org_slug=org_slug, db_slug=db_slug))


def delete_db(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(DeleteDatabaseRequest(org_slug=org_slug, db_slug=db_slug))


def start_db(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(StartDatabaseRequest(org_slug=org_slug, db_slug=db_slug))


def stop_db(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(StopDatabaseRequest(org_slug=org_slug, db_slug=db_slug))


def update_db(
    org_slug: str,
    db_slug: str,
    workers: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
) -> dict[str, Any]:
    return api_call(
        UpdateDatabaseRequest(
            org_slug=org_slug, db_slug=db_slug, workers=workers, cpu=cpu, memory_mb=memory_mb, disk_gb=disk_gb
        )
    )


def get_upload_url(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(GetBundleUploadUrlRequest(org_slug=org_slug, db_slug=db_slug))


def trigger_runtime_update(org_slug: str, db_slug: str, bundle_s3_key: str) -> dict[str, Any]:
    return api_call(UpdateRuntimeRequest(org_slug=org_slug, db_slug=db_slug, bundle_s3_key=bundle_s3_key))


def list_services(org_slug: str, db_slug: str) -> dict[str, Any]:
    return api_call(ListServicesRequest(org_slug=org_slug, db_slug=db_slug))


def create_service(
    org_slug: str,
    db_slug: str,
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
            org_slug=org_slug,
            db_slug=db_slug,
            service_name=service_name,
            base_path=base_path,
            workers_min=workers,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )


def get_service(org_slug: str, db_slug: str, svc_name: str) -> dict[str, Any]:
    return api_call(GetServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=svc_name))


def delete_service(org_slug: str, db_slug: str, svc_name: str) -> dict[str, Any]:
    return api_call(DeleteServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=svc_name))


def start_service(org_slug: str, db_slug: str, svc_name: str) -> dict[str, Any]:
    return api_call(StartServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=svc_name))


def stop_service(org_slug: str, db_slug: str, svc_name: str) -> dict[str, Any]:
    return api_call(StopServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=svc_name))


def update_service(
    org_slug: str,
    db_slug: str,
    svc_name: str,
    workers: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
    service_config: str | None = None,
) -> dict[str, Any]:
    return api_call(
        UpdateServiceRequest(
            org_slug=org_slug,
            db_slug=db_slug,
            service_name=svc_name,
            workers_min=workers,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )
