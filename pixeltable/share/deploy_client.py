"""Client for the Pixeltable cloud control-plane API."""

from __future__ import annotations

import os
from typing import Any

import requests

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.share.protocol.database import (
    DeleteSecretRequest,
    ListSecretsRequest,
    SetSecretRequest,
)
from pixeltable.share.protocol.service import (
    CreateDatabaseRequest,
    CreateServiceRequest,
    DeleteDatabaseRequest,
    DeleteServiceRequest,
    GetDatabaseRequest,
    GetServiceRequest,
    ListDatabasesRequest,
    ListOrgsRequest,
    ListServicesRequest,
    StartDatabaseRequest,
    StartServiceRequest,
    StopDatabaseRequest,
    StopServiceRequest,
    UpdateRuntimeRequest,
)

PIXELTABLE_API_URL = os.environ.get('PIXELTABLE_API_URL', 'https://internal-api.pixeltable.com')


def _api_headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    api_key = Env.get().pxt_api_key
    if api_key is None:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            'A Pixeltable API key is required. '
            'Set it with `os.environ["PIXELTABLE_API_KEY"] = "your-key"`, '
            f'or add `api_key = "your-key"` to the `[pixeltable]` section in {Config.get().config_file}.',
        )
    headers['X-api-key'] = api_key
    return headers


def _post(request: Any) -> dict[str, Any]:
    resp = requests.post(PIXELTABLE_API_URL, data=request.model_dump_json(), headers=_api_headers(), timeout=30)
    if resp.status_code not in (200, 201):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Control-plane error {resp.status_code}: {resp.text}',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    return resp.json()


# ── Database ──────────────────────────────────────────────────────────────────


def database_create(
    org_slug: str, db_name: str, location: str = 'aws', region: str = 'us-east-1', json_output: bool = False
) -> dict[str, Any]:
    resp = _post(CreateDatabaseRequest(org_slug=org_slug, db_name=db_name, location=location, region=region))
    db = resp['database']
    if json_output:
        import json

        print(json.dumps(db))
    else:
        _print_db(db)
    return db


def database_get(org_slug: str, db_name: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(GetDatabaseRequest(org_slug=org_slug, db_name=db_name))
    db = resp['database']
    if json_output:
        import json

        print(json.dumps(db))
    else:
        _print_db(db)
    return db


def database_list(org_slug: str, json_output: bool = False) -> list[dict[str, Any]]:
    resp = _post(ListDatabasesRequest(org_slug=org_slug))
    dbs = resp.get('databases', [])
    if json_output:
        import json

        print(json.dumps(dbs))
    elif not dbs:
        print('No databases.')
    else:
        for db in dbs:
            _print_db(db)
    return dbs


def database_delete(org_slug: str, db_name: str, json_output: bool = False) -> None:
    _post(DeleteDatabaseRequest(org_slug=org_slug, db_name=db_name))
    if json_output:
        import json

        print(json.dumps({'deleted': db_name}))
    else:
        print(f"Deleted database '{db_name}'.")


def database_start(db_slug: str, org_slug: str | None = None, json_output: bool = False) -> None:
    resp = _post(StartDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
    if json_output:
        import json

        print(json.dumps(resp))
    else:
        print(resp.get('message', 'started'))


def database_stop(db_slug: str, org_slug: str | None = None, json_output: bool = False) -> None:
    resp = _post(StopDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
    if json_output:
        import json

        print(json.dumps(resp))
    else:
        print(resp.get('message', 'stopped'))


def database_update_runtime(
    db_slug: str, org_slug: str | None = None, runtime_image: str | None = None, json_output: bool = False
) -> None:
    resp = _post(UpdateRuntimeRequest(org_slug=org_slug, db_slug=db_slug, runtime_image=runtime_image))
    if json_output:
        import json

        print(json.dumps(resp))
    else:
        print(resp.get('message', 'runtime update triggered'))


# ── Org ───────────────────────────────────────────────────────────────────────


def org_list(json_output: bool = False) -> list[dict]:
    resp = _post(ListOrgsRequest())
    orgs = resp.get('orgs', [])
    if json_output:
        import json

        print(json.dumps(orgs))
    elif not orgs:
        print('No orgs.')
    else:
        for org in orgs:
            slug = org.get('org_slug', '')
            org_id = org.get('org_id', '')
            default_db = org.get('default_db_slug') or ''
            line = f'{slug}  id={org_id}'
            if default_db:
                line += f'  default_db={default_db}'
            print(line)
    return orgs


# ── Service ───────────────────────────────────────────────────────────────────


def service_create(
    org_slug: str,
    db_name: str,
    service_name: str,
    table_path: str,
    workers_min: int = 1,
    workers_max: int = 1,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(
        CreateServiceRequest(
            org_slug=org_slug,
            db_name=db_name,
            service_name=service_name,
            table_path=table_path,
            workers_min=workers_min,
            workers_max=workers_max,
        )
    )
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_get(org_slug: str, db_name: str, service_name: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(GetServiceRequest(org_slug=org_slug, db_name=db_name, service_name=service_name))
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_list(org_slug: str, db_name: str, json_output: bool = False) -> list[dict[str, Any]]:
    resp = _post(ListServicesRequest(org_slug=org_slug, db_name=db_name))
    svcs = resp.get('services', [])
    if json_output:
        import json

        print(json.dumps(svcs))
    elif not svcs:
        print(f"No services in database '{db_name}'.")
    else:
        for svc in svcs:
            _print_service(svc)
    return svcs


def service_start(
    org_slug: str,
    db_name: str,
    service_name: str,
    workers_min: int | None = None,
    workers_max: int | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(
        StartServiceRequest(
            org_slug=org_slug,
            db_name=db_name,
            service_name=service_name,
            workers_min=workers_min,
            workers_max=workers_max,
        )
    )
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_stop(org_slug: str, db_name: str, service_name: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(StopServiceRequest(org_slug=org_slug, db_name=db_name, service_name=service_name))
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_delete(org_slug: str, db_name: str, service_name: str, json_output: bool = False) -> None:
    _post(DeleteServiceRequest(org_slug=org_slug, db_name=db_name, service_name=service_name))
    if json_output:
        import json

        print(json.dumps({'deleted': service_name}))
    else:
        print(f"Deleted service '{service_name}'.")


# ── Secrets ───────────────────────────────────────────────────────────────────


def secret_set(org_slug: str, db_slug: str | None, key: str, value: str, json_output: bool = False) -> None:
    _post(SetSecretRequest(org_slug=org_slug, db_slug=db_slug, key=key, value=value))
    if json_output:
        import json

        print(json.dumps({'key': key}))
    else:
        print(f"Secret '{key}' set.")


def secret_delete(org_slug: str, db_slug: str | None, key: str, json_output: bool = False) -> None:
    _post(DeleteSecretRequest(org_slug=org_slug, db_slug=db_slug, key=key))
    if json_output:
        import json

        print(json.dumps({'deleted': key}))
    else:
        print(f"Secret '{key}' deleted.")


def secret_list(org_slug: str, db_slug: str | None, json_output: bool = False) -> list[str]:
    resp = _post(ListSecretsRequest(org_slug=org_slug, db_slug=db_slug))
    keys = resp.get('keys', [])
    if json_output:
        import json

        print(json.dumps(keys))
    elif not keys:
        print('No secrets.')
    else:
        for k in keys:
            print(f'  {k}')
    return keys


# ── Display helpers ───────────────────────────────────────────────────────────


def _print_db(db: dict[str, Any]) -> None:
    name = db.get('db_name', '')
    state = db.get('state', '')
    location = db.get('location', '')
    region = db.get('region', '')
    endpoint = db.get('endpoint') or ''
    print(f'{name}  state={state}  {location}/{region}  {endpoint}'.rstrip())


def _print_service(svc: dict[str, Any]) -> None:
    name = svc.get('service_name', '')
    state = svc.get('state', '')
    table = svc.get('table_path', '')
    workers = f'workers={svc.get("workers_min")}-{svc.get("workers_max")}'
    endpoint = svc.get('endpoint') or ''
    print(f'{name}  state={state}  table={table}  {workers}  {endpoint}'.rstrip())
