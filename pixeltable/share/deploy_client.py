"""Client for the Pixeltable cloud control-plane API."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.share.protocol.database import (
    CreateDatabaseRequest,
    DeleteDatabaseRequest,
    DeleteSecretRequest,
    GetDatabaseRequest,
    ListDatabaseRequest,
    ListSecretsRequest,
    SetSecretRequest,
    StartDatabaseRequest,
    StopDatabaseRequest,
    UpdateDatabaseRequest,
    UpdateRuntimeRequest,
)
from pixeltable.share.protocol.service import (
    CreateServiceRequest,
    DeleteServiceRequest,
    GetBundleUploadUrlRequest,
    GetServiceRequest,
    ListOrgsRequest,
    ListServicesRequest,
    StartServiceRequest,
    StopServiceRequest,
    UpdateServiceRequest,
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


_LONG_OPS = frozenset({'create_database', 'update_runtime', 'delete_database'})


def _post(request: Any) -> dict[str, Any]:
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


_PROVISIONING_POLL_INTERVAL = 5  # seconds between get_database polls
_PROVISIONING_TIMEOUT = 600  # 10 minutes max wait
_UPDATE_TIMEOUT = 360  # 6 minutes max wait for update/start/stop


def _poll_db(
    org_slug: str,
    db_slug: str,
    pending_states: frozenset[str],
    timeout: float,
    label: str,
) -> dict[str, Any]:
    """Poll get_database until state leaves pending_states or timeout expires.

    Shows a rich spinner with elapsed time in interactive mode. Returns the
    last-seen database dict regardless of final state.
    """
    db: dict[str, Any] = {}
    deadline = time.monotonic() + timeout
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        TimeElapsedColumn(),
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        progress.add_task(label, total=None)
        while time.monotonic() < deadline:
            time.sleep(_PROVISIONING_POLL_INTERVAL)
            try:
                resp = _post(GetDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
                db = resp['database']
            except Exception:
                pass
            if db.get('state') not in pending_states:
                break
    return db


def database_create(
    org_slug: str, db_slug: str, location: str = 'aws', region: str = 'us-east-1', json_output: bool = False
) -> dict[str, Any]:
    resp = _post(CreateDatabaseRequest(org_slug=org_slug, db_slug=db_slug, location=location))
    db = resp['database']

    if db.get('state') == 'PROVISIONING' and not json_output:
        db = _poll_db(
            org_slug, db_slug,
            frozenset({'PROVISIONING'}),
            _PROVISIONING_TIMEOUT,
            f"Database '{db_slug}' is provisioning...",
        )

    if json_output:
        import json as _json

        print(_json.dumps(db))
    else:
        _print_db(db)
    return db


def database_get(org_slug: str, db_slug: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(GetDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
    db = resp['database']
    if json_output:
        import json

        print(json.dumps(db))
    else:
        _print_db(db)
    return db


def database_list(org_slug: str, json_output: bool = False) -> list[dict[str, Any]]:
    resp = _post(ListDatabaseRequest(org_slug=org_slug))
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


def database_delete(org_slug: str, db_slug: str, json_output: bool = False) -> None:
    _post(DeleteDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
    if json_output:
        import json

        print(json.dumps({'deleted': db_slug}))
    else:
        print(f"Deleted database '{db_slug}'.")


def database_update(
    org_slug: str,
    db_slug: str,
    workers: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(
        UpdateDatabaseRequest(
            org_slug=org_slug,
            db_slug=db_slug,
            workers=workers,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
        )
    )
    db = resp['database']
    if db.get('state') == 'UPDATING' and not json_output:
        db = _poll_db(
            org_slug, db_slug,
            frozenset({'UPDATING'}),
            _UPDATE_TIMEOUT,
            f"Database '{db_slug}' is updating...",
        )

    if json_output:
        import json

        print(json.dumps(db))
    else:
        _print_db(db)
    return db


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


def get_bundle_upload_url(org_slug: str | None, db_slug: str) -> tuple[str, str]:
    """Get a presigned S3 URL for a runtime bundle upload.

    Returns (presigned_url, bundle_s3_key).
    """
    resp = _post(GetBundleUploadUrlRequest(org_slug=org_slug, db_slug=db_slug))
    return resp['presigned_url'], resp['bundle_s3_key']


_RUNTIME_UPDATE_POLL_INTERVAL = 10  # seconds
_RUNTIME_UPDATE_TIMEOUT = 900  # 15 minutes max (CodeBuild can take up to ~8 min)


def database_update_runtime(
    db_slug: str, org_slug: str | None = None, watch: bool = False, json_output: bool = False
) -> None:
    import json as _json

    from pixeltable.serving.deploy import build_runtime_bundle

    if not json_output:
        print('Building runtime bundle...', end=' ', flush=True)
    bundle_path = build_runtime_bundle()
    if not json_output:
        size_mb = bundle_path.stat().st_size / (1024 * 1024)
        print(f'done ({size_mb:.1f} MB)')

    try:
        if not json_output:
            print('Uploading bundle...', end=' ', flush=True)
        presigned_url, bundle_s3_key = get_bundle_upload_url(org_slug, db_slug)
        with bundle_path.open('rb') as fh:
            upload_resp = requests.put(presigned_url, data=fh, timeout=300)
            upload_resp.raise_for_status()
        if not json_output:
            print('done')
    finally:
        bundle_path.unlink(missing_ok=True)

    resp = _post(UpdateRuntimeRequest(org_slug=org_slug, db_slug=db_slug, bundle_s3_key=bundle_s3_key))

    if watch:
        deadline = time.monotonic() + _RUNTIME_UPDATE_TIMEOUT
        if not json_output:
            print('Waiting for runtime build', end='', flush=True)
        final_state = ''
        while time.monotonic() < deadline:
            time.sleep(_RUNTIME_UPDATE_POLL_INTERVAL)
            try:
                db_resp = _post(GetDatabaseRequest(org_slug=org_slug, db_slug=db_slug))
                state = db_resp.get('database', {}).get('state', '')
                if state != 'UPDATING':
                    final_state = state
                    break
            except Exception:
                pass
            if not json_output:
                print('.', end='', flush=True)
        if not json_output:
            print()
            if final_state:
                print(f'Runtime build {final_state.lower()}.')
            else:
                print('Timed out waiting for runtime build.')

    if json_output:
        print(_json.dumps(resp))
    else:
        print(resp.get('message', 'runtime update triggered'))


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


def org_get(org_slug: str, json_output: bool = False) -> dict | None:
    import sys

    resp = _post(ListOrgsRequest())
    orgs = resp.get('orgs', [])
    org = next((o for o in orgs if o.get('org_slug') == org_slug), None)
    if org is None:
        print(f"Org '{org_slug}' not found.", file=sys.stderr)
        return None
    if json_output:
        import json

        print(json.dumps(org))
    else:
        slug = org.get('org_slug', '')
        org_id = org.get('org_id', '')
        default_db = org.get('default_db_slug') or ''
        line = f'{slug}  id={org_id}'
        if default_db:
            line += f'  default_db={default_db}'
        print(line)
    return org


def service_create(
    org_slug: str,
    db_slug: str,
    service_name: str,
    table_path: str,
    workers_min: int = 1,
    workers_max: int = 1,
    cpu: float = 0.5,
    memory_mb: int = 512,
    disk_gb: int = 10,
    service_config: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(
        CreateServiceRequest(
            org_slug=org_slug,
            db_slug=db_slug,
            service_name=service_name,
            table_path=table_path,
            workers_min=workers_min,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_update(
    org_slug: str,
    db_slug: str,
    service_name: str,
    workers_min: int | None = None,
    cpu: float | None = None,
    memory_mb: int | None = None,
    disk_gb: int | None = None,
    service_config: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(
        UpdateServiceRequest(
            org_slug=org_slug,
            db_slug=db_slug,
            service_name=service_name,
            workers_min=workers_min,
            cpu=cpu,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            service_config=service_config,
        )
    )
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_get(org_slug: str, db_slug: str, service_name: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(GetServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=service_name))
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_list(org_slug: str, db_slug: str, json_output: bool = False) -> list[dict[str, Any]]:
    resp = _post(ListServicesRequest(org_slug=org_slug, db_slug=db_slug))
    svcs = resp.get('services', [])
    if json_output:
        import json

        print(json.dumps(svcs))
    elif not svcs:
        print(f"No services in database '{db_slug}'.")
    else:
        for svc in svcs:
            _print_service(svc)
    return svcs


def service_start(
    org_slug: str,
    db_slug: str,
    service_name: str,
    workers_min: int | None = None,
    workers_max: int | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    resp = _post(StartServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=service_name))
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_stop(org_slug: str, db_slug: str, service_name: str, json_output: bool = False) -> dict[str, Any]:
    resp = _post(StopServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=service_name))
    svc = resp['service']
    if json_output:
        import json

        print(json.dumps(svc))
    else:
        _print_service(svc)
    return svc


def service_delete(org_slug: str, db_slug: str, service_name: str, json_output: bool = False) -> None:
    _post(DeleteServiceRequest(org_slug=org_slug, db_slug=db_slug, service_name=service_name))
    if json_output:
        import json

        print(json.dumps({'deleted': service_name}))
    else:
        print(f"Deleted service '{service_name}'.")


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
