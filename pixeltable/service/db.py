from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import pydantic

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service import management_client
from pixeltable.service.management_protocol import (
    BuildImageRequest,
    CreateDbRequest,
    DeleteSecretRequest,
    GetDbRequest,
    GetProjectUploadUrlRequest,
    GetProjectUploadUrlResponse,
    ListSecretsRequest,
    ListSecretsResponse,
    SetProjectRequest,
    SetSecretRequest,
    UpdateDbRequest,
)
from pixeltable.utils.project import ProjectFingerprint, create_project_archive, project_fingerprint
from pixeltable_cli.types import DbChangeOp, DbPlan, DbTarget

_UPLOAD_TIMEOUT = 300

_DB_DESTRUCTIVE_HINT = "Re-run 'pxt db update' with --allow-destructive to apply these changes."

# a declared secret names the environment variable holding its value, as 'env:NAME'
_ENV_BINDING = 'env:'

# how long a hosted database may stay in a transitional state before an update gives up on it
_DB_SETTLE_TIMEOUT = 1200.0
_DB_POLL_INTERVAL = 5.0

# the states a database passes through while it applies something
_DB_TRANSITIONAL = frozenset({'PROVISIONING', 'UPDATING', 'STARTING', 'STOPPING'})


class DatabaseState(pydantic.BaseModel):
    """The state of a hosted db."""

    model_config = pydantic.ConfigDict(extra='ignore', populate_by_name=True)

    state: str = ''
    db_name: str | None = None
    default_bucket: str | None = None
    location: str | None = None
    region: str | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None

    # vars and their values
    vars: dict[str, str] = pydantic.Field(default_factory=dict)

    # None: not included in this report
    secret_keys: list[str] | None = None

    # the pods serving the database, which is the observed worker count; DatabaseConfig.workers is the
    # declared one. Named 'workers' on the wire.
    worker_status: list[dict[str, Any]] = pydantic.Field(default_factory=list, alias='workers')

    fingerprint: ProjectFingerprint | None = None

    # the image the database runs on, and the outcome of the build that produced it. Named 'runtime_image'
    # on the wire.
    image: str = pydantic.Field(default='', alias='runtime_image')
    last_build_state: str | None = None
    last_build_error: str | None = None


def db_diff(db_uri: str) -> DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    current = _get_db_state(db_path)
    if current is None:
        return DbPlan.from_ops(db_uri, None, [], [])

    fingerprint = project_fingerprint(_validated_project_root(), config)
    ops, not_compared = _compare_db(current, config, fingerprint)
    return DbPlan.from_ops(db_uri, current.state, ops, not_compared)


def db_update(db_uri: str, *, allow_destructive: bool = False) -> DbPlan:
    """Reconcile the database at db_uri with the entry declaring it: create it if it is absent, then apply
    secrets, the two artifacts, and capacity, in that order.

    Secrets go first, since code the pods run reads them as they start; capacity last, so that the resize
    restarts pods already on the new image and the new sources. A placement difference is skipped, since no
    update can carry it out. A database created here is given every secret and both artifacts: it reports no
    fingerprint yet, so the diff has nothing to compare against.

    Returns the plan that was applied, each operation annotated with its status.

    Args:
        db_uri: the pxt://org:db uri of the database the entry configures.
        allow_destructive: whether to apply changes that take capacity away or delete a secret.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    plan = db_diff(db_uri)
    if plan.destructive and not allow_destructive:
        for op in plan.ops:
            op.status = 'refused'
        destructive = ', '.join(op.name or '' for op in plan.ops if op.destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {db_uri} would apply destructive changes: {destructive}.\n{_DB_DESTRUCTIVE_HINT}',
        )

    for op in _get_target_ops(plan, 'placement'):
        op.status = 'skipped'

    if not plan.exists:
        management_client.api_call(
            CreateDbRequest(
                org=db_path.org,
                db=db_path.db,
                location=config.location,
                region=config.region,
                **_capacity_settings(config),
            )
        )
        _await_db_settled(db_path)
        # a database that has just been created reports no fingerprint, so there is nothing to diff against:
        # it needs every secret and both artifacts, whatever the plan says
        plan.ops = [DbChangeOp.secret(key, 'add') for key in sorted(config.secrets or {})]
        plan.ops += [DbChangeOp.build_image(), DbChangeOp.upload_archive()]

    for op in _get_target_ops(plan, 'secret'):
        _apply_secret_op(db_path, op, config)
        op.status = 'applied'

    image_ops, archive_ops = _get_target_ops(plan, 'image'), _get_target_ops(plan, 'archive')
    if len(image_ops) > 0 or len(archive_ops) > 0:
        _ship_project(config, db_path, image=len(image_ops) > 0)
        for op in image_ops + archive_ops:
            op.status = 'applied'

    changed = {op.name for op in _get_target_ops(plan, 'capacity')}
    if len(changed) > 0:
        # one request carrying every changed number, so the pods restart once
        management_client.api_call(
            UpdateDbRequest(
                org=db_path.org,
                db=db_path.db,
                workers=config.workers if 'workers' in changed else None,
                cpu=config.cpu if 'cpu' in changed else None,
                memory_mb=config.memory_mb if 'memory_mb' in changed else None,
                disk_gb=config.disk_gb if 'disk_gb' in changed else None,
            )
        )
        for op in _get_target_ops(plan, 'capacity'):
            op.status = 'applied'

    settled = _await_db_settled(db_path)
    plan.state = settled.state
    plan.exists = True
    plan.status = 'failed' if settled.state == 'FAILED' else 'applied'
    return plan


def db_build_image(db_uri: str) -> list[DbChangeOp]:
    """Ship this project to the database at db_uri and build its image, and wait for both.

    Ships and builds whatever the project holds, without comparing it to the database first.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    if _get_db_state(db_path) is None:
        raise excs.NotFoundError(
            excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} does not exist; run `pxt db update` to create it'
        )
    ops = [DbChangeOp.build_image(), DbChangeOp.upload_archive()]
    _ship_project(config, db_path)
    for op in ops:
        op.status = 'applied'
    return ops


def _ship_project(config: DatabaseConfig, db_path: catalog.Path, *, image: bool = True) -> None:
    """Store the project's archive, build the image from it when the environment moved, and roll the pods."""
    # both artifacts come from one archive, so it is stored once whichever of the two moved
    key = _upload_project_archive(config, db_path)
    if image:
        _build_image(config, db_path, key)
    _set_project(db_path, key)


def _capacity_settings(config: DatabaseConfig) -> dict[str, float | int]:
    """The capacity fields from DatabaseConfig as a dict."""
    declared = (('cpu', config.cpu), ('memory_mb', config.memory_mb), ('disk_gb', config.disk_gb))
    return {field: value for field, value in declared if value is not None}


def _get_target_ops(plan: DbPlan, target: DbTarget) -> list[DbChangeOp]:
    """The plan's operations against one target."""
    return [op for op in plan.ops if op.target == target]


def _build_image(config: DatabaseConfig, db_path: catalog.Path, project_key: str) -> None:
    project_root = _validated_project_root()
    fingerprint = project_fingerprint(project_root, config)
    management_client.api_call(
        BuildImageRequest(
            org=db_path.org,
            db=db_path.db,
            project_key=project_key,
            image_digest=fingerprint.image_digest(),
            python_version=fingerprint.python_version,
            system_dependencies=fingerprint.system_dependencies,
            pxt_md_version=metadata.VERSION,
        )
    )
    current = _await_db_settled(db_path)
    if current.last_build_state == 'FAILED':
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'The image build for {db_path.uri_str} failed: {current.last_build_error or "no reason was reported"}',
            provider='pixeltable_cloud',
        )


def _set_project(db_path: catalog.Path, project_key: str) -> None:
    """Point the database's pods at the stored archive, and wait for them to come back on it."""
    management_client.api_call(SetProjectRequest(org=db_path.org, db=db_path.db, project_key=project_key))
    current = _await_db_settled(db_path)
    if current.state == 'FAILED':
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'{db_path.uri_str} did not come back on the new project; it is {current.state}',
            provider='pixeltable_cloud',
        )


def _secret_keys(db_path: catalog.Path) -> list[str]:
    """The keys of the secrets the named database holds."""
    response = ListSecretsResponse.model_validate(
        management_client.api_call(ListSecretsRequest(org=db_path.org, db=db_path.db))
    )
    return response.keys


def _apply_secret_op(db_path: catalog.Path, op: DbChangeOp, config: DatabaseConfig) -> None:
    """Apply one secret operation: set the declared value, or delete the key."""
    key = op.name
    if op.op == 'drop':
        management_client.api_call(DeleteSecretRequest(org=db_path.org, db=db_path.db, key=key))
        return
    binding = (config.secrets or {})[key]
    management_client.api_call(
        SetSecretRequest(org=db_path.org, db=db_path.db, key=key, value=_secret_value(key, binding))
    )


def _secret_value(key: str, binding: str) -> str:
    """Read a declared secret's value from the environment variable its binding names."""
    name = binding[len(_ENV_BINDING) :] if binding.startswith(_ENV_BINDING) else None
    if name is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"secret {key!r} is declared as {binding!r}; write '{_ENV_BINDING}NAME' to name the environment "
            'variable holding the value, which keeps the value out of the project',
        )
    value = os.environ.get(name)
    if value is None or value == '':
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'secret {key!r} is bound to {name}, which is not set in the environment',
        )
    return value


def _await_db_settled(db_path: catalog.Path) -> DatabaseState:
    """Poll the named database until it leaves a transitional state, and return the state it reached."""
    deadline = time.monotonic() + _DB_SETTLE_TIMEOUT
    while True:
        current = _get_db_state(db_path)
        if current is None:
            # a database that is gone has no state to report
            raise excs.NotFoundError(excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} no longer exists')
        if current.state not in _DB_TRANSITIONAL:
            return current
        if time.monotonic() >= deadline:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'{db_path.uri_str} is still {current.state} after {int(_DB_SETTLE_TIMEOUT)}s',
                provider='pixeltable_cloud',
            )
        time.sleep(_DB_POLL_INTERVAL)


def _validated_project_root() -> Path:
    project_root = Config.get().project_root
    if project_root is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, 'no project configuration here; run `pxt init` to write one'
        )
    return project_root


def _validated_db_uri(db_uri_str: str) -> catalog.Path:
    path = catalog.Path.parse(db_uri_str, allow_empty_path=True)
    if path.org is None or path.db is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'{db_uri_str!r} does not name a hosted database; write pxt://org:db'
        )
    return path


def _get_db_config(db_uri: catalog.Path) -> DatabaseConfig:
    config = Config.get().get_database_config(db_uri)
    if config is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'no [[pixeltable.database]] entry names {db_uri.uri_str!r}; add one to the project configuration',
        )
    return config


def _get_db_state(db_path: catalog.Path) -> DatabaseState | None:
    """The named database as the control plane reports it, secret keys included; None if it holds no such database."""
    try:
        response = management_client.api_call(GetDbRequest(org=db_path.org, db=db_path.db))
    except excs.ExternalServiceError as e:
        if e.provider_http_status_code != 404:
            raise
        return None
    state = DatabaseState.model_validate(response.get('database', response))
    state.secret_keys = _secret_keys(db_path)
    return state


def _compare_db(
    current: DatabaseState, config: DatabaseConfig, fingerprint: ProjectFingerprint
) -> tuple[list[DbChangeOp], list[str]]:
    """The operations that make current match config + fingerprint."""
    ops: list[DbChangeOp] = []
    not_compared: list[str] = []

    if current.fingerprint is None:
        not_compared.append('image')
    else:
        if fingerprint.image_needed(current.fingerprint):
            ops.append(DbChangeOp.image_moved(fingerprint.changes(current.fingerprint, 'image')))
        if fingerprint.archive_needed(current.fingerprint):
            ops.append(DbChangeOp.archive_moved(fingerprint.changes(current.fingerprint, 'archive')))

    for field, config_value, current_value in (
        ('cpu', config.cpu, current.cpu),
        ('memory_mb', config.memory_mb, current.memory_mb),
        ('disk_gb', config.disk_gb, current.disk_gb),
        ('workers', config.workers, len(current.worker_status)),
    ):
        if config_value is None or config_value == current_value:
            continue
        ops.append(DbChangeOp.capacity(field, current_value, config_value))

    if current.secret_keys is None:
        not_compared.append('secrets')
    else:
        for key in sorted(set(config.secrets or {}) - set(current.secret_keys)):
            ops.append(DbChangeOp.secret(key, 'add'))
        for key in sorted(set(current.secret_keys) - set(config.secrets or {})):
            ops.append(DbChangeOp.secret(key, 'drop'))

    for field, config_value, current_value in (
        ('location', config.location, current.location),
        ('region', config.region, current.region),
    ):
        if config_value is None or current_value is None or config_value == current_value:
            continue
        ops.append(DbChangeOp.placement(field, current_value, config_value))

    if config.vars is not None:
        # a hosted database has nowhere to keep a var that is not a secret
        not_compared.append('vars')

    return ops, not_compared


def _upload_project_archive(config: DatabaseConfig, db_path: catalog.Path, *, show_progress: bool = False) -> str:
    """Store the project as the archive the named database's pods and image builds read.

    Returns the control plane's key for it. A project whose digest names an archive the control plane
    already holds is neither packaged nor uploaded.
    """
    project_root = _validated_project_root()
    digest = project_fingerprint(project_root, config).archive_digest()
    response = GetProjectUploadUrlResponse.model_validate(
        management_client.api_call(GetProjectUploadUrlRequest(org=db_path.org, db=db_path.db, digest=digest))
    )
    if response.presigned_url is None:
        return response.project_key

    archive_path = create_project_archive(project_root, config, show_progress=show_progress)
    try:
        content = archive_path.read_bytes()
        request = urllib.request.Request(response.presigned_url, data=content, method='PUT')
        request.add_header('Content-Type', 'application/octet-stream')
        request.add_header('Content-Length', str(len(content)))
        with urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT) as r:
            if r.status >= 400:
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_ERROR,
                    f'Project upload failed: HTTP {r.status}',
                    provider='pixeltable_cloud',
                    status_code=r.status,
                )
    finally:
        archive_path.unlink(missing_ok=True)
    return response.project_key
