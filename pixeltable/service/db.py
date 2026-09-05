from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service import management_client
from pixeltable.service.management_protocol import (
    ArtifactUpload,
    DatabaseSpec,
    DatabaseState,
    DatabaseStatus,
    DeleteSecretRequest,
    GetArchiveRequest,
    GetArchiveResponse,
    GetDbRequest,
    GetDbResponse,
    ReportServiceInstanceRequest,
    SetSecretRequest,
    UpdateDbRequest,
    UpdateDbResponse,
)
from pixeltable.utils.project import (
    ProjectFingerprint,
    ProjectPart,
    create_image_context,
    create_project_archive,
    loaded_fingerprint,
    project_fingerprint,
    unpacked_digest,
)
from pixeltable_cli.types import DbChangeOp, DbPlan, DbTarget

_logger = logging.getLogger('pixeltable')

_UPLOAD_TIMEOUT = 300
_DOWNLOAD_TIMEOUT = 300

_ARCHIVE_DIR = 'project'

_DB_DESTRUCTIVE_HINT = "Re-run 'pxt db update' with --allow-destructive to apply these changes."

# a defined secret names the environment variable holding its value, as 'env:NAME'
_ENV_BINDING = 'env:'

# how long a hosted database may stay in a transitional state before an update gives up on it
_DB_SETTLE_TIMEOUT = 3600.0
_DB_POLL_INTERVAL = 5.0

# the states a database passes through while it applies something
_DB_TRANSITIONAL = frozenset({'PROVISIONING', 'UPDATING', 'STARTING', 'STOPPING'})

# one round names the missing artifacts and the next finds them stored, so a third is not making progress
_MAX_UPLOAD_ROUNDS = 2


def db_diff(db_uri: str) -> DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    db_path = _validated_db_uri(db_uri)
    return _update_db(db_path, _target_spec(_get_db_config(db_path)), dry_run=True).plan


def published_fingerprint(db_path: catalog.Path) -> ProjectFingerprint | None:
    """The fingerprint db_path's pods are running."""
    if db_path.org is None or db_path.db is None:
        return None
    state = _get_db_state(db_path)
    return None if state is None else state.status.fingerprint


def db_plan(db_uri: str, target: DatabaseSpec, current: DatabaseStatus | None) -> DbPlan:
    """The plan that makes the database at db_uri provide target; current is None for one that does not exist.

    Comparing against what the database provides, rather than the spec it was last given, covers a changed
    project, an interrupted rollout and a failed build in one comparison.
    """
    status = DatabaseStatus() if current is None else current
    ops: list[DbChangeOp] = []
    if target.fingerprint is not None:
        ops += _artifact_ops(target.fingerprint, status.fingerprint)

    for field, wanted, running in (
        ('cpu', target.cpu, status.cpu),
        ('memory_mb', target.memory_mb, status.memory_mb),
        ('disk_gb', target.disk_gb, status.disk_gb),
        ('workers', target.workers, status.workers),
    ):
        if wanted is None or wanted == running:
            continue
        ops.append(DbChangeOp.capacity(field, running, wanted))

    ops += _secret_ops(target, status)
    return DbPlan.from_ops(db_uri, None if current is None else status.state, ops)


def _artifact_ops(target: ProjectFingerprint, running: ProjectFingerprint | None) -> list[DbChangeOp]:
    """The operations that give the pods target's image and archive."""
    if running is None:
        # a database serving nothing yet needs both artifacts, and there is no difference to name
        return [DbChangeOp.build_image(), DbChangeOp.upload_archive()]
    ops: list[DbChangeOp] = []
    moved = target.compare(running)
    if ProjectPart.IMAGE in moved:
        ops.append(DbChangeOp.build_image(target.changes(running, {ProjectPart.IMAGE})))
    if ProjectPart.ARCHIVE in moved:
        ops.append(DbChangeOp.upload_archive(target.changes(running, {ProjectPart.ARCHIVE})))
    return ops


def _secret_ops(target: DatabaseSpec, current: DatabaseStatus) -> list[DbChangeOp]:
    """The operations that set the secrets target binds, and restart the pods onto the stored values.

    Setting a secret and running with it are two steps: only the client can resolve a binding, and only the
    control plane can tell whether a pod holds the stored value.
    """
    stored = {key: value.digest for key, value in target.stored_secrets.items()}
    # a key set from a different binding has to be set again: the two sources may hold different values
    rebound = {
        key for key in set(target.secrets) & set(stored) if target.secrets[key] != target.stored_secrets[key].binding
    }
    to_set = sorted((set(target.secrets) - set(stored)) | rebound)
    to_drop = sorted(set(stored) - set(target.secrets))
    ops = [DbChangeOp.secret(key, 'add') for key in to_set]
    ops += [DbChangeOp.secret(key, 'drop') for key in to_drop]

    # a key this plan sets or deletes restarts the pods already
    settled = (set(stored) | set(current.secret_digests)) - set(to_set) - set(to_drop)
    behind = {key for key in settled if stored.get(key) != current.secret_digests.get(key)}
    return ops + [DbChangeOp.stale_secret(key) for key in sorted(behind)]


def db_update(db_uri: str, *, allow_destructive: bool = False) -> DbPlan:
    """Reconcile the database at db_uri with its corresponding DatabaseConfig in the project configuration.

    This is the one verb that creates a hosted database. The control plane records the spec before acting
    on any of it, so an update interrupted anywhere is finished by running it again.

    Secrets go first: their values live outside the project, so the client resolves each binding and the
    control plane cannot.

    Returns the plan that was applied, each operation annotated with its status.

    Args:
        db_uri: the pxt://org:db uri of the database the entry configures.
        allow_destructive: whether to apply changes that take capacity away or delete a secret.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    spec = _target_spec(config)
    plan = _update_db(db_path, spec, dry_run=True).plan
    if plan.destructive and not allow_destructive:
        for op in plan.ops:
            op.status = 'refused'
        destructive = ', '.join(op.name or '' for op in plan.ops if op.destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {db_uri} would apply destructive changes: {destructive}.\n{_DB_DESTRUCTIVE_HINT}',
        )

    for op in _get_target_ops(plan, 'secret'):
        # the reconcile below restarts the pods onto a stored value, so an alter needs nothing here
        if op.op != 'alter':
            _apply_secret_op(db_path, op, config)

    settled = _apply_spec(db_path, config, spec)
    for op in plan.ops:
        op.status = 'applied'
    plan.state = settled.status.state
    plan.exists = True
    plan.status = 'applied'
    # what the plan asked for has been applied; an operation no update carries out is what is left
    plan.resolution = 'up_to_date'
    return plan


def _apply_spec(
    db_path: catalog.Path, config: DatabaseConfig, spec: DatabaseSpec, *, force_image_build: bool = False
) -> DatabaseState:
    """Ask db_path to provide spec, store the artifacts it asks for, and wait for it to settle."""
    response = _update_db(db_path, spec, force_image_build=force_image_build)
    rounds = 0
    while len(response.uploads) > 0:
        rounds += 1
        if rounds > _MAX_UPLOAD_ROUNDS:
            wanted = ', '.join(upload.artifact for upload in response.uploads)
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'{db_path.uri_str} still asks for {wanted} after it was stored',
                provider='pixeltable_cloud',
            )
        _store_artifacts(response.uploads, config)
        response = _update_db(db_path, spec, force_image_build=force_image_build)

    settled = _await_db_settled(db_path)
    if settled.status.state == 'FAILED':
        reason = settled.status.failure_reason or 'no reason was reported'
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'{db_path.uri_str} is FAILED after the update: {reason}',
            provider='pixeltable_cloud',
        )
    if settled.status.last_build_outcome == 'FAILED':
        # a failed build leaves the database serving what it served before, rather than FAILED
        reason = settled.status.last_build_error or 'no reason was reported'
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'The image build for {db_path.uri_str} failed: {reason}',
            provider='pixeltable_cloud',
        )
    return settled


def _update_db(
    db_path: catalog.Path, spec: DatabaseSpec, *, dry_run: bool = False, force_image_build: bool = False
) -> UpdateDbResponse:
    return UpdateDbResponse.model_validate(
        management_client.api_call(
            UpdateDbRequest(
                org=db_path.org, db=db_path.db, spec=spec, dry_run=dry_run, force_image_build=force_image_build
            )
        )
    )


def db_build_image(db_uri: str) -> list[DbChangeOp]:
    """Store this project's files at db_uri and rebuild its image, and wait for both.

    Stores and builds whatever the project holds, without comparing it to the database first.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    if _get_db_state(db_path) is None:
        raise excs.NotFoundError(
            excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} does not exist; run `pxt db update` to create it'
        )
    ops = [DbChangeOp.build_image(), DbChangeOp.upload_archive()]
    _apply_spec(db_path, config, _target_spec(config), force_image_build=True)
    for op in ops:
        op.status = 'applied'
    return ops


def unpack_project_archive(db_uri: str, dest: Path, *, expected_digest: str | None = None) -> str:
    """Unpack db_uri's project archive into dest, and return the archive's digest.

    Refuses an archive whose digest is not expected_digest: a pod is told which project to run, and a
    different one would serve code nobody asked for.
    """
    db_path = _validated_db_uri(db_uri)
    response = GetArchiveResponse.model_validate(
        management_client.api_call(GetArchiveRequest(org=db_path.org, db=db_path.db))
    )
    if expected_digest is not None and response.digest != expected_digest:
        raise excs.Error(
            excs.ErrorCode.INVALID_STATE,
            f'{db_path.uri_str} serves project {response.digest}, not the {expected_digest} this process runs',
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # unpacked next to dest and moved into place, so that dest never holds a file the archive dropped
    unpacking = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f'.{dest.name}.'))
    archive_path = unpacking / 'project.tar.bz2'
    try:
        # streamed to disk: a project may select files too large to hold in memory
        with (
            urllib.request.urlopen(response.presigned_url, timeout=_DOWNLOAD_TIMEOUT) as r,
            archive_path.open('wb') as f,
        ):
            shutil.copyfileobj(r, f)

        prefix = f'{_ARCHIVE_DIR}/'
        project_dir = unpacking / _ARCHIVE_DIR
        project_dir.mkdir()  # an archive holding no files still unpacks to an empty project
        with tarfile.open(archive_path, mode='r:bz2') as tf:
            members: list[tarfile.TarInfo] = []
            for member in tf.getmembers():
                if member.name == _ARCHIVE_DIR:
                    continue
                if not member.name.startswith(prefix):
                    raise excs.Error(
                        excs.ErrorCode.INVALID_DATA_FORMAT,
                        f'{db_path.uri_str} serves an archive holding {member.name!r}, which is outside {prefix}',
                    )
                member.name = member.name[len(prefix) :]
                members.append(member)
            # filter='data': refuses a member naming a path outside the directory, and drops ownership bits
            tf.extractall(project_dir, members=members, filter='data')

        unpacked = unpacked_digest(project_dir)
        if unpacked != response.digest:
            # what arrived is not what the control plane named, whatever it named
            raise excs.Error(
                excs.ErrorCode.INVALID_DATA_FORMAT,
                f'{db_path.uri_str} served an archive holding project {unpacked}, not {response.digest}',
            )

        if dest.exists():
            shutil.rmtree(dest)
        project_dir.rename(dest)
    finally:
        shutil.rmtree(unpacking, ignore_errors=True)
    return response.digest


def report_instance_fingerprint(db_uri: str, service_name: str, base_path: str = '') -> None:
    """Tell the database at db_uri which of its project files the named service instance loaded."""
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    management_client.api_call(
        ReportServiceInstanceRequest(
            org=db_path.org,
            db=db_path.db,
            service_name=service_name,
            base_path=base_path,
            fingerprint=loaded_fingerprint(_validated_project_root(), config),
        )
    )


def _target_spec(config: DatabaseConfig) -> DatabaseSpec:
    """The spec config asks its database to provide."""
    return DatabaseSpec(
        fingerprint=project_fingerprint(_validated_project_root(), config),
        pxt_md_version=metadata.VERSION,
        secrets=config.secrets or {},
        cpu=config.cpu,
        memory_mb=config.memory_mb,
        disk_gb=config.disk_gb,
        workers=config.workers,
    )


def _store_artifacts(uploads: list[ArtifactUpload], config: DatabaseConfig) -> None:
    """Package each artifact the control plane asked for and store it at the url it gave."""
    project_root = _validated_project_root()
    for upload in uploads:
        if upload.artifact == 'archive':
            path = create_project_archive(project_root, config, show_progress=True)
        else:
            path = create_image_context(project_root)
        try:
            _put_artifact(upload.url, path)
        finally:
            path.unlink(missing_ok=True)


def _put_artifact(url: str, path: Path) -> None:
    with path.open('rb') as f:
        request = urllib.request.Request(url, data=f, method='PUT')
        request.add_header('Content-Type', 'application/octet-stream')
        request.add_header('Content-Length', str(path.stat().st_size))
        with urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT) as r:
            if r.status >= 400:
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_ERROR,
                    f'Storing {path.name} failed: HTTP {r.status}',
                    provider='pixeltable_cloud',
                    status_code=r.status,
                )


def _get_target_ops(plan: DbPlan, target: DbTarget) -> list[DbChangeOp]:
    """The plan's operations against one target."""
    return [op for op in plan.ops if op.target == target]


def _apply_secret_op(db_path: catalog.Path, op: DbChangeOp, config: DatabaseConfig) -> None:
    """Apply one secret operation: set the defined value, or delete the key."""
    key = op.name
    if op.op == 'drop':
        management_client.api_call(DeleteSecretRequest(org=db_path.org, db=db_path.db, key=key))
        return
    binding = (config.secrets or {})[key]
    management_client.api_call(
        SetSecretRequest(org=db_path.org, db=db_path.db, key=key, value=_secret_value(key, binding))
    )


def _secret_value(key: str, binding: str) -> str:
    """Read a defined secret's value from the environment variable its binding names."""
    name = binding[len(_ENV_BINDING) :] if binding.startswith(_ENV_BINDING) else None
    if name is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"secret {key!r} is defined as {binding!r}; write '{_ENV_BINDING}NAME' to name the environment "
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
        if current.status.state not in _DB_TRANSITIONAL:
            return current
        if time.monotonic() >= deadline:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'{db_path.uri_str} is still {current.status.state} after {int(_DB_SETTLE_TIMEOUT)}s',
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
        where = Config.get().project_config_file or 'the project configuration'
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'no [[pixeltable.database]] entry names {db_uri.uri_str!r}; add one to {where}:\n'
            f'  [[pixeltable.database]]\n  name = {db_uri.uri_str!r}',
        )
    return config


def _get_db_state(db_path: catalog.Path) -> DatabaseState | None:
    """The named database as the control plane reports it; None if it holds no such database."""
    try:
        response = management_client.api_call(GetDbRequest(org=db_path.org, db=db_path.db))
    except excs.ExternalServiceError as exc:
        if exc.provider_http_status_code == 404:
            return None
        raise
    return GetDbResponse.model_validate(response).database
