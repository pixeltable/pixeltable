from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service import management_client
from pixeltable.service.db_md import DatabaseResources, DatabaseStatus
from pixeltable.service.management_protocol import (
    ArtifactUpload,
    DatabaseReport,
    GetArchiveRequest,
    GetArchiveResponse,
    GetDbRequest,
    GetDbResponse,
    ReportServiceInstanceRequest,
    UpdateDbRequest,
    UpdateDbResponse,
)
from pixeltable.utils.project import (
    ProjectFingerprint,
    ProjectPart,
    create_image_context,
    create_project_archive,
    project_fingerprint,
    unpacked_digest,
)
from pixeltable_cli.types import DbArtifact, DbChangeOp, DbPlan, DbState, DbTarget

_logger = logging.getLogger('pixeltable')

_UPLOAD_TIMEOUT = 300
_DOWNLOAD_TIMEOUT = 300

_ARCHIVE_DIR = 'project'

_DB_DESTRUCTIVE_HINT = "Re-run 'pxt db update' with --allow-destructive to apply these changes."

# how long a hosted database may stay in a transitional state before an update gives up on it
_DB_SETTLE_TIMEOUT = 3600.0
_DB_POLL_INTERVAL = 5.0

# the states a database passes through while it applies something
_DB_TRANSITIONAL = frozenset({DbState.PROVISIONING, DbState.UPDATING, DbState.STOPPING})


def db_diff(db_uri: str) -> DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    db_path = _validated_db_uri(db_uri)
    return _update_db_request(db_path, target=_db_resources(_get_db_config(db_path)), dry_run=True).plan


def db_fingerprint(db_path: catalog.Path) -> ProjectFingerprint | None:
    """Return the fingerprint of the project deployed to a hosted database; None for a local one."""
    if db_path.org is None or db_path.db is None:
        return None
    report = _get_db_report(db_path)
    return None if report is None or report.current is None else report.current.resources.fingerprint


def create_db_update_ops(target: DatabaseResources, current: DatabaseResources | None) -> list[DbChangeOp]:
    """The operations needed to reconcile current with target."""
    ops: list[DbChangeOp] = []

    if target.fingerprint is not None:
        if current is None or current.fingerprint is None:
            # nothing to diff against: the database is new, or still on the base image
            # TODO: record the base image's fingerprint at provisioning and remove this branch
            ops += [DbChangeOp.build_image(), DbChangeOp.upload_archive()]
        else:
            changed = target.fingerprint.compare(current.fingerprint)
            if ProjectPart.IMAGE in changed:
                ops.append(DbChangeOp.build_image(target.fingerprint.changes(current.fingerprint, {ProjectPart.IMAGE})))
            if ProjectPart.ARCHIVE in changed:
                ops.append(
                    DbChangeOp.upload_archive(target.fingerprint.changes(current.fingerprint, {ProjectPart.ARCHIVE}))
                )

    current_capacity = {} if current is None else current.capacity()
    combined = current_capacity | target.capacity()  # target settings take precedence
    for name, val in combined.items():
        if val != current_capacity.get(name):
            ops.append(DbChangeOp.capacity(name, current_capacity.get(name), val))

    return ops


def db_update(db_uri: str, *, allow_destructive: bool = False) -> DbPlan:
    """Reconcile the database at db_uri with its corresponding DatabaseConfig in the project configuration.

    This is the one verb that creates a hosted database. The control plane records the spec before acting
    on any of it, so an update interrupted anywhere is finished by running it again.

    Returns the plan that was applied, each operation annotated with its status.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    target = _db_resources(config)
    plan = _update_db_request(db_path, target=target, dry_run=True).plan
    if plan.destructive and not allow_destructive:
        destructive = ', '.join(op.name or '' for op in plan.ops if op.destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {db_uri} would apply destructive changes: {destructive}.\n{_DB_DESTRUCTIVE_HINT}',
        )

    settled, _ = _update_db(db_path, config, target)
    for op in plan.ops:
        op.status = 'applied'
    plan.state = settled.state
    plan.exists = True
    plan.status = 'applied'
    # what the plan asked for has been applied; an operation no update carries out is what is left
    plan.resolution = 'up_to_date'
    return plan


def _update_db(
    db_path: catalog.Path, config: DatabaseConfig, target: DatabaseResources, *, force_image_build: bool = False
) -> tuple[DatabaseStatus, set[DbArtifact]]:
    """Ask db_path to provide config's resources, store the artifacts it asks for, and wait for it to settle.

    Returns the status it settled in and the artifacts this call stored; one the store already held is
    not stored again.
    """
    response = _update_db_request(db_path, target=target, force_image_build=force_image_build)
    stored: set[DbArtifact] = set()
    rounds = 0
    while len(response.uploads) > 0:
        rounds += 1
        if rounds > 2:
            # > 2 rounds: we're not making progress
            wanted = ', '.join(upload.artifact for upload in response.uploads)
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR, f'{db_path.uri_str} still asks for {wanted} after it was stored'
            )
        stored.update(upload.artifact for upload in response.uploads)
        _store_artifacts(response.uploads, config)
        response = _update_db_request(db_path, target=target, force_image_build=force_image_build)

    settled = _await_db_settled(db_path)
    if settled.state == DbState.FAILED:
        reason = settled.failure_reason or 'no reason was reported'
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'{db_path.uri_str} is FAILED after the update: {reason}',
            provider='pixeltable_cloud',
        )
    if settled.last_build_outcome == 'FAILED':
        # a failed build leaves the database serving what it served before, rather than FAILED
        reason = settled.last_build_error or 'no reason was reported'
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'The image build for {db_path.uri_str} failed: {reason}',
            provider='pixeltable_cloud',
        )
    if settled.failure_reason is not None:
        # a step that failed and left the database serving what it served before still failed
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'{db_path.uri_str} did not reach the state it was given: {settled.failure_reason}',
            provider='pixeltable_cloud',
        )
    if settled.resources is None or settled.resources.fingerprint != target.fingerprint:
        # settling on a project other than the one asked for reports nothing else, so say so here
        raise excs.InternalError(
            excs.ErrorCode.INTERNAL_ERROR, f'{db_path.uri_str} settled on a project other than the one it was given'
        )
    return settled, stored


def _update_db_request(
    db_path: catalog.Path,
    *,
    target: DatabaseResources | None = None,
    dry_run: bool = False,
    force_image_build: bool = False,
) -> UpdateDbResponse:
    return UpdateDbResponse.model_validate(
        management_client.api_call(
            UpdateDbRequest(
                org=db_path.org, db=db_path.db, target=target, dry_run=dry_run, force_image_build=force_image_build
            )
        )
    )


def db_build_image(db_uri: str) -> list[DbChangeOp]:
    """Store this project's files at db_uri and rebuild its image, and wait for both.

    Builds whatever the project holds, without comparing it to the database first. Each returned operation
    carries what it did: the archive is stored only where the store does not hold it already.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    report = _get_db_report(db_path)
    if report is None:
        raise excs.NotFoundError(
            excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} does not exist; run `pxt db update` to create it'
        )
    if report.current is None:
        raise excs.NotFoundError(
            excs.ErrorCode.DEPLOYMENT_NOT_FOUND,
            f'{db_path.uri_str} is still being created; run `pxt db status` to check the status',
        )
    settled, stored = _update_db(db_path, config, _db_resources(config), force_image_build=True)
    image_op = DbChangeOp.build_image()
    # a build that did not run leaves nothing to report, whatever was asked for
    image_op.status = 'applied' if settled.last_build_outcome == 'SUCCEEDED' else 'skipped'
    archive_op = DbChangeOp.upload_archive()
    archive_op.status = 'applied' if 'archive' in stored else 'skipped'
    return [image_op, archive_op]


def unpack_project_archive(db_uri: str, dest: Path) -> GetArchiveResponse:
    """Unpack db_uri's project archive into dest, and return what the control plane served it as."""
    db_path = _validated_db_uri(db_uri)
    response = GetArchiveResponse.model_validate(
        management_client.api_call(GetArchiveRequest(org=db_path.org, db=db_path.db))
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
                    raise excs.RequestError(
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
            raise excs.RequestError(
                excs.ErrorCode.INVALID_DATA_FORMAT,
                f'{db_path.uri_str} served an archive holding project {unpacked}, not {response.digest}',
            )

        if dest.exists():
            shutil.rmtree(dest)
        project_dir.rename(dest)
    finally:
        shutil.rmtree(unpacking, ignore_errors=True)
    return response


def report_instance_fingerprint(
    db_uri: str, service_name: str, fingerprint: ProjectFingerprint, base_path: str = ''
) -> None:
    """Tell the database at db_uri which project the named service instance loaded."""
    db_path = _validated_db_uri(db_uri)
    management_client.api_call(
        ReportServiceInstanceRequest(
            org=db_path.org, db=db_path.db, service_name=service_name, base_path=base_path, fingerprint=fingerprint
        )
    )


def _db_resources(config: DatabaseConfig) -> DatabaseResources:
    # TODO: default_bucket?
    return DatabaseResources(
        fingerprint=project_fingerprint(_validated_project_root(), config),
        pxt_md_version=metadata.VERSION,
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
        try:
            # urlopen() raises for every 4xx and 5xx, so the status is only reachable through HTTPError
            urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT).close()
        except urllib.error.HTTPError as e:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Storing {path.name} failed: HTTP {e.code}',
                provider='pixeltable_cloud',
                status_code=e.code,
            ) from e


def _get_target_ops(plan: DbPlan, target: DbTarget) -> list[DbChangeOp]:
    """The plan's operations against one target."""
    return [op for op in plan.ops if op.target == target]


def _await_db_settled(db_path: catalog.Path) -> DatabaseStatus:
    """Poll the named database until it leaves a transitional state, and return the status it reached."""
    deadline = time.monotonic() + _DB_SETTLE_TIMEOUT
    while True:
        report = _get_db_report(db_path)
        if report is None or report.current is None:
            # a database that is gone has no status to report
            raise excs.NotFoundError(excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} no longer exists')
        if report.current.state not in _DB_TRANSITIONAL:
            return report.current
        if time.monotonic() >= deadline:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'{db_path.uri_str} is still {report.current.state} after {int(_DB_SETTLE_TIMEOUT)}s',
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


def _get_db_report(db_path: catalog.Path) -> DatabaseReport | None:
    try:
        response = management_client.api_call(GetDbRequest(org=db_path.org, db=db_path.db))
    except excs.ExternalServiceError as exc:
        if exc.provider_http_status_code == 404:
            return None
        raise
    return GetDbResponse.model_validate(response).report
