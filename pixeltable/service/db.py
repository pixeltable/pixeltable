from __future__ import annotations

import base64
import hashlib
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Literal

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service import management_client, receipts
from pixeltable.service.db_md import DatabaseResources
from pixeltable.service.management_protocol import (
    BlobRef,
    BlobUpload,
    DatabaseReport,
    DatabaseTarget,
    DbReceiptResponse,
    DeleteDbRequest,
    GetDbRequest,
    GetDbResponse,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    ReportServiceInstanceRequest,
    RestartDbRequest,
    StartDbRequest,
    StopDbRequest,
    SubmitUpdateRequest,
)
from pixeltable.utils.project import (
    ProjectFingerprint,
    ProjectPart,
    image_input_files,
    package_project_archive,
    project_fingerprint,
)
from pixeltable_cli.types import DbChangeOp, DbPlan, GenerationReceipt, OpStatus

_logger = logging.getLogger('pixeltable')

_UPLOAD_TIMEOUT = 300
_UPLOAD_ATTEMPTS = 2

_DB_DESTRUCTIVE_HINT = "Re-run 'pxt db update' with --allow-destructive to apply these changes."

DbAction = Literal['start', 'stop', 'restart', 'delete']

_LIFECYCLE_REQUESTS = {
    'start': StartDbRequest,
    'stop': StopDbRequest,
    'restart': RestartDbRequest,
    'delete': DeleteDbRequest,
}


def db_diff(db_uri: str) -> DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    db_path = _validated_db_uri(db_uri)
    return _plan(db_path, _db_target(_get_db_config(db_path)))[0]


def db_fingerprint(db_path: catalog.Path, *, desired: bool = False) -> ProjectFingerprint | None:
    """Return the fingerprint of the project deployed to a hosted database; None for a local one.

    desired: return the project of the current desired generation, which may still be in progress.
    """
    if db_path.org is None or db_path.org == 'local' or db_path.db is None:
        return None
    report = _get_db_report(db_path)
    if report is None:
        return None
    if desired and report.target_resources is not None and report.target_resources.fingerprint is not None:
        return report.target_resources.fingerprint
    return None if report.current is None else report.current.resources.fingerprint


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
            if ProjectPart.BINDINGS in changed:
                ops.append(DbChangeOp.rebind(target.fingerprint.changes(current.fingerprint, {ProjectPart.BINDINGS})))

    current_capacity = {} if current is None else current.capacity()
    combined = current_capacity | target.capacity()  # target settings take precedence
    for name, val in combined.items():
        if val != current_capacity.get(name):
            ops.append(DbChangeOp.capacity(name, current_capacity.get(name), val))

    return ops


def db_update(
    db_uri: str, *, allow_destructive: bool = False, expected_generation: int | None = None, wait: bool = True
) -> DbPlan:
    """Submit the project and capacity that the project configuration declares for the database at db_uri.

    This is the only way to create a hosted database. The control plane applies an accepted submission whether
    or not this call waits, and resubmitting the same project returns the receipt of the existing generation.

    Args:
        expected_generation: the database generation the caller's plan was computed against; a submission
            against an older one is refused. None submits against the generation this call plans against.
        wait: wait until the accepted generation finishes.

    Returns the submitted plan, with its receipt.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    target = _db_target(config)
    plan, prepared = _plan(db_path, target)
    # TODO: put allow_destructive into the submission and let the control plane validate it
    if plan.destructive and not allow_destructive:
        destructive = ', '.join(op.name or '' for op in plan.ops if op.destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {db_uri} would apply destructive changes: {destructive}.\n{_DB_DESTRUCTIVE_HINT}',
        )

    if expected_generation is None:
        expected_generation = prepared.generations.database or 0
    accepted, _ = _submit(db_path, config, target, expected_generation=expected_generation)
    settled = _settle(db_path, accepted, wait=wait)
    status: OpStatus = 'applied' if wait else 'accepted'
    for op in plan.ops:
        op.status = status
    plan.status = status
    plan.exists = True
    plan.receipts = settled
    if wait:
        report = _get_db_report(db_path)
        plan.state = None if report is None or report.current is None else report.current.state
        # the receipt is observed, so every planned operation has been applied
        plan.resolution = 'up_to_date'
    return plan


def db_build_image(db_uri: str, *, wait: bool = True) -> tuple[list[DbChangeOp], list[GenerationReceipt]]:
    """Store this project's files at db_uri and rebuild its image, whether or not the project changed.

    Returns the operations, each with its status, and the receipts of the rebuild. The archive is uploaded only
    if the control plane does not hold it already.
    """
    db_path = _validated_db_uri(db_uri)
    config = _get_db_config(db_path)
    target = _db_target(config)
    prepared = _prepare(db_path, target, blob=None)
    if prepared.report.current is None:
        raise excs.NotFoundError(
            excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} does not exist; run `pxt db update` to create it'
        )
    desired = prepared.report.target_resources or prepared.report.current.resources
    target = target.model_copy(
        update={
            'cpu': desired.cpu,
            'memory_mb': desired.memory_mb,
            'disk_gb': desired.disk_gb,
            'workers': desired.workers,
            'force_build_nonce': uuid.uuid4().hex,
        }
    )
    accepted, uploaded = _submit(db_path, config, target, expected_generation=prepared.generations.database or 0)
    settled = _settle(db_path, accepted, wait=wait)
    image_op = DbChangeOp.build_image()
    image_op.status = 'applied' if wait else 'accepted'
    archive_op = DbChangeOp.upload_archive()
    archive_op.status = 'applied' if uploaded else 'skipped'
    return [image_op, archive_op], settled


def db_change_lifecycle(db_uri: str, action: DbAction, *, wait: bool = True) -> GenerationReceipt:
    """Start, stop, restart or delete the database at db_uri, and return its receipt."""
    db_path = _validated_db_uri(db_uri)
    request = _LIFECYCLE_REQUESTS[action](org=db_path.org, db=_db_name(db_path))
    receipt = DbReceiptResponse.model_validate(management_client.api_call(request)).receipt
    return _settle(db_path, [receipt], wait=wait)[0]


def db_receipts(db_uri: str, accepted: list[GenerationReceipt]) -> list[GenerationReceipt]:
    """Re-read the given receipts of the database at db_uri."""
    return receipts.read_receipts(_validated_db_uri(db_uri), accepted)


def db_retry(db_uri: str, *, wait: bool = True) -> GenerationReceipt:
    """Start a new attempt of the current generation of the database at db_uri, which must have failed."""
    db_path = _validated_db_uri(db_uri)
    report = _get_db_report(db_path)
    if report is None:
        raise excs.NotFoundError(excs.ErrorCode.DEPLOYMENT_NOT_FOUND, f'{db_path.uri_str} does not exist')
    if report.receipt is None or not report.receipt.failed:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'the current generation of {db_path.uri_str} has not failed; nothing to retry',
        )
    return _settle(db_path, [receipts.retry(db_path, report.receipt)], wait=wait)[0]


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


def _settle(db_path: catalog.Path, accepted: list[GenerationReceipt], *, wait: bool) -> list[GenerationReceipt]:
    if not wait:
        receipts.raise_if_unsuccessful(db_path, accepted)
        return accepted
    return receipts.await_receipts(db_path, accepted)


def _plan(db_path: catalog.Path, target: DatabaseTarget) -> tuple[DbPlan, PrepareUpdateResponse]:
    """The plan for submitting target to db_path, and the prepare_update response it came from."""
    prepared = _prepare(db_path, target, blob=None)
    plan = prepared.plan
    if plan is None:
        current = prepared.report.current
        plan = DbPlan.from_ops(
            db_path.uri_str,
            None if current is None else current.state,
            create_db_update_ops(target, None if current is None else current.resources),
        )
    plan.generation = prepared.generations.database or 0
    receipt = prepared.report.receipt
    if receipt is not None and not receipt.observed:
        plan.receipts = [receipt]
        if plan.resolution == 'up_to_date':
            plan.ops.append(DbChangeOp.unsettled_generation(receipt))
            plan.resolution = 'update_additive'
    return plan, prepared


def _prepare(db_path: catalog.Path, target: DatabaseTarget, *, blob: BlobRef | None) -> PrepareUpdateResponse:
    request = PrepareUpdateRequest(org=db_path.org, db=_db_name(db_path), database_target=target, blob=blob)
    return PrepareUpdateResponse.model_validate(management_client.api_call(request))


def _submit(
    db_path: catalog.Path, config: DatabaseConfig, target: DatabaseTarget, *, expected_generation: int
) -> tuple[list[GenerationReceipt], bool]:
    """Package the project, upload it unless the control plane holds it, and submit target against
    expected_generation, read before the project was packaged.

    Returns the resulting receipts, and whether the archive was uploaded.
    """
    if target.fingerprint is None:
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, 'a project was submitted without a fingerprint')
    project_root = _validated_project_root()
    image_input_files(project_root)
    archive = package_project_archive(project_root, config, show_progress=True)
    try:
        changed = {
            path
            for path in set(archive.files) | set(target.fingerprint.files)
            if archive.files.get(path) != target.fingerprint.files.get(path)
        }
        if len(changed) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                f'the project changed while it was being packaged ({"; ".join(sorted(changed))}); '
                'run the command again',
            )
        blob = _blob_ref(archive.path)
        prepared = _prepare(db_path, target, blob=blob)
        if prepared.upload is not None:
            _put_blob(prepared.upload, archive.path, blob)
        request = SubmitUpdateRequest(
            org=db_path.org,
            db=_db_name(db_path),
            database_target=target,
            blob=blob,
            expected_generations=prepared.generations.model_copy(update={'database': expected_generation}),
        )
        accepted = receipts.submit(db_path, request)
        if len(accepted) != 1:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR,
                f'{db_path.uri_str} accepted a database update as {len(accepted)} receipts rather than one',
            )
        return accepted, prepared.upload is not None
    finally:
        archive.path.unlink(missing_ok=True)


def _blob_ref(path: Path) -> BlobRef:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return BlobRef(sha256=digest.hexdigest(), size=path.stat().st_size)


def _put_blob(upload: BlobUpload, path: Path, blob: BlobRef) -> None:
    """Upload the blob at path; a blob already in the store counts as uploaded.

    A network failure is retried once, which If-None-Match makes safe.
    """
    headers = {
        'Content-Type': 'application/octet-stream',
        **upload.headers,
        'Content-Length': str(blob.size),
        'x-amz-checksum-sha256': base64.b64encode(bytes.fromhex(blob.sha256)).decode(),
        'If-None-Match': '*',
    }
    for attempt in range(_UPLOAD_ATTEMPTS):
        with path.open('rb') as f:
            request = urllib.request.Request(upload.url, data=f, method='PUT', headers=headers)
            try:
                # urlopen() raises for every 4xx and 5xx, so the status is only reachable through HTTPError
                urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT).close()
                return
            except urllib.error.HTTPError as e:
                if e.code == 412:
                    return
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_ERROR,
                    f'Uploading the project archive failed: HTTP {e.code}',
                    provider='pixeltable_cloud',
                    status_code=e.code,
                ) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt + 1 < _UPLOAD_ATTEMPTS:
                    continue
                reason = e.reason if isinstance(e, urllib.error.URLError) else e
                raise excs.ExternalServiceError(
                    excs.ErrorCode.PROVIDER_ERROR,
                    f'Uploading the project archive failed: {reason}',
                    provider='pixeltable_cloud',
                ) from e


def _db_target(config: DatabaseConfig) -> DatabaseTarget:
    return DatabaseTarget(
        fingerprint=project_fingerprint(_validated_project_root(), config),
        pxt_md_version=metadata.VERSION,
        cpu=config.cpu,
        memory_mb=config.memory_mb,
        disk_gb=config.disk_gb,
        workers=config.workers,
    )


def _db_name(db_path: catalog.Path) -> str:
    assert db_path.db is not None
    return db_path.db


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
            f'no [[pixeltable.database]] entry for {db_uri.uri_str!r}; add one to {where}:\n'
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
