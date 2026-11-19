from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator, Literal, Sequence

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service import management_client, receipts
from pixeltable.service.db_md import DatabaseResources, ImageKind
from pixeltable.service.management_protocol import (
    BlobRef,
    BlobUpload,
    DatabaseReport,
    DatabaseTarget,
    DbReceiptResponse,
    DefaultImage,
    DeleteDbRequest,
    GetDbRequest,
    GetDbResponse,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    ReportServiceInstanceRequest,
    RestartDbRequest,
    ServiceMutation,
    StartDbRequest,
    StopDbRequest,
    SubmitUpdateRequest,
)
from pixeltable.service.resources import DatabaseCapacity
from pixeltable.utils.project import (
    ProjectFingerprint,
    ProjectPart,
    check_hosted_pixeltable,
    declared_requirements,
    image_input_files,
    normalized_name,
    package_project_archive,
    project_fingerprint,
    requirement_name,
)
from pixeltable_cli.types import DbChangeOp, DbPlan, GenerationReceipt, OpStatus
from pixeltable_cli.utils import hosted_db_uri_error

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


@dataclasses.dataclass(frozen=True)
class Deployment:
    """What a hosted database runs, as its report says."""

    # the deployed project; None while the database holds no project files
    fingerprint: ProjectFingerprint | None

    # None from a control plane that does not say, which runs a project only on an image built by `pxt db update`
    image: ImageKind | None

    default_image: DefaultImage | None

    @property
    def runtime(self) -> str | None:
        """What the default image runs, printable; None if the control plane does not say."""
        if self.default_image is None:
            return None
        return f'pixeltable {self.default_image.pixeltable_version}, Python {self.default_image.python_version}'


def db_deployment(db_path: catalog.Path, *, desired: bool = False) -> Deployment | None:
    """Return what a hosted database runs; None for a local one, or one that does not exist.

    desired: return what the current desired generation runs, which may still be in progress.
    """
    if db_path.org is None or db_path.org == 'local' or db_path.db is None:
        return None
    report = _get_db_report(db_path)
    if report is None:
        return None
    current = None if report.current is None else report.current.resources
    target = report.target_resources if desired else None
    fingerprint = None if current is None else current.fingerprint
    if target is not None and target.fingerprint is not None:
        fingerprint = target.fingerprint
    image = (target or current or DatabaseResources()).image
    return Deployment(fingerprint=fingerprint, image=image, default_image=report.default_image)


def default_image_mismatches(project_root: Path, config: DatabaseConfig | None, image: DefaultImage) -> list[str]:
    """What the project declares that the default image does not provide, one printable line each.

    Packages are compared by name: the image's versions apply, whatever version a requirement asks for.
    """
    mismatches: list[str] = []
    for file_name, entry in declared_requirements(project_root):
        name = requirement_name(entry)
        if name == 'pixeltable':
            # the image runs its own pixeltable, which the plan names
            continue
        if name is None:
            mismatches.append(f'{entry!r} in {file_name}')
        elif name not in image.packages:
            mismatches.append(f'{name} ({file_name})')
    if config is not None:
        system_packages = {normalized_name(p) for p in image.system_packages}
        for spec in config.system_dependencies or []:
            name = requirement_name(spec)
            if name is None or name not in system_packages:
                mismatches.append(f'system dependency {spec!r}')
        if config.python_version is not None and config.python_version != image.python_version:
            mismatches.append(f'python_version {config.python_version} (the image runs {image.python_version})')
        if config.uv_options:
            mismatches.append('uv_options')
    return mismatches


def default_image_target(config: DatabaseConfig | None) -> DatabaseTarget:
    """The target that puts this project's files on a database that runs the default image.

    It names no size and carries no settings: only `pxt db update` sends those.
    """
    return DatabaseTarget(
        fingerprint=project_fingerprint(_validated_project_root(), config, project_config={}), image='default'
    )


def upload_project(db_uri: str, *, wait: bool = True) -> list[GenerationReceipt]:
    """Put this project's files on the database at db_uri, which runs the default image, keeping its size and settings.

    Returns the receipts of the database generation that runs them.
    """
    db_path = _validated_db_uri(db_uri)
    config = Config.get().get_database_config(db_path)
    target = default_image_target(config)
    prepared = _prepare(db_path, target, blob=None)
    accepted, _ = _submit(db_path, config, target, expected_generation=prepared.generations.database or 0)
    return _settle(db_path, accepted, wait=wait)


def create_db_update_ops(target: DatabaseResources, current: DatabaseResources | None) -> list[DbChangeOp]:
    """The operations needed to reconcile current with target."""
    ops: list[DbChangeOp] = []

    if target.fingerprint is not None:
        if target.image == 'default':
            # the project's files alone, which the default image runs
            if current is None or current.fingerprint is None:
                ops.append(DbChangeOp.upload_archive())
            elif ProjectPart.ARCHIVE in target.fingerprint.compare(current.fingerprint):
                changes = target.fingerprint.changes(current.fingerprint, {ProjectPart.ARCHIVE})
                ops.append(DbChangeOp.upload_archive(changes))
        elif current is None or current.fingerprint is None or current.image == 'default':
            # nothing to diff against: the database is new, or still on the base or default image, which holds no
            # image built from the project
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
            if ProjectPart.CONFIG in changed:
                ops.append(DbChangeOp.config(target.fingerprint.changes(current.fingerprint, {ProjectPart.CONFIG})))

    current_capacity = DatabaseCapacity() if current is None else current.capacity()
    for field, value in current_capacity.changes(target.capacity()).model_dump(exclude_none=True).items():
        ops.append(DbChangeOp.capacity(field, getattr(current_capacity, field), value))

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
    target = target.model_copy(update={**desired.capacity().model_dump(), 'force_build_nonce': uuid.uuid4().hex})
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


def _prepare(
    db_path: catalog.Path,
    target: DatabaseTarget,
    *,
    blob: BlobRef | None,
    service_mutations: Sequence[ServiceMutation] = (),
) -> PrepareUpdateResponse:
    request = PrepareUpdateRequest(
        org=db_path.org,
        db=_db_name(db_path),
        database_target=target,
        service_mutations=list(service_mutations),
        blob=blob,
    )
    return PrepareUpdateResponse.model_validate(management_client.api_call(request))


@contextlib.contextmanager
def packaged_project(config: DatabaseConfig | None, target: DatabaseTarget) -> Iterator[tuple[Path, BlobRef]]:
    """Package the project whose files target's fingerprint describes, and remove the archive afterwards.

    Yields the archive's path and its blob reference. Raises if the project changed since it was fingerprinted.
    """
    if target.fingerprint is None:
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, 'a project was submitted without a fingerprint')
    project_root = _validated_project_root()
    if target.image != 'default':
        # the default image installs nothing from the project
        image_input_files(project_root)
        check_hosted_pixeltable(project_root)
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
        yield archive.path, _blob_ref(archive.path)
    finally:
        archive.path.unlink(missing_ok=True)


def upload_archive(prepared: PrepareUpdateResponse, path: Path, blob: BlobRef) -> bool:
    """Upload the archive at path where prepared says, unless the control plane holds it already.

    Returns whether it was uploaded.
    """
    if prepared.upload is None:
        return False
    _put_blob(prepared.upload, path, blob)
    return True


def _submit(
    db_path: catalog.Path, config: DatabaseConfig | None, target: DatabaseTarget, *, expected_generation: int
) -> tuple[list[GenerationReceipt], bool]:
    """Package the project, upload it unless the control plane holds it, and submit target against
    expected_generation, read before the project was packaged.

    Returns the resulting receipts, and whether the archive was uploaded.
    """
    with packaged_project(config, target) as (path, blob):
        prepared = _prepare(db_path, target, blob=blob)
        uploaded = upload_archive(prepared, path, blob)
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
        return accepted, uploaded


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
        fingerprint=project_fingerprint(
            _validated_project_root(), config, project_config=Config.get().create_project_config(config)
        ),
        pxt_md_version=metadata.VERSION,
        **config.capacity().model_dump(),
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
    uri_error = hosted_db_uri_error(path.org, path.db)
    if uri_error is not None:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, uri_error)
    return path


def _get_db_config(db_uri: catalog.Path) -> DatabaseConfig:
    config = Config.get().get_database_config(db_uri)
    if config is None:
        where = Config.get().project_config_file or 'the project configuration'
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f"no [[pixeltable.database]] entry for {db_uri.uri_str!r}; run 'pxt db diff {db_uri.uri_str} "
            f"--add-config-entry' to add it and see the plan, or add one to {where}:\n"
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
