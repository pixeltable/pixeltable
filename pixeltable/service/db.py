from __future__ import annotations

import base64
import dataclasses
import hashlib
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Literal

from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.config import Config, DatabaseConfig
from pixeltable.env import Env
from pixeltable.service import management_client, receipts
from pixeltable.service.db_md import DatabaseResources
from pixeltable.service.management_protocol import (
    LOCAL_DB_CLOUD_NAME_PREFIX,
    BlobRef,
    BlobUpload,
    DatabaseReport,
    DatabaseTarget,
    DbReceiptResponse,
    DeleteDbRequest,
    GetDbRequest,
    GetDbResponse,
    GetUdfServersRequest,
    GetUdfServersResponse,
    ListOrgsRequest,
    ListOrgsResponse,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    ReportServiceInstanceRequest,
    RestartDbRequest,
    StartDbRequest,
    StopDbRequest,
    SubmitUpdateRequest,
)
from pixeltable.service.resources import DatabaseCapacity
from pixeltable.utils.project import (
    ProjectFingerprint,
    ProjectPart,
    check_hosted_pixeltable,
    image_input_files,
    package_project_archive,
    project_fingerprint,
)
from pixeltable_cli.types import DbChangeOp, DbPlan, GenerationReceipt, OpStatus
from pixeltable_cli.utils import hosted_db_uri_error

_logger = logging.getLogger('pixeltable')

_UPLOAD_TIMEOUT = 300
_UPLOAD_ATTEMPTS = 2

_DB_DESTRUCTIVE_HINT = "Re-run 'pxt db update' with --allow-destructive to apply these changes."

DbAction = Literal['start', 'stop', 'restart', 'delete']

# TODO: replace with the names of the local databases, once an install can have more than one
LOCAL_DB_NAME = 'main'

_LIFECYCLE_REQUESTS = {
    'start': StartDbRequest,
    'stop': StopDbRequest,
    'restart': RestartDbRequest,
    'delete': DeleteDbRequest,
}


@dataclasses.dataclass(frozen=True)
class LocalDatabase:
    """A local database, identified by its install and its name within the install."""

    # SystemInfoMd.install_id
    install_id: str
    name: str

    def cloud_db_name(self) -> str:
        """Return the database's name in the control plane: name and a hash, unique within an organization."""
        digest = hashlib.sha256(f'{self.install_id}:{self.name}'.encode()).hexdigest()[:12]
        # a name in the control plane has at most 29 characters of [a-z0-9-]; the prefix, a hyphen and digest
        # leave 10 for name
        label = ''.join(c if c.isascii() and c.isalnum() else '-' for c in self.name.lower())
        label = label[:10].strip('-')
        if label == '':
            return LOCAL_DB_CLOUD_NAME_PREFIX + digest
        return f'{LOCAL_DB_CLOUD_NAME_PREFIX}{label}-{digest}'


def local_database() -> LocalDatabase:
    """Return the install's local database, which this call starts if it isn't running."""
    env = Env.get()
    env.ensure_connected()
    return LocalDatabase(install_id=metadata.install_id(env.engine), name=LOCAL_DB_NAME)


def local_db_uri() -> str:
    """Return the uri under which the control plane registers the local database.

    The database belongs to the organization of the configured credential.
    """
    # TODO: remove this check one release after the one that dropped the pixeltable.db_uri setting
    stale_db_uri = Config.get().get_string_value('db_uri')
    if stale_db_uri is not None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            'The pixeltable.db_uri setting is no longer supported, and a command without a uri acts on the local '
            f'database. Remove the setting, and pass the uri to act on the hosted database: {stale_db_uri}',
        )
    return f'pxt://{_local_db_org()}:{local_database().cloud_db_name()}'


def _local_db_org() -> str:
    """Return the name of the organization with the local database's UDF servers: the credential's only one."""
    orgs = ListOrgsResponse.model_validate(management_client.api_call(ListOrgsRequest())).orgs
    if len(orgs) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            'The UDF servers of a local database belong to an organization, and your sign-in has no organization. '
            'Create one in Pixeltable Cloud first.',
        )
    if len(orgs) > 1:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'The UDF servers of a local database belong to one organization, and your sign-in has {len(orgs)}. '
            'Set `PIXELTABLE_API_KEY` to an API key of the organization.',
        )
    return orgs[0].org


def local_db_udf_servers() -> GetUdfServersResponse:
    """Return the local database's UDF servers. Raises if they run another version of the project."""
    if Config.get().project_root is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            'The pixeltable.udf setting needs a project in order to create UDF servers. '
            'Run `pxt init` to write a project configuration, then `pxt db update`.',
        )
    request = GetUdfServersRequest(org=_local_db_org(), db=local_database().cloud_db_name())
    try:
        servers = GetUdfServersResponse.model_validate(management_client.api_call(request))
    except excs.ExternalServiceError as exc:
        if exc.provider_http_status_code != 404:
            raise
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            'The UDFs in the pixeltable.udf setting have no servers. Run `pxt db update` to deploy them.',
        ) from None
    fingerprint = _db_target(_local_db_config()).fingerprint
    assert fingerprint is not None
    if len(fingerprint.compare(servers.fingerprint)) > 0:
        changes = '\n'.join(f'  {line}' for line in fingerprint.changes(servers.fingerprint))
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'The UDF servers of the pixeltable.udf setting run another version of the project:\n{changes}\n'
            'Run `pxt db update` to deploy this one.',
        )
    return servers


def db_diff(db_uri: str) -> DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    db_path = _validated_db_uri(db_uri)
    _check_udf_setting(db_path)
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


def create_db_update_ops(
    target: DatabaseResources, current: DatabaseResources | None, *, has_pods: bool = True
) -> list[DbChangeOp]:
    """The operations needed to reconcile current with target.

    has_pods: False for a local database
    """
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
            if ProjectPart.CONFIG in changed:
                changes = target.fingerprint.changes(current.fingerprint, {ProjectPart.CONFIG})
                ops.append(DbChangeOp.config(changes, has_pods=has_pods))

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
    _check_udf_setting(db_path)
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
            create_db_update_ops(
                target, None if current is None else current.resources, has_pods=_as_local_database(db_path) is None
            ),
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


def _local_db_config() -> DatabaseConfig:
    config = Config.get().get_database_config(catalog.Path())
    return DatabaseConfig() if config is None else config


def _as_local_database(db_path: catalog.Path) -> LocalDatabase | None:
    """Return the local database if db_path is local_db_uri(), and None for a hosted database.

    Raises an error for the name of another install's local database.
    """
    assert db_path.db is not None
    # a name without the prefix belongs to a hosted database; local_database() would start the local one
    if not db_path.db.startswith(LOCAL_DB_CLOUD_NAME_PREFIX):
        return None
    local = local_database()
    if db_path.db != local.cloud_db_name():
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'{db_path.uri_str} is not the local database of this install: the prefix {LOCAL_DB_CLOUD_NAME_PREFIX!r} '
            'is reserved for local databases.',
        )
    return local


def _check_udf_setting(db_path: catalog.Path) -> None:
    """Raise an error for an invalid pixeltable.udf setting, or for an empty one if db_path is the local database.

    An update of a local database deploys the UDF servers of that setting, and nothing else.
    """
    if len(Env.get().udf_configs()) == 0 and _as_local_database(db_path) is not None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            'The local database has nothing to deploy: the project has no pixeltable.udf setting. '
            'For a hosted database, pass its uri: pxt://org:db.',
        )


def _get_db_config(db_uri: catalog.Path) -> DatabaseConfig:
    if _as_local_database(db_uri) is not None:
        return _local_db_config()
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
