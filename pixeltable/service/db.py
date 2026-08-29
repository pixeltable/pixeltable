"""The operations that reconcile a hosted database with the project entry declaring it."""

from __future__ import annotations

from typing import Any

import pydantic

from pixeltable.config import DatabaseConfig
from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbChangeOp

_UPLOAD_TIMEOUT = 300


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


def db_diff(db_uri: str) -> types.DbPlan:
    """Diff the database at db_uri with the corresponding DatabaseConfig in the project configuration."""
    from pixeltable.service.db import compare_db

    entry, project_dir, org, db = _db_entry(db_uri)
    current = _get_db_state(org, db)
    if current is None:
        return _db_plan(db_uri, None, [], [])

    fingerprint = project_fingerprint(project_dir, entry)
    ops, not_compared = _compare_db(current, entry, fingerprint)
    return _db_plan(db_uri, current.state, ops, not_compared)

def _get_db_state(org: str, db: str) -> DatabaseState | None:
    try:
        response = management_client.api_call(GetDbRequest(org=org, db=db))
    except excs.ExternalServiceError as e:
        if e.provider_http_status_code != 404:
            raise
        return None
    return DatabaseState.model_validate(response.get('database', response))


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

    for key in sorted(config.secrets or {}):
        if key not in current.secret_keys:
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


def _db_config(project_dir: Path, db_name: str | None = None) -> DatabaseConfig | None:
    """The project's configuration for the named database, or for its only one when name is absent."""
    pxt_toml = _load_from_toml(project_dir / 'pixeltable.toml', ['pixeltable', 'database'], db_name)
    if pxt_toml is not None:
        return pxt_toml

    py_toml = _load_from_toml(project_dir / 'pyproject.toml', ['tool', 'pixeltable', 'database'], db_name)
    if py_toml is not None:
        return py_toml

    # Fall back on system config.
    # TODO: This should be removed, but doing it now will break a bunch of tests
    return _select_database(Config.get().get_value('database', list), db_name)


def _load_from_toml(toml_path: Path, resolution: list[str], db_name: str | None) -> DatabaseConfig | None:
    if not toml_path.is_file():
        return None

    try:
        cfg = toml.load(toml_path)
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid TOML in {toml_path.name}: {e}') from e

    for key in resolution:
        if not isinstance(cfg, dict) or key not in cfg:
            return None
        cfg = cfg[key]

    entries = cfg if isinstance(cfg, list) else [cfg]  # a single table is one entry, written [pixeltable.database]
    try:
        validated = [DatabaseConfig.model_validate(entry) for entry in entries]
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [[pixeltable.database]] in {toml_path.name}: {e}'
        ) from e
    # a name addresses one entry, so two entries sharing one leave the target ambiguous; Config enforces the
    # same rule for the entries it reads
    seen: set[str | None] = set()
    for db in validated:
        if db.name in seen:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Duplicate [[pixeltable.database]] name {db.name!r} in {toml_path.name}',
            )
        seen.add(db.name)
    return _select_database(validated, db_name)


def _select_database(databases: list[DatabaseConfig] | None, name: str | None) -> DatabaseConfig | None:
    """The entry name addresses; the only entry when name addresses none, so a lone entry configures any target."""
    if databases is None or len(databases) == 0:
        return None
    if name is not None:
        named = next((db for db in databases if db.name == name), None)
        if named is not None:
            return named
    return databases[0] if len(databases) == 1 else None


def upload_project_archive(project_dir: Path, org: str, db: str, *, show_progress: bool = False) -> str:
    """Store the project as the archive the named database's pods and image builds read.

    Returns the control plane's key for it. A project whose digest names an archive the control plane
    already holds is neither packaged nor uploaded.
    """
    db_name = f'pxt://{org}:{db}'
    digest = project_fingerprint(project_dir, _db_config(project_dir, db_name)).archive_digest()
    response = GetProjectUploadUrlResponse.model_validate(
        management_client.api_call(GetProjectUploadUrlRequest(org=org, db=db, digest=digest))
    )
    if response.presigned_url is None:
        return response.project_key

    archive_path = create_project_archive(project_dir, show_progress=show_progress, db_name=db_name)
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
