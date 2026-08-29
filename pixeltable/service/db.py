"""The operations that reconcile a hosted database with the project entry declaring it."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import pydantic

from pixeltable.config import DatabaseConfig
from pixeltable.utils.project import ProjectFingerprint

_UPLOAD_TIMEOUT = 300

Severity = Literal['additive', 'destructive', 'unsupported']

# what an operation acts on. The two artifacts are separate: 'image' is the environment the pods run on,
# 'project' the sources they fetch, and a source edit moves only the second.
Target = Literal['image', 'project', 'capacity', 'secret', 'placement']


class DbChangeOp(TypedDict):
    """One operation reconciling a hosted database with the entry declaring it.

    Mirrored by pixeltable_cli.db_types.DbChangeOp; adding, removing or retyping a field here means doing
    the same there.
    """

    target: Target
    name: str  # capacity and placement: the field; secret: the key; image and project: their own name
    op: Literal['add', 'drop', 'alter']
    severity: Severity
    description: str  # one sentence, ready to print
    details: dict[str, str]  # 'from' and 'to' for an alter, 'changes' for the image
    requires_restart: bool  # whether applying this interrupts what the database is serving

def _image_built_op() -> DbChangeOp:
    """The operation for an image build the caller asked for rather than one a difference calls for."""
    return {
        'target': 'image',
        'name': 'image',
        'op': 'alter',
        'severity': 'additive',
        'description': "the image will be rebuilt from the project's environment",
        'details': {},
        'requires_restart': True,
    }


def _project_shipped_op() -> DbChangeOp:
    """The operation for shipping the project the caller named rather than one a difference calls for."""
    return {
        'target': 'project',
        'name': 'project',
        'op': 'alter',
        'severity': 'additive',
        'description': 'the project will be shipped',
        'details': {},
        'requires_restart': True,
    }


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


def compare_db(
    current: DatabaseState, config: DatabaseConfig, fingerprint: ProjectFingerprint
) -> tuple[list[DbChangeOp], list[str]]:
    """The operations that make current match config + fingerprint."""
    ops: list[DbChangeOp] = []
    not_compared: list[str] = []

    if current.fingerprint is None:
        not_compared.append('image')
    else:
        if fingerprint.image_needed(current.fingerprint):
            ops.append(_image_op(fingerprint.changes(current.fingerprint, 'image')))
        if fingerprint.archive_needed(current.fingerprint):
            ops.append(_project_op(fingerprint.changes(current.fingerprint, 'archive')))

    for field, config_value, current_value in (
        ('cpu', config.cpu, current.cpu),
        ('memory_mb', config.memory_mb, current.memory_mb),
        ('disk_gb', config.disk_gb, current.disk_gb),
        ('workers', config.workers, len(current.worker_status)),
    ):
        if config_value is None or config_value == current_value:
            continue
        ops.append(_capacity_op(field, current_value, config_value))

    for key in sorted(config.secrets or {}):
        if key not in current.secret_keys:
            ops.append(_secret_op(key, 'add'))
    for key in sorted(set(current.secret_keys) - set(config.secrets or {})):
        ops.append(_secret_op(key, 'drop'))

    for field, config_value, current_value in (
        ('location', config.location, current.location),
        ('region', config.region, current.region),
    ):
        if config_value is None or current_value is None or config_value == current_value:
            continue
        ops.append(_placement_op(field, current_value, config_value))

    if config.vars is not None:
        # a hosted database has nowhere to keep a var that is not a secret
        not_compared.append('vars')

    return ops, not_compared


def _image_op(changes: list[str]) -> DbChangeOp:
    """The operation for an environment that differs from the one the current image holds.

    changes are the causes, from ProjectFingerprint.changes().
    """
    return {
        'target': 'image',
        'name': 'image',
        'op': 'alter',
        # the image is replaced, not removed: what the database serves is unchanged until a pod restarts
        'severity': 'additive',
        'description': f'the image will be rebuilt: {_summary(changes)}',
        'details': {'changes': '; '.join(changes)},
        'requires_restart': True,
    }


def _project_op(changes: list[str]) -> DbChangeOp:
    """The operation for sources the database's pods are not running.

    changes are the causes, from ProjectFingerprint.changes().
    """
    return {
        'target': 'project',
        'name': 'project',
        'op': 'alter',
        # what the pods serve is unchanged; they restart to run the new sources
        'severity': 'additive',
        'description': f'the project will be shipped again: {_summary(changes)}',
        'details': {'changes': '; '.join(changes)},
        'requires_restart': True,
    }


def _summary(changes: list[str]) -> str:
    """The first few causes, printable, with the rest counted."""
    if len(changes) <= 3:
        return '; '.join(changes)
    return f'{"; ".join(changes[:3])} and {len(changes) - 3} more'


def _capacity_op(field: str, current: float | int | None, declared: float | int) -> DbChangeOp:
    was = 'unreported' if current is None else str(current)
    return {
        'target': 'capacity',
        'name': field,
        'op': 'alter',
        'severity': 'destructive' if current is not None and declared < current else 'additive',
        'description': f'{field} will be {declared} rather than {was}, which restarts the database',
        'details': {'from': was, 'to': str(declared)},
        'requires_restart': True,
    }


def _placement_op(field: str, current: str, declared: str) -> DbChangeOp:
    """The operation for a field fixed at creation, which no update can carry out."""
    return {
        'target': 'placement',
        'name': field,
        'op': 'alter',
        'severity': 'unsupported',
        'description': f'{field} is {current!r} and cannot be changed to {declared!r}; create a database there instead',
        'details': {'from': current, 'to': declared},
        'requires_restart': False,
    }


def _secret_op(key: str, op: Literal['add', 'drop']) -> DbChangeOp:
    if op == 'add':
        return {
            'target': 'secret',
            'name': key,
            'op': 'add',
            'severity': 'additive',
            'description': f'secret {key!r} will be set',
            'details': {},
            'requires_restart': False,
        }
    return {
        'target': 'secret',
        'name': key,
        'op': 'drop',
        'severity': 'destructive',
        'description': f'secret {key!r} will be deleted, and code reading it will fail',
        'details': {},
        'requires_restart': False,
    }


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
