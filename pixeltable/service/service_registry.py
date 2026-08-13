"""The local record of which services are running, where their models are bound, and what they serve.

One file per service under `$PIXELTABLE_HOME/services`, in a tree mirroring the catalog directory the
service's models bind against: a service named 'ingest' bound at 'd/c' is recorded in
`services/d/c/ingest.json`. One file per service means concurrent starts never write the same file, and the
layout means the services deployed against one directory are a single non-recursive read.

The registry is derived, not authoritative: a record's liveness is the liveness of the process it names, so
a service that crashed or was killed is absent from the next read, with no cleanup pass and no state that
can contradict the process table.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.utils.process import pid_alive

if TYPE_CHECKING:
    from pixeltable.serving import ServiceSpec

_DIR_NAME = 'services'


class ServiceDeployment(TypedDict):
    """A service definition applied to a catalog directory, and the process serving it.

    The field names it shares with the hosted `ServiceRecord` carry the same meaning, so a local and a
    hosted deployment print identically. There is deliberately no state field: `state` is the liveness of
    `pid`.
    """

    service_name: str

    # the catalog directory the definition's models are bound to
    base_path: str

    endpoint: str
    pid: int
    created_at: float  # when the process serving this deployment was started

    # the file the definition was loaded from, which a diff reports the deployment as coming from
    app_file: str

    spec: ServiceSpec


def services_dir() -> Path:
    """The root of the registry tree."""
    return Config.get().home / _DIR_NAME


def target_dir(base_path: str = '') -> Path:
    """The registry directory holding the services bound at `base_path`.

    Only the in-catalog components of `base_path` are used, so a path naming a database resolves to the same
    directory as the bare path: the registry records local deployments only.
    """
    return services_dir().joinpath(*catalog.Path.parse(base_path, allow_empty_path=True).components)


def _record_file(name: str, base_path: str = '') -> Path:
    """The file holding the record of the named service bound at `base_path`."""
    if not catalog.is_valid_identifier(name, allow_hyphens=True):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} is not a valid service name')
    return target_dir(base_path) / f'{name}.json'


def save(deployment: ServiceDeployment) -> None:
    """Record a running service, replacing any record of the same name at the same target."""
    path = _record_file(deployment['service_name'], deployment['base_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    # write a temp file in the target directory and rename it into place, so that a concurrent read sees
    # either the old record or the new one, never a partial one
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'{path.stem}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(deployment, f)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def get(name: str, base_path: str = '') -> ServiceDeployment | None:
    """The named service bound at `base_path`, or None if it is not recorded or no longer running."""
    return _read(_record_file(name, base_path))


def list_at(base_path: str = '') -> list[ServiceDeployment]:
    """Every running service bound at `base_path`.

    A service bound below `base_path` belongs to that directory rather than to this one, and is not
    included.
    """
    return _read_dir(target_dir(base_path))


def list_all() -> list[ServiceDeployment]:
    """Every running service, at any target."""
    root = services_dir()
    if not root.is_dir():
        return []
    return [d for d in (_read(p) for p in sorted(root.rglob('*.json'))) if d is not None]


def remove(name: str, base_path: str = '') -> None:
    """Forget the named service bound at `base_path`, whether or not it is running. Does not stop it."""
    _record_file(name, base_path).unlink(missing_ok=True)


def _read_dir(path: Path) -> list[ServiceDeployment]:
    """Every running service recorded directly in `path`."""
    if not path.is_dir():
        return []
    return [d for d in (_read(p) for p in sorted(path.glob('*.json'))) if d is not None]


def _read(path: Path) -> ServiceDeployment | None:
    """The deployment recorded in `path`, or None if that record is missing, unreadable, or has no live process."""
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None  # a file that is absent, unreadable or not valid json records nothing
    if not isinstance(record, dict) or not isinstance(record.get('pid'), int):
        return None
    return cast(ServiceDeployment, record) if pid_alive(record['pid']) else None
