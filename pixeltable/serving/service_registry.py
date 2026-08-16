"""The local record of which services are running, where their models are bound, and what they serve.

One file per service under $PIXELTABLE_HOME/services, in a tree mirroring the catalog directory the
service's models bind against: a service named 'ingest' bound at 'd/c' is recorded in
services/d/c/ingest.json. One file per service means concurrent starts never write the same file, and the
layout means the services deployed against one directory are a single non-recursive read.

The registry is derived, not authoritative: a record's liveness is the liveness of the process it names, so
a service that crashed or was killed is absent from the next read, with no cleanup pass and no state that
can contradict the process table.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pixeltable import catalog, exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.process import pid_alive

if TYPE_CHECKING:
    from pixeltable.serving import ServiceSpec


@dataclasses.dataclass(frozen=True)
class ServiceDeployment:
    """A service definition applied to a catalog directory, and the process serving it.

    The field names it shares with the hosted ServiceRecord carry the same meaning, so a local and a
    hosted deployment print identically. There is deliberately no state field: state is the liveness of pid.
    """

    service_name: str

    # the catalog directory to which the service models are bound
    base_path: str

    endpoint: str
    pid: int
    created_at: float

    # the file from which the service definition was loaded
    app_file: str

    spec: ServiceSpec

    @classmethod
    def read(cls, name: str, base_path: str = '') -> ServiceDeployment | None:
        """The named service bound at base_path, or None if it is not recorded or no longer running."""
        return cls._read(cls._record_file(name, base_path))

    @classmethod
    def list(cls, base_path: str | None = None, recursive: bool = False) -> list[ServiceDeployment]:
        """Every running service bound at base_path, plus those bound below it if recursive.

        Args:
            base_path: the catalog directory the services' models are bound to; None means the root.
            recursive: whether to include the services bound below base_path, which were deployed against
                those directories rather than against this one.
        """
        path = cls._dir('' if base_path is None else base_path)
        if not path.is_dir():
            return []
        files = sorted(path.rglob('*.json') if recursive else path.glob('*.json'))
        return [d for d in (cls._read(p) for p in files) if d is not None]

    @classmethod
    def _dir(cls, base_path: str = '') -> Path:
        return Env.get().services_dir.joinpath(*catalog.Path.parse(base_path, allow_empty_path=True).components)

    def write(self) -> None:
        """Record this service as running, replacing any record of the same name at the same target."""
        path = self._record_file(self.service_name, self.base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write a temp file in the target directory and rename it into place, so that a concurrent read sees
        # either the old record or the new one, never a partial one
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'{path.stem}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(dataclasses.asdict(self), f)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def remove(self) -> None:
        self._record_file(self.service_name, self.base_path).unlink(missing_ok=True)

    @classmethod
    def _record_file(cls, name: str, base_path: str = '') -> Path:
        """The file holding the record of the named service bound at base_path."""
        if not catalog.is_valid_identifier(name, allow_hyphens=True):
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} is not a valid service name')
        return cls._dir(base_path) / f'{name}.json'

    @classmethod
    def _read(cls, path: Path) -> ServiceDeployment | None:
        """The deployment recorded in path, or None if that record cannot be read or has no live process."""
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None  # a file that is absent, unreadable or not valid json records nothing
        if not isinstance(record, dict):
            return None
        # a record missing a field this version needs is one this version cannot manage; one carrying a field
        # this version does not know is read without it, and belongs to whichever version wrote it
        names = {f.name for f in dataclasses.fields(cls)}
        if not names <= record.keys():
            return None
        deployment = cls(**{name: value for name, value in record.items() if name in names})
        if type(deployment.pid) is not int:
            return None  # bool is an int, and a pid of True would probe pid 1
        return deployment if pid_alive(deployment.pid) else None
