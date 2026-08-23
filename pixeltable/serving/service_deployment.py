"""The local record of which services are running, where their models are bound, and what they serve.

One file per service under $PIXELTABLE_HOME/services, in a tree mirroring the catalog directory the
service's models bind against: a service named 'ingest' bound at 'd/c' is recorded in
services/d/c/ingest.json. One file per service means concurrent starts never write the same file, and the
layout means the services deployed against one directory are a single non-recursive read. The tree records
the path within one catalog, so a local service binds to the local catalog only: a pxt:// target is rejected
rather than recorded under the name a local service would use.

The registry is derived, not authoritative: a record's liveness is the liveness of the process it names, so
a service that crashed or was killed is absent from the next read, with no cleanup pass and no state that
can contradict the process table. A record names its process by pid and creation time, because the OS
reissues a pid once the process holding it exits.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.serving._app import load_service_routers
from pixeltable.utils.process import is_pid, pid_alive, process_timestamp

if TYPE_CHECKING:
    from pixeltable.serving import ServiceSpec

_logger = logging.getLogger(__name__)

# the locks that serialize start(), keyed by record file, and the lock guarding that dict
_service_locks: dict[Path, threading.Lock] = {}
_service_locks_guard = threading.Lock()


@dataclasses.dataclass(frozen=True)
class ServiceDeployment:
    """A locally running service derived from a FastAPIRouter instance.

    The field names it shares with the hosted ServiceRecord carry the same meaning, so a local and a
    hosted deployment print identically. The process state is derived, not recorded.
    """

    _HEALTH_TIMEOUT = 2.0
    _STOP_TIMEOUT = 10.0
    _STARTUP_TIMEOUT = 60.0
    _LOG_TAIL_LINES = 40

    service_name: str

    # the catalog directory to which the service models are bound
    base_path: str

    endpoint: str
    pid: int

    # creation time of pid, None where the platform does not report one
    process_started_at: float | None

    # the file from which the service definition was loaded
    app_file: str

    spec: ServiceSpec

    @classmethod
    def log_path(cls, name: str, base_path: str) -> Path:
        components = catalog.Path.parse(base_path, allow_empty_path=True).components
        return Config.get().home.joinpath('logs', 'services', *components, f'{name}.log')

    @classmethod
    def read(cls, name: str, base_path: str = '') -> ServiceDeployment | None:
        """The named service bound at base_path, or None if it is not recorded or no longer running."""
        return cls._read(cls._record_file(name, base_path))

    @classmethod
    def list(cls, base_path: str | None = None, recursive: bool = False) -> list[ServiceDeployment]:
        """Every running service bound at base_path, plus those bound below it if recursive."""
        path = cls._dir('' if base_path is None else base_path)
        if not path.is_dir():
            return []
        files = sorted(path.rglob('*.json') if recursive else path.glob('*.json'))
        return [d for d in (cls._read(p) for p in files) if d is not None]

    @classmethod
    def _dir(cls, base_path: str = '') -> Path:
        """Return the local directory path containing the record file of this deployment."""
        parsed_path = catalog.Path.parse(base_path, allow_empty_path=True)
        if not parsed_path.is_local:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'A local service binds its models to the local catalog; {base_path!r} names another one.',
            )
        return Env.get().services_dir.joinpath(*parsed_path.components)

    @classmethod
    def start(cls, app_file: str, service_name: str, base_path: str = '') -> ServiceDeployment:
        """
        Atomically start and record the named service from app_file with its models bound at base_path, or return the
        running one.

        Raises RequestError if the service cannot be served, and Error if its process never becomes healthy.
        """
        with cls._service_lock(service_name, base_path):
            return cls._start(app_file, service_name, base_path)

    @classmethod
    def _service_lock(cls, service_name: str, base_path: str) -> threading.Lock:
        """The lock that serializes start() for one service at one target."""
        path = cls._record_file(service_name, base_path)
        with _service_locks_guard:
            return _service_locks.setdefault(path, threading.Lock())

    @classmethod
    def _start(cls, app_file: str, service_name: str, base_path: str) -> ServiceDeployment:
        """Start the service and wait for it to report healthy, with cls._start_lock() held."""
        deployment = cls.read(service_name, base_path)
        if deployment is not None and deployment.health_ok():
            return deployment

        # fail here, in the caller's process, on everything that can be detected without serving: an app file
        # that does not declare the service is a request error, not a process that dies in the background
        services = load_service_routers(app_file)
        if service_name not in services:
            declared = ', '.join(sorted(services))
            raise excs.NotFoundError(
                excs.ErrorCode.SERVICE_NOT_FOUND,
                f'{app_file} contains no FastAPIRouter named {service_name!r}; it declares: {declared}',
            )

        log_path = cls.log_path(service_name, base_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            '-m',
            'pixeltable.serving.service_runner',
            '--app-file',
            app_file,
            '--name',
            service_name,
            '--base-path',
            base_path,
        ]

        # the service outlives this call, so it must not inherit our stdio: attached to a pipe it would block
        # the reader on EOF, and attached to a terminal it would write into that session. Its own session keeps
        # signals sent to the launching process away from it.
        with open(log_path, 'a', encoding='utf-8') as log_file:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )

        deadline = time.monotonic() + cls._STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            deployment = cls.read(service_name, base_path)
            if deployment is not None and deployment.health_ok():
                return deployment
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        msg = f'Service {service_name!r} failed to start within {cls._STARTUP_TIMEOUT:.0f}s'
        returncode = proc.poll()
        if returncode is None:
            msg += '; its process is still running but never reported healthy'
            # take the process down and drop the record it wrote, so that nothing is left claiming this target
            cls._terminate(proc)
            unhealthy = cls._parse_record(cls._record_file(service_name, base_path))
            if unhealthy is not None and unhealthy.pid == proc.pid:
                unhealthy.remove()
        else:
            msg += f'; its process exited with code {returncode}'
        tail = cls._tail_log(log_path)
        if tail != '':
            msg += f'\n--- service log tail ---\n{tail}'
        raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, msg)

    @classmethod
    def _terminate(cls, proc: subprocess.Popen) -> None:
        """Stop proc and reap it, escalating to kill if it does not exit."""
        proc.terminate()
        try:
            proc.wait(timeout=cls._STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=cls._STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _logger.warning('Service process %d did not exit', proc.pid)

    @classmethod
    def _tail_log(cls, path: Path) -> str:
        """The last lines of the log at path, or the empty string if it cannot be read."""
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return ''
        return '\n'.join(lines[-cls._LOG_TAIL_LINES :])

    @classmethod
    def create(
        cls, service_name: str, base_path: str, port: int, app_file: str, spec: ServiceSpec
    ) -> ServiceDeployment:
        """Create and write a new service record."""
        d = ServiceDeployment(
            service_name=service_name,
            base_path=base_path,
            endpoint=f'http://127.0.0.1:{port}',
            pid=os.getpid(),
            process_started_at=process_timestamp(os.getpid()),
            app_file=str(Path(app_file).resolve()),
            spec=spec,
        )

        path = d._record_file(d.service_name, d.base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write a temp file in the target directory and rename it into place, so that a concurrent read sees
        # either the old record or the new one
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'{path.stem}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(dataclasses.asdict(d), f)
            os.replace(tmp_name, path)
            return d
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def stop(self) -> None:
        """Terminate this deployment and remove its record."""
        if not self.is_live():
            # the recorded process is gone; its pid may belong to something else by now
            self.remove()
            return

        pid: int | None = self.pid
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # POSIX raises ProcessLookupError once the process is gone; Windows raises PermissionError or OSError
            # from TerminateProcess for an already-exited process. In every case there is nothing left to wait on.
            pid = None

        if pid is not None:
            deadline = time.monotonic() + self._STOP_TIMEOUT
            while time.monotonic() < deadline and self.is_live():
                # reap promptly if the service is our own child, so its zombie isn't mistaken for a live process;
                # a no-op when another process launched it, in which case init reaps it
                if hasattr(os, 'WNOHANG'):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except (ChildProcessError, OSError):
                        pass
                time.sleep(0.05)
            if self.is_live():
                # graceful shutdown overran the timeout; SIGKILL is absent on Windows, where os.kill() already
                # terminates unconditionally
                try:
                    os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        self.remove()

    def remove(self) -> None:
        """Remove this deployment's record."""
        path = self._record_file(self.service_name, self.base_path)
        recorded = self._parse_record(path)
        # make sure this is our deployment
        if recorded is None or (recorded.pid, recorded.process_started_at) == (self.pid, self.process_started_at):
            path.unlink(missing_ok=True)

    def health_ok(self) -> bool:
        try:
            # every FastAPI application serves its schema
            return httpx.get(f'{self.endpoint}/openapi.json', timeout=self._HEALTH_TIMEOUT).status_code == 200
        except httpx.HTTPError:
            return False

    @classmethod
    def _record_file(cls, name: str, base_path: str = '') -> Path:
        """The file holding the record of the named service bound at base_path."""
        if not catalog.is_valid_identifier(name, allow_hyphens=True):
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} is not a valid service name')
        return cls._dir(base_path) / f'{name}.json'

    @classmethod
    def _read(cls, record_file_path: Path) -> ServiceDeployment | None:
        """The deployment recorded in record_file_path, or None if that record cannot be read or has no live process."""
        recorded = cls._parse_record(record_file_path)
        return recorded if recorded is not None and recorded.is_live() else None

    @classmethod
    def _parse_record(cls, record_file_path: Path) -> ServiceDeployment | None:
        """The deployment recorded in record_file_path."""
        try:
            record = json.loads(record_file_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict):
            return None
        # a record missing a field is one this version cannot manage; one carrying a field
        # this version does not know is read without it, and belongs to whichever version wrote it
        names = {f.name for f in dataclasses.fields(cls)}
        if not names <= record.keys():
            return None
        if not is_pid(record['pid']):
            return None
        return cls(**{name: value for name, value in record.items() if name in names})

    def is_live(self) -> bool:
        """Whether the process this record was written for is still running.

        A pid alone does not identify it: the OS reissues the pid of an exited process, so a record left
        behind by a crash would otherwise report whichever process inherited it.
        """
        if not pid_alive(self.pid):
            return False
        if self.process_started_at is None:
            return True  # nothing to compare against: the platform reported no creation time when writing
        return process_timestamp(self.pid) == self.process_started_at
