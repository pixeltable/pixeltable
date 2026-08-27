"""The service instances running on this machine: starting them, recording them, stopping them.

One record file per instance under $PIXELTABLE_HOME/services, in a tree mirroring the catalog directory the
instance's models are bound to: an instance of 'ingest' bound at 'd/c' is recorded in
services/d/c/ingest.json. One file per instance means concurrent starts never write the same file, and the
layout means the instances serving one directory are a single non-recursive read. The tree records the path
within one catalog, so this manager serves the local catalog only.

The records are derived, not authoritative: an instance's liveness is the liveness of the process it names,
so one that crashed or was killed is absent from the next read, with no cleanup pass and no state that can
contradict the process table. A record names its process by pid and creation time, because the OS reissues a
pid once the process holding it exits.
"""

from __future__ import annotations

import abc
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
import pydantic

from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.serving._app import load_service_routers
from pixeltable.utils.process import is_pid, pid_alive, process_timestamp

from .service_instance import ServiceInstance, ServiceInstanceRecord

if TYPE_CHECKING:
    from ._spec import ServiceSpec

_logger = logging.getLogger(__name__)

# the locks that serialize start(), keyed by record file, and the lock guarding that dict
_service_locks: dict[Path, threading.Lock] = {}
_service_locks_guard = threading.Lock()


class ServiceManagerBase(abc.ABC):
    """The container of the service instances of one catalog."""

    catalog_uri: catalog.Path

    @abc.abstractmethod
    def get(self, name: str, base_path: str = '') -> ServiceInstance | None:
        """The instance of that name serving base_path."""

    @abc.abstractmethod
    def list(self, base_path: str = '', recursive: bool = False) -> list[ServiceInstance]:
        """The instances serving base_path, plus those serving below it if recursive."""

    @abc.abstractmethod
    def start(self, app_file: str, name: str, base_path: str = '', *, otel: bool = False) -> ServiceInstance:
        """Make the named router in app_file serve base_path, and return its instance."""

    @abc.abstractmethod
    def stop(self, instance: ServiceInstance) -> None:
        """Stop instance and forget it."""


def get_manager(target: str = '') -> ServiceManagerBase:
    """The manager of the service instances in the catalog that target names."""
    path = catalog.Path.parse(target, allow_empty_path=True)
    if path.is_local:
        return ServiceManager()
    raise excs.RequestError(
        excs.ErrorCode.UNSUPPORTED_OPERATION, f'{target!r}: services in a hosted database are not supported yet.'
    )


class ServiceManager(ServiceManagerBase):
    """The container of the service instances running on this machine, each one a process this manager
    starts and stops.
    """

    _HEALTH_TIMEOUT = 2.0
    _STOP_TIMEOUT = 10.0
    _STARTUP_TIMEOUT = 60.0
    _LOG_TAIL_LINES = 40

    def __init__(self) -> None:
        self.catalog_uri = catalog.ROOT_PATH

    def get(self, name: str, base_path: str = '') -> ServiceInstance | None:
        return self._read(self._record_file(name, base_path))

    def list(self, base_path: str = '', recursive: bool = False) -> list[ServiceInstance]:
        path = self._dir(base_path)
        if not path.is_dir():
            return []
        files = sorted(path.rglob('*.json') if recursive else path.glob('*.json'))
        return [i for i in (self._read(p) for p in files) if i is not None]

    def start(self, app_file: str, name: str, base_path: str = '', *, otel: bool = False) -> ServiceInstance:
        """
        Atomically start and record the named service from app_file with its models bound at base_path, or return the
        running one.

        Raises RequestError if the service cannot be served, and Error if its process never becomes healthy.
        """
        with self._service_lock(name, base_path):
            return self._start(app_file, name, base_path, otel)

    def stop(self, instance: ServiceInstance) -> None:
        record = instance.record
        pid: int | None = record.pid
        assert pid is not None  # a record this manager wrote names its process
        if not self._is_live(record):
            # the recorded process is gone; its pid may belong to something else by now
            self.remove(record)
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # POSIX raises ProcessLookupError once the process is gone; Windows raises PermissionError or OSError
            # from TerminateProcess for an already-exited process. In every case there is nothing left to wait on.
            pid = None

        if pid is not None:
            deadline = time.monotonic() + self._STOP_TIMEOUT
            while time.monotonic() < deadline and self._is_live(record):
                # reap promptly if the service is our own child, so its zombie isn't mistaken for a live process;
                # a no-op when another process launched it, in which case init reaps it
                if hasattr(os, 'WNOHANG'):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except (ChildProcessError, OSError):
                        pass
                time.sleep(0.05)
            if self._is_live(record):
                # graceful shutdown overran the timeout; SIGKILL is absent on Windows, where os.kill() already
                # terminates unconditionally
                try:
                    os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        self.remove(record)

    def create(
        self, service_name: str, base_path: str, port: int, app_file: str, spec: ServiceSpec, otel: bool = False
    ) -> ServiceInstanceRecord:
        """Write the record of the instance this process serves."""
        record = ServiceInstanceRecord(
            service_name=service_name,
            base_path=base_path,
            endpoint=f'http://127.0.0.1:{port}',
            pid=os.getpid(),
            process_started_at=process_timestamp(os.getpid()),
            app_file=str(Path(app_file).resolve()),
            spec=spec,
            otel=otel,
        )

        path = self._record_file(record.service_name, record.base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write a temp file in the target directory and rename it into place, so that a concurrent read sees
        # either the old record or the new one
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'{path.stem}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                # 'state' is left out: the process table decides whether this instance serves
                f.write(record.model_dump_json(exclude={'state'}))
            os.replace(tmp_name, path)
            return record
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def remove(self, record: ServiceInstanceRecord) -> None:
        """Remove record's file, unless another process has since written its own there."""
        path = self._record_file(record.service_name, record.base_path)
        recorded = self._parse_record(path)
        # make sure the record is ours
        if recorded is None or (recorded.pid, recorded.process_started_at) == (record.pid, record.process_started_at):
            path.unlink(missing_ok=True)

    def _log_path(self, name: str, base_path: str) -> Path:
        components = catalog.Path.parse(base_path, allow_empty_path=True).components
        return Config.get().home.joinpath('logs', 'services', *components, f'{name}.log')

    def _dir(self, base_path: str = '') -> Path:
        """The directory holding the record files of the instances serving base_path."""
        parsed_path = catalog.Path.parse(base_path, allow_empty_path=True)
        if not parsed_path.is_local:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'An instance on this machine binds its models to the local catalog; {base_path!r} names another one.',
            )
        return Env.get().services_dir.joinpath(*parsed_path.components)

    def _record_file(self, name: str, base_path: str = '') -> Path:
        """The file holding the record of the named instance serving base_path."""
        if not catalog.is_valid_identifier(name, allow_hyphens=True):
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} is not a valid service name')
        return self._dir(base_path) / f'{name}.json'

    def _service_lock(self, name: str, base_path: str) -> threading.Lock:
        """The lock that serializes start() for one service at one target."""
        path = self._record_file(name, base_path)
        with _service_locks_guard:
            return _service_locks.setdefault(path, threading.Lock())

    def _start(self, app_file: str, name: str, base_path: str, otel: bool) -> ServiceInstance:
        """Start the service and wait for it to report healthy, with self._service_lock() held."""
        instance = self.get(name, base_path)
        if instance is not None and self._health_ok(instance.record):
            return instance

        # fail here, in the caller's process, on everything that can be detected without serving: an app file
        # that does not declare the service is a request error, not a process that dies in the background
        services = load_service_routers(app_file)
        if name not in services:
            declared = ', '.join(sorted(services))
            raise excs.NotFoundError(
                excs.ErrorCode.SERVICE_NOT_FOUND,
                f'{app_file} contains no FastAPIRouter named {name!r}; it declares: {declared}',
            )

        log_path = self._log_path(name, base_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            '-m',
            'pixeltable.serving.service_runner',
            '--app-file',
            app_file,
            '--name',
            name,
            '--base-path',
            base_path,
        ]
        if otel:
            argv.append('--otel')
        project_root = Config.get().project_root
        assert project_root is not None
        argv += ['--project-root', str(project_root)]

        # the service outlives this call, so it must not inherit our stdio: attached to a pipe it would block
        # the reader on EOF, and attached to a terminal it would write into that session. Its own session keeps
        # signals sent to the launching process away from it.
        with open(log_path, 'a', encoding='utf-8') as log_file:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )

        deadline = time.monotonic() + self._STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            instance = self.get(name, base_path)
            if instance is not None and self._health_ok(instance.record):
                return instance
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        msg = f'Service {name!r} failed to start within {self._STARTUP_TIMEOUT:.0f}s'
        returncode = proc.poll()
        if returncode is None:
            msg += '; its process is still running but never reported healthy'
            # take the process down and drop the record it wrote, so that nothing is left claiming this target
            self._terminate(proc)
            unhealthy = self._parse_record(self._record_file(name, base_path))
            if unhealthy is not None and unhealthy.pid == proc.pid:
                self.remove(unhealthy)
        else:
            msg += f'; its process exited with code {returncode}'
        tail = self._tail_log(log_path)
        if tail != '':
            msg += f'\n--- service log tail ---\n{tail}'
        raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, msg)

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Stop proc and reap it, escalating to kill if it does not exit."""
        proc.terminate()
        try:
            proc.wait(timeout=self._STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=self._STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _logger.warning('Service process %d did not exit', proc.pid)

    def _tail_log(self, path: Path) -> str:
        """The last lines of the log at path, or the empty string if it cannot be read."""
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return ''
        return '\n'.join(lines[-self._LOG_TAIL_LINES :])

    def _read(self, record_file_path: Path) -> ServiceInstance | None:
        """The instance recorded in record_file_path, or None if that record cannot be read or its process is gone."""
        recorded = self._parse_record(record_file_path)
        if recorded is None or not self._is_live(recorded):
            return None
        return ServiceInstance(recorded, self)

    def _parse_record(self, record_file_path: Path) -> ServiceInstanceRecord | None:
        """The record in record_file_path, or None if this version cannot manage what it holds."""
        try:
            record = json.loads(record_file_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or not is_pid(record.get('pid')):
            return None
        try:
            return ServiceInstanceRecord.model_validate(record)
        except pydantic.ValidationError:
            # a record missing a field, or holding a value this version does not know, belongs to whichever
            # version wrote it
            return None

    def _is_live(self, record: ServiceInstanceRecord) -> bool:
        """Whether the process record was written for is still running.

        A pid alone does not identify it: the OS reissues the pid of an exited process, so a record left
        behind by a crash would otherwise report whichever process inherited it.
        """
        assert record.pid is not None
        if not pid_alive(record.pid):
            return False
        if record.process_started_at is None:
            return True  # nothing to compare against: the platform reported no creation time when writing
        return process_timestamp(record.pid) == record.process_started_at

    def _health_ok(self, record: ServiceInstanceRecord) -> bool:
        try:
            # every FastAPI application serves its schema
            return httpx.get(f'{record.endpoint}/openapi.json', timeout=self._HEALTH_TIMEOUT).status_code == 200
        except httpx.HTTPError:
            return False
