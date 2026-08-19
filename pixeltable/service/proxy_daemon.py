"""Proxy daemon: a Pixeltable process that serves the proxy protocol over HTTP, running catalog
operations against its own database in direct mode.

This module is both the daemon entrypoint (python -m pixeltable.service.proxy_daemon) and the
lifecycle and discovery API for these daemons: creating, starting, stopping, and deleting a daemon for
a given database, and locating a running one via the port.lock file in its home directory.
"""

# This module intentionally omits from __future__ import annotations. FastAPI resolves a route
# handler's parameter annotations against the handler's *module* globals; our handlers are defined inside
# _build_app() and reference fastapi types imported locally there. Under PEP 563 those annotations would
# be strings unresolvable from module scope, and FastAPI would mis-parse the request body. Keeping
# annotations as real objects (evaluated at def time) lets FastAPI see the actual types.

import atexit
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.runtime import get_runtime, reset_runtime

from . import proxy_dispatch
from .proxy_protocol import decode_body, encode_body

if TYPE_CHECKING:
    from fastapi import FastAPI

_LOCK_NAME = 'port.lock'
_STARTUP_TIMEOUT = 60.0  # generous: the daemon creates/migrates its db on first init
_STOP_TIMEOUT = 10.0


def proxy_home(db: str) -> Path:
    """The daemon's home directory (its own media/tmp + port.lock), under the global home."""
    return Config.get().home / f'proxy_{db}'


def _port_lock(db: str) -> Path:
    return proxy_home(db) / _LOCK_NAME


def _win_pid_alive(pid: int) -> bool:
    """Windows liveness check via the Win32 API.

    os.kill(pid, 0) cannot probe liveness on Windows: CPython maps os.kill() to
    OpenProcess(PROCESS_ALL_ACCESS) + TerminateProcess(handle, sig), so signal 0 would terminate a live
    process, and OpenProcess raises Access-denied (WinError 5) for an already-exited process, which would
    read as alive. Instead, open the process with only SYNCHRONIZE rights and check whether its handle is
    signaled: a running process's handle is unsignaled (WAIT_TIMEOUT); an exited one is signaled.
    """
    # ctypes.wintypes exists only on Windows, so import it inside this platform-guarded path.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]  # Windows-only
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    synchronize = 0x00100000  # SYNCHRONIZE access right, the minimum needed to wait on the process handle
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False  # no such process
    try:
        # WaitForSingleObject returns WAIT_TIMEOUT (0x102) while the process runs; it returns WAIT_OBJECT_0
        # (0) once the process has exited and its handle becomes signaled.
        return kernel32.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    """True if pid is a live process. An already-exited but unreaped child (zombie) counts as dead."""
    if sys.platform == 'win32':
        return _win_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except (OSError, SystemError):
        return False
    # os.kill(pid, 0) also succeeds for a zombie (exited but not yet reaped). A zombie has terminated, so
    # treat it as dead; otherwise a daemon that we launched and that has already exited reads as running.
    try:
        with open(f'/proc/{pid}/stat', encoding='ascii') as f:
            # the state is the field after the parenthesized comm, which may itself contain spaces/parens
            state = f.read().rsplit(') ', 1)[1].split()[0]
    except (OSError, IndexError):
        return True  # no /proc (non-Linux) or a transient read race: trust the os.kill result
    return state != 'Z'


def read_port_lock(db: str) -> dict[str, Any] | None:
    """Return {'port', 'pid'} for a live daemon, or None if the lock is absent or stale."""
    lock = _port_lock(db)
    if not lock.exists():
        return None
    try:
        info = json.loads(lock.read_text())
    except (ValueError, OSError):
        return None
    return info if _pid_alive(info.get('pid', -1)) else None


def endpoint(db: str) -> str | None:
    info = read_port_lock(db)
    return None if info is None else f'http://127.0.0.1:{info["port"]}'


def _health_ok(ep: str) -> bool:
    try:
        return httpx.get(f'{ep}/health', timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


_LOG_TAIL_BYTES = 64 * 1024  # bound memory on a large log while leaving headroom for _LOG_TAIL_LINES
_LOG_TAIL_LINES = 40


def _tail_log(path: Path, n_lines: int = _LOG_TAIL_LINES) -> str:
    """Return the last n_lines of the log at path, or '' if it is absent or unreadable."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return ''
    lines = data.decode('utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-n_lines:])


def create(db: str) -> None:
    """Create the daemon's home directory. The database itself is created on first start()."""
    proxy_home(db).mkdir(parents=True, exist_ok=True)


def start(db: str, test_mode: bool = False) -> str:
    """Ensure a daemon for db is running and ready; return its endpoint.

    test_mode starts the daemon with --test (enables test-only endpoints and diagnostic detail on the errors it
    returns). It only takes effect if this call actually launches the daemon; an already-running one is returned
    as is.
    """
    create(db)
    ep = endpoint(db)
    if ep is not None and _health_ok(ep):
        return ep

    parent_home = Config.get().home
    pgdata = os.environ.get('PIXELTABLE_PGDATA') or str(parent_home / 'pgdata')
    env = {
        **os.environ,
        'PIXELTABLE_HOME': str(proxy_home(db)),  # own media/tmp + port.lock
        'PIXELTABLE_PGDATA': pgdata,  # shared postmaster
        'PIXELTABLE_DB': db,  # own database
    }
    # The daemon outlives this call, so it must not inherit our stdio: leaving the child's stdout/stderr
    # attached to a pipe blocks the reader on EOF forever, and attached to a terminal it would spew daemon
    # output into that session. Redirect to a log file and detach into its own session so signals sent to
    # the launching process don't reach the daemon.
    log_dir = proxy_home(db) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'daemon.log'
    argv = [sys.executable, '-m', 'pixeltable.service.proxy_daemon']
    if test_mode:
        argv.append('--test')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        proc = subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        ep = endpoint(db)
        if ep is not None and _health_ok(ep):
            return ep
        time.sleep(0.25)

    # The daemon never became healthy. Surface whether the process crashed (exit code) or is hung (still
    # running), plus the tail of its log, so the failure is diagnosable without access to the daemon's home.
    msg = f'Local proxy daemon for {db!r} failed to start within {_STARTUP_TIMEOUT:.0f}s'
    returncode = proc.poll()
    if returncode is None:
        msg += '; the daemon process is still running but never reported healthy'
    else:
        msg += f'; the daemon process exited with code {returncode}'
    tail = _tail_log(log_path)
    if tail != '':
        msg += f'\n--- daemon log tail ({log_path}) ---\n{tail}'
    raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, msg)


def stop(db: str) -> None:
    """Stop the daemon for db (if running) and remove its port.lock."""
    info = read_port_lock(db)
    if info is not None:
        pid = info['pid']
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # POSIX raises ProcessLookupError once the process is gone. Windows os.kill() maps to
            # TerminateProcess via OpenProcess(PROCESS_ALL_ACCESS) and raises PermissionError (WinError 5) or
            # OSError (WinError 87) for an already-exited process. In every case there is nothing left to wait on.
            pid = None
        if pid is not None:
            deadline = time.monotonic() + _STOP_TIMEOUT
            while time.monotonic() < deadline and _pid_alive(pid):
                # Reap promptly if the daemon is our own child, so its zombie isn't mistaken for a live
                # process and we don't wait out the full timeout; a no-op when it was launched by a
                # different process (e.g. a prior `pxt localproxy start`), in which case init reaps it.
                # os.WNOHANG is POSIX-only (Windows defines os.waitpid but not WNOHANG); Windows has no
                # zombies, so there is nothing to reap.
                if hasattr(os, 'WNOHANG'):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except (ChildProcessError, OSError):
                        pass
                time.sleep(0.05)
            if _pid_alive(pid):
                # Graceful shutdown overran the timeout; force termination so we never leak the daemon.
                # SIGKILL is absent on Windows, where os.kill() already terminates unconditionally.
                try:
                    os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                except (ProcessLookupError, PermissionError, OSError):
                    # The process exited between the liveness check and here; on Windows os.kill() raises
                    # PermissionError/OSError rather than ProcessLookupError once the target is gone.
                    pass
    _port_lock(db).unlink(missing_ok=True)


def reinitialize(db: str) -> None:
    """Test only. Drop the running daemon's cached catalog state, in place (no restart, so its endpoint stays valid).

    Does not clear the store state.
    """
    ep = endpoint(db)
    if ep is None:
        raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'No running proxy daemon for {db!r}')
    response = httpx.post(f'{ep}/reinitialize', timeout=60.0)
    response.raise_for_status()


def delete(db: str) -> None:
    """Stop the daemon, drop its database, and remove its home directory."""
    stop(db)
    _drop_database(db)
    home = proxy_home(db)
    if home.exists():
        shutil.rmtree(home)


def _reinitialize() -> None:
    """Test only. Discard this daemon's cached catalog state and reload it from the store. Runs inside the daemon
    process.
    """
    reset_runtime()
    pxt.init()


def _drop_database(db: str) -> None:
    env = Env.get()
    if env._db_server is None:
        return  # not running against the embedded postmaster (e.g. external DB); nothing to drop
    engine = sql.create_engine(env._dbms.default_system_db_url(), future=True, isolation_level='AUTOCOMMIT')
    try:
        with engine.begin() as conn:
            conn.execute(
                sql.text(
                    'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
                    'WHERE datname = :db AND pid <> pg_backend_pid()'
                ),
                {'db': db},
            )
            conn.execute(sql.text(f'DROP DATABASE IF EXISTS "{db}"'))
    finally:
        engine.dispose()


def _build_app(test_mode: bool = False) -> 'FastAPI':
    """The app served by the daemon: a /rpc endpoint running the generic dispatch and a /health endpoint.

    test_mode enables the test-only endpoints, and lets /rpc return the diagnostic detail of the errors it reports.

    fastapi is imported here rather than at module level because it is an optional dependency, needed
    only when the daemon is actually served.
    """
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse

    app = FastAPI()

    @app.post('/rpc')
    async def rpc(request: Request) -> Response:
        request_json, request_parts = decode_body(await request.body())
        # dispatch is synchronous and touches the database; keep it off the event loop
        response_json, response_parts = await run_in_threadpool(
            proxy_dispatch.handle, request_json.decode(), request_parts, include_error_detail=test_mode
        )
        return Response(
            content=encode_body(response_json.encode(), response_parts), media_type='application/octet-stream'
        )

    if test_mode:

        @app.post('/reinitialize')
        async def reinitialize_endpoint() -> Response:
            # _reinitialize() rebuilds the runtime, which is synchronous; keep it off the event loop
            await run_in_threadpool(_reinitialize)
            return Response(content='{"status": "ok"}', media_type='application/json')

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/media/{ref:path}')
    def serve_media(ref: str) -> FileResponse:
        # serve a persisted media file by its media-dir-relative ref
        media_dir = Env.get().media_dir.resolve()
        resolved = (media_dir / ref).resolve()
        if resolved != media_dir and media_dir not in resolved.parents:
            raise HTTPException(status_code=404, detail='not found')
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail='not found')
        return FileResponse(resolved)

    return app


def _serve(test_mode: bool = False) -> None:
    """Daemon entrypoint.

    Local mode (default): binds to a random loopback port, writes a port.lock file
    for the SDK to discover, and manages the database lifecycle itself.

    Fixed-address mode: when PIXELTABLE_DAEMON_HOST or PIXELTABLE_DAEMON_PORT is set,
    binds to that address and port instead and skips the lock file. Used when an
    external orchestrator (e.g. a sidecar) handles routing and discovery.

    test_mode: exposes the test-only endpoints and returns diagnostic detail with errors.
    """
    # mark this process as a hosted-catalog server (no client-accessible local store) before the catalog inits
    os.environ['PIXELTABLE_PROXY_DAEMON'] = '1'
    try:
        import uvicorn
    except ModuleNotFoundError as e:
        raise excs.Error(
            excs.ErrorCode.INTERNAL_ERROR,
            'The proxy daemon requires the serve dependencies (fastapi, uvicorn). '
            'Install them with: pip install pixeltable[serve]',
        ) from e

    app = _build_app(test_mode)

    # eagerly create/migrate this daemon's database before announcing readiness
    _ = get_runtime().catalog

    config = Config.get()
    daemon_host = config.get_string_value('daemon_host')
    daemon_port = config.get_int_value('daemon_port')

    # pixeltable log level also drives uvicorn
    log_level = logging.getLogger('pixeltable').getEffectiveLevel()

    # log_config=None suppresses uvicorn's own logging setup which results in closing every handler registered so far.
    # Note: at this point, uvicorn logging has already been configured by Env.
    if daemon_host is not None or daemon_port is not None:
        uvicorn.run(
            app, host=daemon_host or '127.0.0.1', port=daemon_port or 8000, log_level=log_level, log_config=None
        )
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]

    lock = config.home / _LOCK_NAME  # this process's home is proxy_<db>
    lock.write_text(json.dumps({'port': port, 'pid': os.getpid()}))

    def _cleanup(*_: Any) -> None:
        lock.unlink(missing_ok=True)
        sys.exit(0)

    atexit.register(lambda: lock.unlink(missing_ok=True))
    signal.signal(signal.SIGTERM, _cleanup)

    uvicorn.Server(uvicorn.Config(app, log_level=log_level, log_config=None)).run(sockets=[sock])


if __name__ == '__main__':
    test_mode = '--test' in sys.argv[1:]
    _serve(test_mode=test_mode)
