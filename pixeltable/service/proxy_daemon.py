"""Proxy daemon: a Pixeltable process that serves the proxy protocol over HTTP, running catalog
operations against its own database in direct mode.

This module is both the daemon entrypoint (``python -m pixeltable.service.proxy_daemon``) and the
lifecycle and discovery API for these daemons: creating, starting, stopping, and deleting a daemon for
a given database, and locating a running one via the ``port.lock`` file in its home directory.
"""

# This module intentionally omits `from __future__ import annotations`. FastAPI resolves a route
# handler's parameter annotations against the handler's *module* globals; our handlers are defined inside
# _build_app() and reference fastapi types imported locally there. Under PEP 563 those annotations would
# be strings unresolvable from module scope, and FastAPI would mis-parse the request body. Keeping
# annotations as real objects (evaluated at def time) lets FastAPI see the actual types.

import atexit
import json
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
from pixeltable.metadata.schema import base_metadata
from pixeltable.runtime import get_runtime, reset_runtime

from . import proxy_dispatch

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


def _pid_alive(pid: int) -> bool:
    """True if pid is a live process. An already-exited but unreaped child (zombie) counts as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
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


def create(db: str) -> None:
    """Create the daemon's home directory. The database itself is created on first start()."""
    proxy_home(db).mkdir(parents=True, exist_ok=True)


def start(db: str) -> str:
    """Ensure a daemon for db is running and ready; return its endpoint."""
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
    with open(log_dir / 'daemon.log', 'a', encoding='utf-8') as log_file:
        subprocess.Popen(
            [sys.executable, '-m', 'pixeltable.service.proxy_daemon'],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        ep = endpoint(db)
        if ep is not None and _health_ok(ep):
            return ep
        time.sleep(0.25)
    raise excs.Error(
        excs.ErrorCode.INTERNAL_ERROR, f'Local proxy daemon for {db!r} failed to start within {_STARTUP_TIMEOUT:.0f}s'
    )


def stop(db: str) -> None:
    """Stop the daemon for db (if running) and remove its port.lock."""
    info = read_port_lock(db)
    if info is not None:
        pid = info['pid']
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid = None
        if pid is not None:
            deadline = time.monotonic() + _STOP_TIMEOUT
            while time.monotonic() < deadline and _pid_alive(pid):
                # Reap promptly if the daemon is our own child, so its zombie isn't mistaken for a live
                # process and we don't wait out the full timeout; a no-op when it was launched by a
                # different process (e.g. a prior `pxt localproxy start`), in which case init reaps it.
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass
                time.sleep(0.05)
            if _pid_alive(pid):
                # Graceful shutdown overran the timeout; force termination so we never leak the daemon.
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    _port_lock(db).unlink(missing_ok=True)


def reset(db: str) -> None:
    """Reset the running daemon's catalog to empty, in place (no restart, so its endpoint stays valid)."""
    ep = endpoint(db)
    if ep is None:
        raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'No running proxy daemon for {db!r}')
    response = httpx.post(f'{ep}/reset', timeout=60.0)
    response.raise_for_status()


def delete(db: str) -> None:
    """Stop the daemon, drop its database, and remove its home directory."""
    stop(db)
    _drop_database(db)
    home = proxy_home(db)
    if home.exists():
        shutil.rmtree(home)


def reset_catalog() -> None:
    """Empty this daemon's catalog in place and reload it. Runs inside the daemon process.

    Drops the data tables and truncates the metadata tables, then reinitializes; the result is the same
    empty-but-initialized state as a freshly created database (init recreates the root directory record),
    so the daemon can be reused across tests without a restart. The truncate/drop logic mirrors the test
    harness's clean_db(); a shared home for it is a later cleanup.
    """
    engine = Env.get().engine
    inspector = sql.inspect(engine)
    all_table_names = set(inspector.get_table_names())
    md_table_names = set(base_metadata.tables.keys())
    data_table_names = all_table_names - md_table_names
    existing_md_names = all_table_names & md_table_names
    with engine.connect() as conn:
        if data_table_names:
            names = ', '.join(f'"{t}"' for t in data_table_names)
            conn.execute(sql.text(f'DROP TABLE IF EXISTS {names} CASCADE'))
        if existing_md_names:
            names = ', '.join(f'"{t}"' for t in existing_md_names)
            conn.execute(sql.text(f'TRUNCATE TABLE {names} CASCADE'))
        conn.commit()
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


def _build_app() -> 'FastAPI':
    """The app served by the daemon: a /rpc endpoint running the generic dispatch and a /health endpoint.

    fastapi is imported here rather than at module level because it is an optional dependency, needed
    only when the daemon is actually served.
    """
    from fastapi import FastAPI, Request, Response
    from fastapi.concurrency import run_in_threadpool

    app = FastAPI()

    @app.post('/rpc')
    async def rpc(request: Request) -> Response:
        body = (await request.body()).decode()
        # dispatch is synchronous and touches the database; keep it off the event loop
        response_json = await run_in_threadpool(proxy_dispatch.handle, body)
        return Response(content=response_json, media_type='application/json')

    @app.post('/reset')
    async def reset_endpoint() -> Response:
        # reset_catalog() truncates tables and reinitializes; keep it off the event loop
        await run_in_threadpool(reset_catalog)
        return Response(content='{"status": "ok"}', media_type='application/json')

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    return app


def _serve() -> None:
    """Daemon entrypoint. Env (PIXELTABLE_HOME/PGDATA/DB) is set by the launching start()."""
    import uvicorn

    app = _build_app()

    # eagerly create/migrate this daemon's database before announcing readiness
    _ = get_runtime().catalog

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]

    lock = Config.get().home / _LOCK_NAME  # this process's home is proxy_<db>
    lock.write_text(json.dumps({'port': port, 'pid': os.getpid()}))

    def _cleanup(*_: Any) -> None:
        lock.unlink(missing_ok=True)
        sys.exit(0)

    atexit.register(lambda: lock.unlink(missing_ok=True))
    signal.signal(signal.SIGTERM, _cleanup)

    uvicorn.Server(uvicorn.Config(app, log_level='warning')).run(sockets=[sock])


if __name__ == '__main__':
    _serve()
