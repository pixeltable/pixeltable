"""Raise max_connections on the Postgres server shared by the pytest-xdist workers, ahead of the test run.

Every worker connects to one local server (`init_env()` in tests/conftest.py), and each worker can run three processes
with their own engine pools: the pytest worker, its proxy daemon and its CLI daemon. With `pytest -n logical`,
initdb's default of 100 connections runs out on a 16-vCPU runner.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pixeltable_pgserver
import psutil
import sqlalchemy as sql

from pixeltable.env import _DEFAULT_DB_POOL_MAX_OVERFLOW, _DEFAULT_DB_POOL_SIZE

_POOLS_PER_WORKER = 3
# initdb's default, kept as headroom for test subprocesses and per-test daemons
_BASE_MAX_CONNECTIONS = 100


def _pgdata() -> Path:
    """The workers' pgdata directory, by the same rules as init_env()."""
    pgdata = os.environ.get('PIXELTABLE_PGDATA', '')
    if pgdata != '':
        return Path(pgdata)
    home = os.environ.get('PIXELTABLE_HOME', '')
    return (Path(home) if home != '' else Path.home() / '.pixeltable') / 'pgdata'


def _execute(server: pixeltable_pgserver.PostgresServer, stmt: str) -> Any:
    """Runs stmt and returns its scalar result, if it has one."""
    engine = sql.create_engine(server.get_uri(driver='psycopg'), isolation_level='AUTOCOMMIT')
    try:
        with engine.connect() as conn:
            result = conn.execute(sql.text(stmt))
            return result.scalar() if result.returns_rows else None
    finally:
        engine.dispose()


def main() -> None:
    workers = psutil.cpu_count(logical=True)  # the worker count of `pytest -n logical`
    assert workers is not None
    max_connections = _BASE_MAX_CONNECTIONS + workers * _POOLS_PER_WORKER * (
        _DEFAULT_DB_POOL_SIZE + _DEFAULT_DB_POOL_MAX_OVERFLOW
    )
    pgdata = _pgdata()
    pgdata.parent.mkdir(parents=True, exist_ok=True)
    # leaving each block stops the server; max_connections takes effect at the next start
    with pixeltable_pgserver.get_server(pgdata, cleanup_mode='stop') as server:
        _execute(server, f'ALTER SYSTEM SET max_connections = {max_connections}')
    with pixeltable_pgserver.get_server(pgdata, cleanup_mode='stop') as server:
        effective = int(_execute(server, 'SHOW max_connections'))
    if effective != max_connections:
        sys.exit(f'max_connections is {effective} after a restart, expected {max_connections}')
    print(f'max_connections = {max_connections} for {workers} workers ({pgdata})')


if __name__ == '__main__':
    main()
