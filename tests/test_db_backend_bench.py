"""Direct DB-backend insert/read benchmark (Pixeltable is not in the path).

Connects straight to each backend via SQLAlchemy and times six workloads, the single- vs multi-row
cross of {scalar table, pgvector HNSW-indexed table} for inserts plus single- vs multi-row reads
against the indexed table. Each workload is driven by a sweep of n_procs processes x n_threads
threads (true parallelism past the GIL); rows/s is aggregated bottom-up from the threads of every
process. The numbers are a floor on the backend's raw write/read cost to compare served numbers to.

Backends: local-1cpu (a one-CPU pgvector container) and planetscale (PXT_BENCH_PLANETSCALE_URL).
Run:  pytest tests/test_db_backend_bench.py -s -v
"""

from __future__ import annotations

import functools
import os
import socket
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest
import sqlalchemy as sql

from .utils import skip_test_if_not_in_path

_DIM = 512
_BATCH = 1000  # rows per multi-row op / single-row op count; LIMIT for the multi-row read
_N_READS = 100  # times each read query is repeated, so the timing is measurable
_N_PROCESSES = [1, 2, 4, 8]  # swept by the test: worker processes
_N_THREADS = [1, 8, 16]  # swept by the test: threads (each its own connection) per process

_SCALAR_INS = sql.text('INSERT INTO bench_scalar (k, v) VALUES (:k, :v)')
_VEC_INS = sql.text('INSERT INTO bench_vec (id, txt, emb) VALUES (:id, :txt, CAST(:emb AS vector))')
_VEC_READ_1 = sql.text('SELECT id FROM bench_vec ORDER BY emb <=> CAST(:q AS vector) LIMIT 1')
_VEC_READ_N = sql.text(f'SELECT id FROM bench_vec ORDER BY emb <=> CAST(:q AS vector) LIMIT {_BATCH}')


def _vec(i: int) -> str:
    """A distinct `_DIM`-d pgvector literal per `i` so HNSW search isn't degenerate."""
    return '[' + str(i % 1000) + ',0' * (_DIM - 1) + ']'


def _latency_ms(conn: sql.Connection) -> float:
    """Median round-trip for a trivial query -- the per-statement network floor for this backend."""
    samples = []
    for _ in range(25):
        t0 = time.monotonic()
        conn.execute(sql.text('SELECT 1')).scalar_one()
        samples.append(time.monotonic() - t0)
    return statistics.median(samples) * 1000


def _setup(conn: sql.Connection) -> None:
    conn.execute(sql.text('CREATE EXTENSION IF NOT EXISTS vector'))
    for ddl in (
        'DROP TABLE IF EXISTS bench_scalar',
        'DROP TABLE IF EXISTS bench_vec',
        'CREATE TABLE bench_scalar (k text, v integer)',
        f'CREATE TABLE bench_vec (id integer, txt text, emb vector({_DIM}))',
        'CREATE INDEX ON bench_vec USING hnsw (emb vector_cosine_ops)',
    ):
        conn.execute(sql.text(ddl))
    conn.commit()


# Each workload processes a shard of row indices and returns the number of rows it actually
# wrote/read, so the throughput metric is aggregated from completed work rather than assumed.
def _ins_scalar_single(conn: sql.Connection, idxs: list[int]) -> int:
    for i in idxs:
        conn.execute(_SCALAR_INS, {'k': f'k{i}', 'v': i})
        conn.commit()
    return len(idxs)


def _ins_scalar_multi(conn: sql.Connection, idxs: list[int]) -> int:
    conn.execute(_SCALAR_INS, [{'k': f'k{i}', 'v': i} for i in idxs])
    conn.commit()
    return len(idxs)


def _ins_vec_single(conn: sql.Connection, idxs: list[int]) -> int:
    for i in idxs:
        conn.execute(_VEC_INS, {'id': i, 'txt': f't{i}', 'emb': _vec(i)})
        conn.commit()
    return len(idxs)


def _ins_vec_multi(conn: sql.Connection, idxs: list[int]) -> int:
    conn.execute(_VEC_INS, [{'id': i, 'txt': f't{i}', 'emb': _vec(i)} for i in idxs])
    conn.commit()
    return len(idxs)


def _read_vec_single(conn: sql.Connection, idxs: list[int]) -> int:
    return sum(len(conn.execute(_VEC_READ_1, {'q': _vec(i)}).fetchall()) for i in idxs)


def _read_vec_multi(conn: sql.Connection, idxs: list[int]) -> int:
    return sum(len(conn.execute(_VEC_READ_N, {'q': _vec(i)}).fetchall()) for i in idxs)


_Workload = Callable[[sql.Connection, list[int]], int]


def _run_shard(eng: sql.Engine, fn: _Workload, idxs: list[int]) -> int:
    """One thread: open a connection, run the workload on its shard, return rows completed."""
    with eng.connect() as conn:
        return fn(conn, idxs)


# Per-process engine: built once in each worker process (engines can't cross the spawn boundary), so
# its connection setup stays out of the per-workload timing.
_PROC_ENGINE: sql.Engine | None = None


def _init_proc(db_url: str, n_threads: int) -> None:
    global _PROC_ENGINE
    _PROC_ENGINE = sql.create_engine(db_url, future=True, pool_size=n_threads, max_overflow=0)


def _proc_run(fn: _Workload, n_threads: int, shards: list[list[int]]) -> int:
    """One process: fan its shards across `n_threads` threads, return rows summed over them."""
    assert _PROC_ENGINE is not None
    with ThreadPoolExecutor(n_threads) as ex:
        return sum(ex.map(functools.partial(_run_shard, _PROC_ENGINE, fn), shards))


# (name, fn, n_items): fn processes a shard of range(n_items) and returns rows done.
# Run in order: every insert workload runs before the reads that depend on the populated table.
_WORKLOADS: list[tuple[str, _Workload, int]] = [
    ('insert_scalar_single', _ins_scalar_single, _BATCH),
    ('insert_scalar_multi', _ins_scalar_multi, _BATCH),
    ('insert_vec_single', _ins_vec_single, _BATCH),
    ('insert_vec_multi', _ins_vec_multi, _BATCH),
    ('read_vec_single', _read_vec_single, _N_READS),
    ('read_vec_multi', _read_vec_multi, _N_READS),
]


@pytest.fixture(params=['local-1cpu', 'planetscale'])
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, str]]:
    if request.param == 'planetscale':
        url = os.environ.get('PXT_BENCH_PLANETSCALE_URL')
        if not url:
            pytest.skip('set PXT_BENCH_PLANETSCALE_URL to benchmark the PlanetScale backend')
        yield 'planetscale', url
        return
    skip_test_if_not_in_path('docker')
    if subprocess.run(['docker', 'info'], check=False, capture_output=True).returncode != 0:
        pytest.skip('docker daemon is not reachable')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    container = f'pxt-bench-pg-{port}'
    subprocess.run(
        ['docker', 'run', '-d', '--rm', '--cpus=1', '--name', container, '-e', 'POSTGRES_PASSWORD=pxt',
         '-e', 'POSTGRES_DB=pixeltable', '-p', f'{port}:5432', 'pgvector/pgvector:pg16'], check=True
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 90
        while (
            subprocess.run(['docker', 'exec', container, 'pg_isready', '-U', 'postgres'], check=False).returncode != 0
        ):
            if time.monotonic() > deadline:
                raise TimeoutError('postgres container not ready')
            time.sleep(1.0)
        yield 'local-1cpu', f'postgresql+psycopg://postgres:pxt@127.0.0.1:{port}/pixeltable'
    finally:
        subprocess.run(['docker', 'rm', '-f', container], check=False)


@pytest.mark.parametrize('n_threads', _N_THREADS)
@pytest.mark.parametrize('n_procs', _N_PROCESSES)
def test_db_backend_direct_bench(backend: tuple[str, str], n_procs: int, n_threads: int) -> None:
    label, db_url = backend
    total = n_procs * n_threads
    eng = sql.create_engine(db_url, future=True)
    try:
        with eng.connect() as conn:
            _setup(conn)
            rtt_ms = _latency_ms(conn)
        print(f'\n=== direct DB benchmark: {label} ({n_procs}p x {n_threads}t = {total}, {rtt_ms:.1f} ms RTT) ===')
        with ProcessPoolExecutor(n_procs, initializer=_init_proc, initargs=(db_url, n_threads)) as ex:
            for name, fn, n_items in _WORKLOADS:
                # `total` shards over range(n_items); hand each process a round-robin slice of n_threads shards.
                shards = [list(range(k, n_items, total)) for k in range(total)]
                chunks = [shards[p::n_procs] for p in range(n_procs)]
                t0 = time.monotonic()
                rows = sum(ex.map(functools.partial(_proc_run, fn, n_threads), chunks))  # threads summed per process
                dt = time.monotonic() - t0
                print(f'  {name:22s} {dt:7.3f}s  {rows / dt:10.1f} rows/s  ({rows} rows)')
        with eng.connect() as conn:
            n = conn.execute(sql.text('SELECT count(*) FROM bench_vec')).scalar_one()
    finally:
        eng.dispose()
    assert n > 0, 'no rows were inserted into the indexed table'
