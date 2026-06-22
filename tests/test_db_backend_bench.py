"""Concurrent-client serving benchmark comparing Pixeltable database backends.

For each parameterized backend the test launches a `pxt serve`-style uvicorn instance (the app is
built exactly the way `pixeltable.serving._config.create_service_from_config` builds it) pointed at
that backend, drives it with several concurrent-client workloads that stress the DB layer along
different performance axes, and reports two metrics per backend:

  1. total wall-clock run time for the whole suite, and
  2. the percentage of backend responses that succeeded.

Backends (parameterized):
  - local-1cpu:  a Postgres + pgvector container pinned to one CPU (`docker run --cpus=1`).
  - planetscale: an external Postgres URL taken from PXT_BENCH_PLANETSCALE_URL (skipped if unset).

Workloads (each maps to a real Pixeltable operation and a distinct performance requirement):
  - bulk_insert             throughput / round-trip       (single-row inserts, one table per client)
  - indexed_insert          write + index maintenance     (insert -> HNSW upsert, one table per client)
  - similarity_search       read latency                  (pgvector ORDER BY ... LIMIT, shared table)
  - single_table_contention per-table write-lock failure  (all clients write one shared table)

The write workloads give each concurrent client its own table because Pixeltable takes a
FOR UPDATE NOWAIT lock on the table's catalog row for every write (catalog.py::_acquire_write_lock),
so concurrent writes to the *same* table fail fast rather than queue. single_table_contention
deliberately points all clients at one table to measure that fail-fast rate as its own axis.

ponytail: workloads are text + embedding only (no image/video) so the *database* round-trips
dominate rather than model inference or media decode -- the embedding is the download-free
`tests.utils.local_embedding`. Adding an image/video table + frame_iterator view is a drop-in
extension if a heavier multimodal mix is wanted later.

Run:  pytest tests/test_db_backend_bench.py -s -v
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .utils import skip_test_if_not_in_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_N_CLIENTS = 8
_N_SMALL = 200  # bulk_insert rows
_N_DOC = 100  # indexed_insert rows
_N_SEARCH = 100  # similarity_search requests
_N_CONTENTION = 100  # single_table_contention requests (all clients, one shared table)


# ---- server process (run as `python -m tests.test_db_backend_bench --port N`) ----


def _build_app(title: str = 'pxt-db-bench'):  # type: ignore[no-untyped-def]
    """Build the same FastAPI app `pxt serve` would, with the benchmark schema and routes."""
    import fastapi

    import pixeltable as pxt
    from pixeltable.serving import FastAPIRouter

    from .utils import local_embedding

    embed = local_embedding.using(dim=512)

    # Clean slate: drop tables left by a prior run (or an older schema version) so the benchmark
    # schema is deterministic on a persistent backend like PlanetScale.
    pxt.drop_dir('bench', force=True, if_not_exists='ignore')
    pxt.create_dir('bench')
    app = fastapi.FastAPI(title=title)
    router = FastAPIRouter()

    # One write table per client so concurrent inserts don't contend on the same catalog row
    # (Pixeltable takes a FOR UPDATE NOWAIT lock per table on every write, see catalog.py).
    for i in range(_N_CLIENTS):
        small_i = pxt.create_table(f'bench.small_{i}', {'k': pxt.String, 'v': pxt.Int}, if_exists='replace_force')
        docs_i = pxt.create_table(f'bench.docs_{i}', {'doc_id': pxt.Int, 'text': pxt.String}, if_exists='replace_force')
        # embedding index directly on the text column: every insert triggers one HNSW upsert
        docs_i.add_embedding_index('text', embedding=embed, if_exists='replace_force')
        router.add_insert_route(small_i, path=f'/ingest_small_{i}', inputs=['k', 'v'])
        router.add_insert_route(docs_i, path=f'/ingest_doc_{i}', inputs=['doc_id', 'text'])

    # Shared table that all clients write to, to measure the single-table NOWAIT lock-failure axis.
    contention = pxt.create_table(
        'bench.contention', {'doc_id': pxt.Int, 'text': pxt.String}, if_exists='replace_force'
    )
    router.add_insert_route(contention, path='/ingest_contention', inputs=['doc_id', 'text'])

    # Reads don't take the write lock, so similarity_search runs concurrently against one shared table
    # (docs_0, populated by the indexed_insert phase).
    read_tbl = pxt.get_table('bench.docs_0')

    @pxt.query
    def search(q: str):  # type: ignore[no-untyped-def]
        sim = read_tbl.text.similarity(string=q)
        return read_tbl.order_by(sim, asc=False).limit(5).select(read_tbl.text, sim)

    router.add_query_route(path='/search', query=search, one_row=False)
    app.include_router(router)
    return app


def _serve_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--cleanup', action='store_true', help='drop the benchmark schema and exit')
    args = parser.parse_args()

    if args.cleanup:
        import pixeltable as pxt

        pxt.drop_dir('bench', force=True, if_not_exists='ignore')
        return

    import uvicorn

    assert args.port is not None, '--port is required to serve'
    uvicorn.run(_build_app(), host=args.host, port=args.port, log_level='warning')


# ---- client side (stdlib only, no httpx dependency) ----


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> tuple[bool, str | None]:
    """POST a JSON body. Returns (ok, error) where ok is True iff the backend answered 2xx."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300, None
    except urllib.error.HTTPError as e:
        return False, f'HTTP {e.code}: {e.read(300).decode("utf-8", "replace")}'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def _run_concurrent(base_url: str, requests: list[tuple[str, dict]], n_clients: int) -> tuple[int, int, list[str]]:
    """Fan `requests` (path, payload) across `n_clients` threads. Returns (successes, total, error_samples)."""
    buckets = [requests[i::n_clients] for i in range(n_clients)]

    def worker(items: list[tuple[str, dict]]) -> tuple[int, list[str]]:
        ok = 0
        errs: list[str] = []
        for path, payload in items:
            good, err = _post_json(base_url + path, payload)
            if good:
                ok += 1
            elif err is not None and len(errs) < 3:
                errs.append(err)
        return ok, errs

    successes = 0
    error_samples: list[str] = []
    with ThreadPoolExecutor(max_workers=n_clients) as ex:
        for ok, errs in ex.map(worker, buckets):
            successes += ok
            for e in errs:
                if e not in error_samples and len(error_samples) < 5:
                    error_samples.append(e)
    return successes, len(requests), error_samples


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_until_ready(base_url: str, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'serve process exited early (code {proc.returncode})')
        try:
            with urllib.request.urlopen(base_url + '/openapi.json', timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f'serve instance at {base_url} did not become ready in {timeout}s')


def _drop_backend_schema(db_url: str) -> None:
    """Post-test cleanup: drop the benchmark schema from `db_url` (matters for persistent backends)."""
    env = {**os.environ, 'PIXELTABLE_DB_CONNECT_STR': db_url}
    subprocess.run(
        [sys.executable, '-m', 'tests.test_db_backend_bench', '--cleanup'], cwd=str(_REPO_ROOT), env=env, check=False
    )


@contextlib.contextmanager
def _served_instance(db_url: str) -> Iterator[str]:
    """Launch a uvicorn serve subprocess bound to `db_url`; yield its base URL."""
    port = _free_port()
    env = {**os.environ, 'PIXELTABLE_DB_CONNECT_STR': db_url}
    proc = subprocess.Popen(
        [sys.executable, '-m', 'tests.test_db_backend_bench', '--port', str(port)], cwd=str(_REPO_ROOT), env=env
    )
    base_url = f'http://127.0.0.1:{port}'
    try:
        _wait_until_ready(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---- backend fixtures ----


def _wait_for_pg(container: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = subprocess.run(['docker', 'exec', container, 'pg_isready', '-U', 'postgres'], check=False)
        if probe.returncode == 0:
            return
        time.sleep(1.0)
    raise TimeoutError(f'postgres container {container} not ready in {timeout}s')


@pytest.fixture(params=['local-1cpu', 'planetscale'])
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, str]]:
    if request.param == 'local-1cpu':
        skip_test_if_not_in_path('docker')
        if subprocess.run(['docker', 'info'], check=False, capture_output=True).returncode != 0:
            pytest.skip('docker daemon is not reachable')
        port = _free_port()
        container = f'pxt-bench-pg-{port}'
        subprocess.run(
            [
                'docker',
                'run',
                '-d',
                '--rm',
                '--cpus=1',
                '--name',
                container,
                '-e',
                'POSTGRES_PASSWORD=pxt',
                '-e',
                'POSTGRES_DB=pixeltable',
                '-p',
                f'{port}:5432',
                'pgvector/pgvector:pg16',
            ],
            check=True,
        )
        try:
            _wait_for_pg(container)
            yield 'local-1cpu', f'postgresql+psycopg://postgres:pxt@127.0.0.1:{port}/pixeltable'
        finally:
            subprocess.run(['docker', 'rm', '-f', container], check=False)
    else:
        url = os.environ.get('PXT_BENCH_PLANETSCALE_URL')
        if not url:
            pytest.skip('set PXT_BENCH_PLANETSCALE_URL to benchmark the PlanetScale backend')
        try:
            yield 'planetscale', url
        finally:
            _drop_backend_schema(url)


# ---- the benchmark ----


def _require(*packages: str) -> None:
    """Skip if a package is missing -- via importlib so the test process never initializes Pixeltable."""
    for package in packages:
        if importlib.util.find_spec(package) is None:
            pytest.skip(f'package {package!r} is not installed')


def _count_bench_tables(db_url: str) -> int:
    """Count the active per-client write tables in the target backend's Pixeltable catalog.

    Proves the served instance actually wrote to `db_url` (not a silent fallback to the embedded db):
    if it had fallen back, the intended backend would have zero of these tables. Matches only the exact
    small_0..N-1 names and excludes rows marked for drop (md.pending_stmt == DROP_TABLE, value 2 per
    pixeltable.metadata.schema.TableStatement) so tombstones left by interrupted runs don't inflate it.
    """
    import sqlalchemy as sql

    names = [f'small_{i}' for i in range(_N_CLIENTS)]
    stmt = sql.text(
        "SELECT count(*) FROM tables WHERE md->>'name' IN :names AND coalesce(md->>'pending_stmt', '-1') != '2'"
    ).bindparams(sql.bindparam('names', expanding=True))
    eng = sql.create_engine(db_url, future=True)
    try:
        with eng.connect() as conn:
            return conn.execute(stmt, {'names': names}).scalar_one()
    finally:
        eng.dispose()


def _run_suite(base_url: str) -> dict:
    # Bucket i (requests[i::_N_CLIENTS]) maps to table i, so each client writes only to its own table.
    small = [(f'/ingest_small_{i % _N_CLIENTS}', {'k': f'k{i}', 'v': i}) for i in range(_N_SMALL)]
    docs = [
        (
            f'/ingest_doc_{i % _N_CLIENTS}',
            {'doc_id': i, 'text': f'sample document {i} about multimodal data infrastructure'},
        )
        for i in range(_N_DOC)
    ]
    searches = [('/search', {'q': f'document {i % 20}'}) for i in range(_N_SEARCH)]
    # all clients write the one shared table: exercises the per-table NOWAIT write lock
    contention = [
        ('/ingest_contention', {'doc_id': i, 'text': f'contention document {i}'}) for i in range(_N_CONTENTION)
    ]

    phases: list[tuple[str, list[tuple[str, dict]]]] = [
        ('bulk_insert', small),  # throughput / round-trip cost (per-client tables)
        ('indexed_insert', docs),  # write + HNSW index maintenance (per-client tables)
        ('similarity_search', searches),  # read latency (shared read table)
        ('single_table_contention', contention),  # per-table NOWAIT write-lock failure axis
    ]

    results: dict = {'workloads': {}}
    total_ok = total_n = 0
    t0 = time.monotonic()
    for name, reqs in phases:
        ok, n, errs = _run_concurrent(base_url, reqs, _N_CLIENTS)
        results['workloads'][name] = {'ok': ok, 'total': n, 'pct': 100.0 * ok / n, 'errors': errs}
        total_ok += ok
        total_n += n
    results['total_run_time_s'] = time.monotonic() - t0
    results['success_pct'] = 100.0 * total_ok / total_n
    return results


def test_db_backend_serving_bench(backend: tuple[str, str]) -> None:
    _require('fastapi', 'uvicorn')
    label, db_url = backend

    with _served_instance(db_url) as base_url:
        results = _run_suite(base_url)

    # Prove the served instance actually used `db_url` rather than silently falling back to the
    # embedded db: the benchmark write tables must exist in the intended backend's catalog.
    n_tables = _count_bench_tables(db_url)
    assert n_tables == _N_CLIENTS, f'expected {_N_CLIENTS} bench tables in {label}, found {n_tables}'

    print(f'\n=== DB backend benchmark: {label} ===')
    for name, w in results['workloads'].items():
        print(f'  {name:24s} {w["ok"]:4d}/{w["total"]:<4d} ({w["pct"]:5.1f}% ok)')
        for err in w['errors']:
            print(f'      ! {err}')
    print(f'  total_run_time     {results["total_run_time_s"]:.2f}s')
    print(f'  success_pct        {results["success_pct"]:.1f}%')

    # The success rate is itself a reported metric: a low rate (e.g. from connection limits or write
    # contention) is a meaningful backend signal, not a reason to fail the run. So we only assert that
    # the harness actually exercised the backend -- the served instance came up and handled at least
    # one request -- and leave the per-workload rates for the printed report to compare.
    assert results['success_pct'] > 0.0, 'served instance handled no requests successfully'


if __name__ == '__main__':
    _serve_main()
