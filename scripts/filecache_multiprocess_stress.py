#!/usr/bin/env python3
"""Multiprocess stress test for the Pixeltable file cache.

Several OS processes share one Pixeltable home (hence one file_cache_dir), sized small enough to force
eviction, and concurrently insert rows referencing remote media and run queries that fetch and decode that
media. If the cache is multi-process safe, every worker completes and every
queried image decodes; a lost in-use file, an eviction deleting another process's file, or index/disk drift
would surface as a worker exception (nonzero exit).

Media is served from a local HTTP server so the run is self-contained and offline. An isolated temporary
home (with its own embedded Postgres) is created and removed per run, so this never touches a real instance.

    python scripts/filecache_multiprocess_stress.py [--procs 4] [--images 100] [--rows 25] [--rounds 20]

Exits 0 if every worker succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import PIL.Image

import pixeltable as pxt


def _generate_images(media_dir: Path, count: int, px: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(count):
        arr = rng.integers(0, 256, size=(px, px, 3), dtype=np.uint8)
        PIL.Image.fromarray(arr, mode='RGB').save(media_dir / f'img_{i}.png')


def _thread_body(
    tbl: pxt.Table, worker_id: int, thread_id: int, port: int, n_images: int, rows: int, rounds: int
) -> int:
    """One thread: repeatedly insert a batch of rows and query just that batch, forcing fetch+decode.

    Each round advances through the image pool (img index = base+row_idx mod n_images), and all threads and
    processes use the same index per round, so they concurrently add the same urls while urls from earlier
    rounds age past the lease and get evicted. Only the current batch is queried, so old urls are left idle
    long enough to age out rather than being kept warm.

    A round that hits the acceptable out-of-capacity error is counted and skipped so the thread keeps churning
    for the full round count; returns the number of such rounds.
    """
    capacity_hits = 0
    for round_idx in range(rounds):
        base = round_idx * rows
        label = f'worker {worker_id}/{thread_id} round {round_idx}'
        try:
            tbl.insert(
                {'idx': base + row_idx, 'img': f'http://127.0.0.1:{port}/img_{(base + row_idx) % n_images}.png'}
                for row_idx in range(rows)
            )
            # rotate() forces each image to be fetched from the cache and decoded; a lost or corrupt file raises.
            result = tbl.where(tbl.idx >= base).select(rotated=tbl.img.rotate(90)).collect()
            assert len(result) == rows, f'{label}: expected {rows} rows, got {len(result)}'
            assert all(row['rotated'] is not None for row in result), f'{label}: missing media'
        except Exception as exc:
            if not _is_capacity_error(exc):
                raise
            capacity_hits += 1
    return capacity_hits


def _is_capacity_error(exc: Exception) -> bool:
    """True if exc (or anything in its chain) is the cache's 'out of capacity' error.

    This is the acceptable outcome: it means every cached file is leased (in concurrent use), which is exactly
    the contention state the test aims to reach, not a correctness failure.
    """
    seen: list[Exception] = []
    cur: Exception | None = exc
    while cur is not None and cur not in seen:
        seen.append(cur)
        if isinstance(cur, pxt.Error) and cur.error_code == pxt.ErrorCode.FILE_CACHE_FULL:
            return True
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if isinstance(nxt, Exception) else None
    return False


def _worker(worker_id: int, port: int, n_images: int, rows: int, rounds: int, n_threads: int) -> None:
    """One process: runs n_threads threads, each driving its own table through the insert/query fetch path.

    Exit code: 1 if any thread hit a non-capacity error (a real failure); 2 if the only errors were the
    acceptable 'out of capacity' contention signal; 0 if every thread completed cleanly.
    """
    fatal: list[Exception] = []
    capacity_hits = [0]
    try:
        tbls = [
            pxt.create_table(f'stress_{worker_id}_{thread_id}', {'idx': pxt.Int, 'img': pxt.Image}, if_exists='replace')
            for thread_id in range(n_threads)
        ]

        def run(thread_id: int) -> None:
            try:
                capacity_hits[0] += _thread_body(tbls[thread_id], worker_id, thread_id, port, n_images, rows, rounds)
            except Exception as exc:
                traceback.print_exc()
                fatal.append(exc)

        threads = [threading.Thread(target=run, args=(thread_id,)) for thread_id in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except Exception:
        # setup failure (setting the lease or creating tables): a real failure, not the acceptable capacity signal
        traceback.print_exc()
        sys.exit(1)

    # exit decision is outside the try so these SystemExits are not caught and downgraded above
    if len(fatal) > 0:
        sys.exit(1)
    if capacity_hits[0] > 0:
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--procs', type=int, default=4, help='number of concurrent worker processes')
    parser.add_argument('--threads', type=int, default=4, help='threads per worker process')
    parser.add_argument('--images', type=int, default=100, help='size of the shared image pool')
    parser.add_argument('--rows', type=int, default=25, help='rows inserted per round per thread')
    parser.add_argument('--rounds', type=int, default=20, help='insert/query rounds per thread')
    parser.add_argument('--cache-gb', type=float, default=0.0007, help='file cache size (GiB); small forces eviction')
    parser.add_argument('--lease-s', type=float, default=2.0, help='file cache eviction lease (s)')
    parser.add_argument('--img-px', type=int, default=128, help='generated image edge length in pixels')
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix='pxt_fc_stress_'))
    home = workdir / 'home'
    media = workdir / 'media'
    home.mkdir(parents=True)
    media.mkdir(parents=True)
    _generate_images(media, args.images, args.img_px)

    config_file = home / 'config.toml'
    config_file.write_text(
        f'[pixeltable]\n'
        f'file_cache_size_g = {args.cache_gb}\n'
        f'file_cache_lease_s = {args.lease_s}\n'
        f'hide_warnings = true\n'
    )

    # Point every process (this one and the spawned workers, which inherit the environment) at the isolated home.
    os.environ['PIXELTABLE_HOME'] = str(home)
    os.environ['PIXELTABLE_CONFIG'] = str(config_file)
    os.environ['PIXELTABLE_PGDATA'] = str(workdir / 'pgdata')
    os.environ['PIXELTABLE_DB'] = 'stress'

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass  # suppress per-request logging, which is heavy under the high request volume of this test

    handler = functools.partial(QuietHandler, directory=str(media))
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    try:
        # bring up the embedded Postgres before the workers connect to it
        pxt.init()

        ctx = mp.get_context('spawn')  # fresh interpreters; a fork would inherit live DB connections
        processes = [
            ctx.Process(
                target=_worker,
                args=(worker_id, port, args.images, args.rows, args.rounds, args.threads),
            )
            for worker_id in range(args.procs)
        ]
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
        exit_codes = [proc.exitcode for proc in processes]

        cache_files = [f for f in (home / 'file_cache').glob('*') if f.suffix != '.lock']
        cache_bytes = sum(f.stat().st_size for f in cache_files)
        capacity_kib = args.cache_gb * (1 << 30) / 1024
        print(f'worker exit codes: {exit_codes}')
        print(f'file cache after run: {len(cache_files)} files, {cache_bytes / 1024:.0f} KiB (capacity ~{capacity_kib:.0f} KiB)')

        # exit 1 == a real (non-capacity) failure in some thread; exit 2 == only the acceptable out-of-capacity
        # contention signal was hit; exit 0 == every thread completed cleanly (no contention reached)
        fatal = any(code == 1 or code is None for code in exit_codes)
        contention_reached = any(code == 2 for code in exit_codes)
        print(f'contention (FILE_CACHE_FULL) reached: {contention_reached}')
        if not contention_reached:
            print('WARNING: no contention reached; increase load or lower --cache-gb to exercise the race')
        print('PASS' if not fatal else 'FAIL')
        return 0 if not fatal else 1
    finally:
        httpd.shutdown()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
