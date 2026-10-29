"""
Kills Pixeltable partway through a large insert, then checks that a fresh process loads it normally.

Each cycle starts an insert in a worker process and kills it without warning at a random point during the insert.
A new process then initializes Pixeltable, reads the whole table and checks that every insert either committed in
full or left no rows, and that every committed row's media files exist.

Kill modes:
- python: only the worker process dies; the Postgres server that Pixeltable started keeps running.
- all: the worker and every Postgres process die together, so the next start runs Postgres crash recovery. Unlike
  a power failure, writes the OS had not yet flushed to disk survive.
- alternate (default): python on odd cycles, all on even cycles.

Usage: python tool/abnormal_termination.py [--cycles N] [--rows N] [--kill MODE] [--home DIR] [--seed N]
"""

import io
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from argparse import SUPPRESS, ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import psutil
from pixeltable_pgserver.utils import PostmasterInfo

import pixeltable as pxt

TABLE_NAME = 'abnormal_termination'
MARKER = '@@ '  # prefix of the lines a worker prints to report progress to the driver
WORKER_TIMEOUT_SECS = 300.0  # time allowed for each wait on a worker, including cycle 0's whole insert


class CheckFailedError(Exception):
    pass


def report(tag: str, value: Any = None) -> None:
    print(f'{MARKER}{tag} {json.dumps(value)}', flush=True)


# Workers: each runs in a fresh process with PIXELTABLE_HOME set by the driver


def make_rows(cycle: int, num_rows: int) -> list[dict[str, Any]]:
    # Rows share a small pool of images; Pixeltable still writes a separate media file for every cell
    rng = np.random.default_rng(cycle)
    images = [PIL.Image.fromarray(rng.integers(0, 16, (64, 64, 3), dtype=np.uint8), 'RGB') for _ in range(64)]
    return [
        {
            'cycle': cycle,
            'idx': i,
            's': f'cycle {cycle} row {i} ' * 10,
            'j': {'cycle': cycle, 'idx': i, 'xs': list(range(i % 50))},
            'img': images[i % len(images)],
        }
        for i in range(num_rows)
    ]


def insert_worker(cycle: int, num_rows: int) -> None:
    pxt.init()
    if cycle == 0:
        t = pxt.create_table(
            TABLE_NAME, {'cycle': pxt.Int, 'idx': pxt.Int, 's': pxt.String, 'j': pxt.Json, 'img': pxt.Image}
        )
        t.add_computed_column(thumb=t.img.resize((16, 16)))
    else:
        t = pxt.get_table(TABLE_NAME)
    rows = make_rows(cycle, num_rows)
    report('inserting')
    start = time.monotonic()
    t.insert(rows)
    report('inserted', time.monotonic() - start)


def verify_worker(num_rows: int) -> None:
    pxt.init()
    t = pxt.get_table(TABLE_NAME)
    res = t.select(t.cycle, img=t.img.localpath, thumb=t.thumb.localpath).collect()
    row_counts = Counter(res['cycle'])
    partial = {cycle: n for cycle, n in sorted(row_counts.items()) if n != num_rows}
    if partial:
        sys.exit(f'Partially committed inserts ({{cycle: rows}}; each insert has {num_rows} rows): {partial}')
    bad_paths = [p for p in res['img'] + res['thumb'] if p is None or not os.path.isfile(p) or os.path.getsize(p) == 0]
    if bad_paths:
        sys.exit(f'{len(bad_paths)} committed media files are missing or empty, e.g. {bad_paths[:5]}')
    report('committed', sorted(row_counts))


# Driver


class Worker:
    """A worker process whose output is echoed to the console and scanned for progress markers."""

    def __init__(self, home: Path, label: str, args: list[str]) -> None:
        self.label = label
        # Inherited settings such as PIXELTABLE_PGDATA or PIXELTABLE_DB_CONNECT_STR could point the worker at a database
        # outside the home
        env = {k: v for k, v in os.environ.items() if not k.startswith('PIXELTABLE_') or k == 'PIXELTABLE_VERBOSITY'}
        env.update(PIXELTABLE_HOME=str(home), PIXELTABLE_PGDATA=str(home / 'pgdata'), PYTHONIOENCODING='utf-8')
        env.setdefault('PIXELTABLE_VERBOSITY', '0')
        self.proc = subprocess.Popen(
            [sys.executable, '-u', __file__, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        self._markers: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self.proc.stdout is not None
        for raw_line in self.proc.stdout:
            line = raw_line.rstrip()
            print(f'    [{self.label}] {line}', flush=True)
            if line.startswith(MARKER):
                tag, _, value = line[len(MARKER) :].partition(' ')
                self._markers.put((tag, json.loads(value)))
        self._markers.put(None)

    def wait_for(self, tag: str) -> Any:
        """Returns the value of the worker's next `tag` marker."""
        deadline = time.monotonic() + WORKER_TIMEOUT_SECS
        while True:
            try:
                item = self._markers.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise CheckFailedError(f'{self.label}: no {tag!r} after {WORKER_TIMEOUT_SECS:.0f}s') from None
            if item is None:
                raise CheckFailedError(f'{self.label}: exited with code {self.proc.wait()} before reporting {tag!r}')
            if item[0] == tag:
                return item[1]

    def join(self, timeout: float | None = None) -> int:
        """Waits for the worker to exit and its output to drain; returns its exit code."""
        returncode = self.proc.wait(timeout=timeout)
        self._reader.join()
        return returncode

    def finish(self) -> None:
        """Waits for the worker to exit and checks that it succeeded."""
        try:
            returncode = self.join(WORKER_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            raise CheckFailedError(f'{self.label}: still running after {WORKER_TIMEOUT_SECS:.0f}s') from None
        if returncode != 0:
            raise CheckFailedError(f'{self.label}: exited with code {returncode}')


def postgres_processes(pgdata: Path) -> list[psutil.Process]:
    """Returns the postmaster and its descendants, or [] if no server is running."""
    # After a kill, the pid in the stale postmaster.pid can belong to an unrelated process; PostmasterInfo sets
    # process only for a postgres process whose -D is this pgdata
    try:
        info = PostmasterInfo.read_from_pgdata(pgdata)
        if info is None or info.process is None:
            return []
        return [info.process, *info.process.children(recursive=True)]
    except (FileNotFoundError, psutil.Error):
        return []


def kill(procs: list[psutil.Process]) -> None:
    """Kills the processes in order (SIGKILL; TerminateProcess on Windows), then waits until all are gone."""
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=30)
    if alive:
        raise CheckFailedError(f'Processes still alive 30s after being killed: {[p.pid for p in alive]}')


def kill_worker(worker: Worker, mode: str, pgdata: Path) -> None:
    victims = [psutil.Process(worker.proc.pid)]
    if mode == 'all':
        pg_procs = postgres_processes(pgdata)
        if not pg_procs:
            raise CheckFailedError(f'No running Postgres server found for {pgdata}')
        # Postgres first: as in a crash, it gets no chance to abort the worker's transaction itself
        victims = pg_procs + victims
    kill(victims)
    worker.join()


def run_cycles(args: Namespace, home: Path) -> None:
    rng = random.Random(args.seed)
    pgdata = home / 'pgdata'

    print('Cycle 0: creating the table and timing an uninterrupted insert', flush=True)
    worker = Worker(home, 'insert 0', ['--worker', 'insert', '--cycle', '0', '--rows', str(args.rows)])
    insert_secs = worker.wait_for('inserted')
    worker.finish()
    print(f'Cycle 0: inserted {args.rows} rows in {insert_secs:.1f}s', flush=True)
    committed = [0]
    scheduled: Counter[str] = Counter()  # cycles per kill mode
    killed_by_mode: Counter[str] = Counter()

    for cycle in range(1, args.cycles + 1):
        mode = args.kill if args.kill != 'alternate' else ('python', 'all')[(cycle - 1) % 2]
        scheduled[mode] += 1
        worker = Worker(
            home, f'insert {cycle}', ['--worker', 'insert', '--cycle', str(cycle), '--rows', str(args.rows)]
        )
        worker.wait_for('inserting')
        delay = rng.uniform(0, insert_secs)
        try:
            returncode = worker.join(timeout=delay)
        except subprocess.TimeoutExpired:
            kill_worker(worker, mode, pgdata)
            killed = True
            print(f'Cycle {cycle}: killed ({mode}) {delay:.1f}s into the insert; verifying', flush=True)
        else:
            if returncode != 0:
                raise CheckFailedError(f'{worker.label}: exited with code {returncode}')
            killed = False
            print(f'Cycle {cycle}: insert finished before the kill at {delay:.1f}s; verifying', flush=True)
        killed_by_mode[mode] += killed

        verifier = Worker(home, f'verify {cycle}', ['--worker', 'verify', '--rows', str(args.rows)])
        found = verifier.wait_for('committed')
        verifier.finish()
        allowed = [committed, [*committed, cycle]] if killed else [[*committed, cycle]]
        if found not in allowed:
            raise CheckFailedError(f'{verifier.label}: committed inserts are {found}; expected one of {allowed}')
        committed = found
        print(f'Cycle {cycle}: OK; committed inserts: {committed}', flush=True)

    kills = ', '.join(f'{killed_by_mode[mode]} of {n} ({mode})' for mode, n in scheduled.items())
    untested = [mode for mode in scheduled if killed_by_mode[mode] == 0]
    if untested:
        raise CheckFailedError(f'No {" or ".join(untested)} kill: each such insert finished before its kill ({kills})')
    print(f'Passed: inserts killed mid-stream: {kills}; Pixeltable loaded normally after each', flush=True)


def main() -> None:
    parser = ArgumentParser(description='Kill Pixeltable partway through inserts and check that it loads normally.')
    parser.add_argument('--cycles', type=int, default=10, help='Number of kill-and-verify cycles (default: 10)')
    parser.add_argument('--rows', type=int, default=20000, help='Rows per insert (default: 20000)')
    parser.add_argument(
        '--kill', choices=['python', 'all', 'alternate'], default='alternate', help='Kill mode (default: alternate)'
    )
    parser.add_argument(
        '--home', type=Path, help='Pixeltable home for the run; must be new or empty (default: a new temp directory)'
    )
    parser.add_argument('--seed', type=int, help='Random seed for the kill times')
    parser.add_argument('--worker', choices=['insert', 'verify'], help=SUPPRESS)
    parser.add_argument('--cycle', type=int, default=0, help=SUPPRESS)
    args = parser.parse_args()

    if args.worker == 'insert':
        insert_worker(args.cycle, args.rows)
        return
    if args.worker == 'verify':
        verify_worker(args.rows)
        return

    if isinstance(sys.stdout, io.TextIOWrapper):
        # Echoed worker output may hold characters the console encoding (cp1252 on Windows) cannot represent
        sys.stdout.reconfigure(errors='backslashreplace')

    # The run kills the Postgres server of its Pixeltable home, so it must never use an existing one
    if args.home is None:
        home = Path(tempfile.mkdtemp(prefix='pxt-abnormal-termination-'))
    else:
        home = args.home.resolve()
        if home.exists() and any(home.iterdir()):
            sys.exit(f'--home must be a new or empty directory: {home}')
    if args.seed is None:
        args.seed = random.randrange(2**32)
    print(
        f'Pixeltable home: {home}\nSeed: {args.seed}  cycles: {args.cycles}  rows: {args.rows}  kill: {args.kill}',
        flush=True,
    )

    try:
        run_cycles(args, home)
    except CheckFailedError as e:
        print(f'FAILED: {e}', flush=True)
        sys.exit(1)
    finally:
        kill(psutil.Process().children(recursive=True) + postgres_processes(home / 'pgdata'))
    if args.home is None:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == '__main__':
    main()
