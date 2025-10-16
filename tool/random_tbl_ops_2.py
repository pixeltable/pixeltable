import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable.config import Config
from tool.worker_harness import run_workers


class RandomTblOps:
    """
    Runs random table operations on a single worker.

    The operations will run over a configurable number of base table names. The base tables will be created and dropped
    from time to time, but always using the same fixed pool of names, 'tbl_0' ... 'tbl_{n-1}'.

    Optionally, the worker can be configured to be read-only, in which case it will be limited to running queries (no
    operations that modify table data or schemas).

    At each iteration, the following steps take place:
    - Select a random operation. The operations will occur with varying frequencies as defined in `RANDOM_OPS_DEF`.
    - Select a random base table for the operation, from 0 to n-1. If the table does not exist (either because it is
        the first time it has been selected, or because it was recently dropped), it will first be created and
        populated with some initial data.
    - If the operation supports only base tables, carry out the operation on the selected table.
    - If the operation supports views and the table has at least one view, then: 50% of the time, carry out the
        operation on the selected base table; 50% of the time, carry it out on a random view of the selected table.
    - Capture the outcome of the operation in $PIXELTABLE_HOME/random-tbl-ops.log and on the console.
    - Sleep for a short random time (0.1 to 0.5 seconds) before starting the next iteration.
    """

    logger = logging.getLogger('random_tbl_ops')

    # TODO: Support additional operations such as index ops, pxt.move(), and replicas
    # TODO: Add additional datatypes including media data
    NUM_BASE_TABLES = 4
    BASE_TABLE_NAMES = tuple(f'tbl_{i}' for i in range(NUM_BASE_TABLES))
    BASIC_SCHEMA: ClassVar[dict[str, type]] = {
        'bc_string': pxt.String,
        'bc_int': pxt.Int,
        'bc_float': pxt.Float,
        'bc_bool': pxt.Bool,
        'bc_timestamp': pxt.Timestamp,
        'bc_date': pxt.Date,
        'bc_array': pxt.Array,
        'bc_json': pxt.Json,
        'bc_image': pxt.Image,
    }
    INITIAL_ROWS: ClassVar[list[dict[str, Any]]] = [
        {
            'bc_string': f'str_{i}',
            'bc_int': i,
            'bc_float': float(i) * 1.1,
            'bc_bool': (i % 3 == 0),
            'bc_timestamp': datetime.now(),
            'bc_date': f'2025-10-{(i % 30) + 1}',
            'bc_array': None,
            'bc_json': None,
            'bc_image': None,
        }
        for i in range(50)
    ]
    PRIMES = (23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    NUM_COLUMN_NAMES = 100  # c0 ... c{n-1}
    NUM_VIEW_NAMES = 100  # view_0 ... view_{n-1}

    # (operation_name, relative_prob, is_read_op)
    # The numbers represent relative probabilities; they will be normalized to sum to 1.0. If this is a read-only
    # worker, then only the operations with is_read_op=True will participate in the normalization.
    RANDOM_OPS_DEF = (
        ('query', 100, True),
        ('insert_rows', 30, False),
        ('update_rows', 15, False),
        ('delete_rows', 15, False),
        ('add_data_column', 5, False),
        ('add_computed_column', 5, False),
        ('drop_column', 3, False),
        ('create_view', 5, False),
        ('rename_view', 5, False),
        ('drop_view', 1, False),
        ('drop_table', 0.25, False),
    )
    OP_NAMES = set(name for name, _, _ in RANDOM_OPS_DEF)

    random_ops: list[tuple[float, Callable]]

    worker_id: int

    def __init__(self, worker_id: int, read_only: bool, exclude_ops: list[str]) -> None:
        self.worker_id = worker_id
        self.read_only = read_only
        ops_config = {
            (op_name, weight)
            for op_name, weight, is_read_op in self.RANDOM_OPS_DEF
            if op_name not in exclude_ops and (is_read_op or not read_only)
        }

        # Initialize random_ops.
        self.random_ops = []
        total_weight = sum(float(weight) for _, weight in ops_config)
        cumulative_weight = 0.0
        for op_name, weight in ops_config:
            cumulative_weight += float(weight)
            self.random_ops.append((cumulative_weight / total_weight, getattr(self, op_name)))

        handler = logging.FileHandler(Config.get().home / 'random-tbl-ops.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def emit(self, op: Callable, msg: Any) -> None:
        line = f'[{datetime.now()}] [Worker {self.worker_id:02d}] [{op.__name__:19s}]: {msg}'
        print(line)
        self.logger.info(line)

    @classmethod
    def tbl_descr(cls, t: pxt.Table) -> str:
        return f'{t._name!r} ({t._id.hex[:6]}...)'

    def get_random_tbl(self, allow_base_tbl: bool = True, allow_view: bool = True) -> pxt.Table | None:
        # Occasionally it happens that we get a list of views, but by the time we try to get one of them, it has been
        # dropped by another process. So we wrap the whole operation in a while loop and keep trying until it succeeds.
        # (At least 99% of the time it succeeds on the first try.)
        while True:
            name = random.choice(self.BASE_TABLE_NAMES)
            # If the table does not already exist, create it and populate with some initial data
            t = pxt.create_table(name, source=self.INITIAL_ROWS, schema_overrides=self.BASIC_SCHEMA, if_exists='ignore')
            if not allow_view:
                return t  # View not allowed
            if allow_base_tbl and random.uniform(0, 1) < 0.5:
                return t  # Return base table 50% of the time
            view_names = t.list_views()
            if len(view_names) == 0:
                return t if allow_base_tbl else None  # No views to choose from
            view_name = random.choice(view_names)
            view = pxt.get_table(view_name, if_not_exists='ignore')
            if view is not None:
                return view

    def query(self) -> Iterator[str]:
        t = self.get_random_tbl()
        num_rows = int(random.uniform(50, 100))
        yield f'Collect {num_rows} rows from {self.tbl_descr(t)}: '
        res = t.sample(n=num_rows).collect()
        yield f'Collected {len(res)} rows.'

    def insert_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        num_rows = int(random.uniform(20, 50))
        yield f'Insert {num_rows} rows into {self.tbl_descr(t)}: '
        i_start = random.randint(100, 1000000000)
        us = t.insert(
            [
                {'c0': i, 'c1': float(i) * 1.1, 'c2': f'str_{i}', 'c3': self.random_img()}
                for i in range(i_start, i_start + num_rows)
            ]
        )
        yield f'Inserted {us.row_count_stats.ins_rows} rows (total now {t.count()}).'

    def random_img(self) -> pxt.Image | None:
        r = random.uniform(0, 1)
        if r < 0.9:
            return None

        if r < 0.95:
            random_data = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
            return PIL.Image.fromarray(random_data, 'RGB')
        else:
            random_data = np.random.randint(0, 256, size=(256, 256, 4), dtype=np.uint8)
            return PIL.Image.fromarray(random_data, 'RGBA')

    def update_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(self.PRIMES)
        yield f'Update rows in {self.tbl_descr(t)} where c0 % {p} == 0: '
        us = t.where(t.c0 % p == 0).update({'c1': t.c1 + 1.9, 'c2': t.c2 + '_u'})
        yield f'Updated {us.row_count_stats.upd_rows}.'

    def delete_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(self.PRIMES)
        yield f'Delete rows from {self.tbl_descr(t)} where c0 % {p} == 0: '
        us = t.where(t.c0 % p == 0).delete()
        yield f'Deleted {us.row_count_stats.del_rows} rows (total now {t.count()}).'

    def add_data_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.NUM_COLUMN_NAMES))
        cname = f'c{n}'
        yield f'Add data column {cname!r} to {self.tbl_descr(t)}: '
        t.add_column(**{cname: pxt.String}, if_exists='ignore')
        yield 'Success.'

    def add_computed_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.NUM_COLUMN_NAMES))
        cname = f'c{n}'
        yield f'Add computed column {cname!r} to {self.tbl_descr(t)}: '
        t.add_computed_column(**{cname: t.c0 * random.uniform(1.0, 5.0)}, if_exists='ignore')
        yield 'Success.'

    def drop_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        yield f'Drop a column from {self.tbl_descr(t)}: '
        cnames = [
            col_name
            for col_name, col in t.get_metadata()['columns'].items()
            if col['defined_in'] == t._name and col_name not in self.BASIC_SCHEMA
        ]
        if len(cnames) == 0:
            yield 'No columns to drop.'
        else:
            cname = random.choice(cnames)
            yield f'Column {cname!r}: '
            t.drop_column(cname, if_not_exists='ignore')
            yield 'Success.'

    def create_view(self) -> Iterator[str]:
        t = self.get_random_tbl()  # Allows views on views
        n = int(random.uniform(0, self.NUM_VIEW_NAMES))
        vname = f'view_{n}'  # This will occasionally lead to name collisions, which is intended
        p = random.choice(self.PRIMES)
        yield f'Create view {vname!r} on {self.tbl_descr(t)}: '
        # TODO: Change 'ignore' to 'replace-force' after fixing PXT-774
        pxt.create_view(vname, t.where(t.c0 % p == 0), if_exists='ignore')
        yield 'Success.'

    def rename_view(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_base_tbl=False)  # Must be a view
        if t is None:
            yield 'No views to rename.'
            return
        n = int(random.uniform(0, self.NUM_VIEW_NAMES))
        if f'view_{n}' == t._name:
            n = (n + 1) % self.NUM_VIEW_NAMES  # Ensure new name is different
        new_name = f'view_{n}'  # This will occasionally lead to name collisions, which is intended
        yield f'Rename view {self.tbl_descr(t)} to {new_name!r}: '
        pxt.move(t._name, new_name, if_exists='ignore', if_not_exists='ignore')
        yield 'Success.'

    def drop_view(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_base_tbl=False)
        if t is None:
            yield 'No views to drop.'
            return
        yield f'Drop view {self.tbl_descr(t)}: '
        pxt.drop_table(t, force=True, if_not_exists='ignore')
        yield 'Success.'

    def drop_table(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        yield f'Drop table {self.tbl_descr(t)}: '
        pxt.drop_table(t, force=True, if_not_exists='ignore')
        yield 'Success.'

    def run_op(self, op: Callable) -> None:
        """Run the given operation once. Capture any "expected" errors and fail fatally on unexpected ones."""
        msg_parts: list[str] = []
        fatal: Exception | None = None
        try:
            for res in op():
                msg_parts.append(res)
        except Exception as e:
            errmsg = str(e).replace('\n', ' ')
            if isinstance(e, pxt.Error) and (
                str(e)[:17]
                in (
                    # Whitelisted errors; these are expected in the current implementation.
                    # Any other exception indicates a failed run.
                    'That Pixeltable o',  # Concurrency conflict
                    'Table was dropped',  # Table dropped by another process
                    'Column was droppe',  # Column dropped by another process
                )
            ):
                msg_parts.append(f'pxt.Error: {errmsg}')
            else:
                msg_parts.append(f'FATAL ERROR: {e.__class__.__qualname__}: {errmsg}')
                fatal = e

        self.emit(op, ''.join(msg_parts))
        if fatal is not None:
            raise fatal

    def random_tbl_op(self) -> None:
        """Select a random operation and run it once."""
        r = random.uniform(0, 1)
        for cumulative_weight, op in self.random_ops:
            if r < cumulative_weight:
                self.run_op(op)
                return

    def run(self) -> None:
        """Run random table operations indefinitely."""
        while True:
            self.random_tbl_op()
            time.sleep(random.uniform(0.1, 0.5))


def run(worker_id: int, read_only: bool, exclude_ops: list[str] | None) -> None:
    """Entrypoint for a worker process."""
    os.environ['PIXELTABLE_DB'] = 'random_tbl_ops'
    os.environ['PIXELTABLE_VERBOSITY'] = '0'
    os.environ['PXTTEST_RANDOM_TBL_OPS'] = str(worker_id)

    # In order to localize initialization to a single process, we call pxt.init() only from worker 0. The timings are
    # adjusted so that all workers start issuing operations at approximately the same time.
    # TODO: Do we want pxt.init() to be concurrency-safe (the first time it is called, when setting up the DB)?
    if worker_id == 0:
        t = time.monotonic()
        pxt.init()
        time.sleep(5 - time.monotonic() + t)  # Sleep until 5 seconds after init
    else:
        time.sleep(5)

    try:
        RandomTblOps(worker_id, read_only, exclude_ops or []).run()
    except KeyboardInterrupt:
        # Suppress the stack trace, but abort.
        pass


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Run random table operations.')
    parser.add_argument('workers', type=int, help='Number of worker processes to start')
    parser.add_argument('duration', type=float, help='Duration to run (in seconds)')
    parser.add_argument(
        '-r', '--read-only-workers', type=int, default=0, help='Number of read-only workers (default: 0)'
    )
    parser.add_argument('--exclude', nargs='+', type=str, help='List of operations to exclude')
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.workers < 1:
        print('`workers` must be at least 1')
        sys.exit(1)

    if args.read_only_workers < 0 or args.read_only_workers > args.workers:
        print('--read-only-workers must be between 0 and `workers`')
        sys.exit(1)

    if args.exclude is not None:
        for op_name in args.exclude:
            if op_name not in RandomTblOps.OP_NAMES:
                print(f'--exclude: unrecognized op name: {op_name}')
                sys.exit(1)

    worker_args = [
        [
            '-c',
            'from tool.random_tbl_ops_2 import run; '
            f'run({i}, {i >= args.workers - args.read_only_workers}, {args.exclude})',
        ]
        for i in range(args.workers)
    ]

    run_workers(args.workers, args.duration, worker_args=worker_args)


if __name__ == '__main__':
    main()
