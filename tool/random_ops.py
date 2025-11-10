import dataclasses
import json
import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Callable, Iterator

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable.config import Config
from tool.worker_harness import run_workers

# List of table operations that can be performed by RandomTableOps.
# (operation_name, relative_prob, is_read_op)
# The numbers represent relative probabilities; they will be normalized to sum to 1.0. If this is a read-only
# worker, then only the operations with is_read_op=True will participate.
TABLE_OPS = (
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

OP_NAMES = {name for name, _, _ in TABLE_OPS}

# Basic schema for all tables created by RandomTableOps. Additional columns may be added or removed as the script
# progresses, but the basic columns (bc_*) will always be present.
BASIC_SCHEMA: dict[str, type] = {
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

# Initial rows to populate a newly created table. These will be augmented by additional rows as the script runs.
INITIAL_ROWS: list[dict[str, Any]] = [
    {
        'bc_string': f'str_{i}',
        'bc_int': i,
        'bc_float': float(i) * 1.1,
        'bc_bool': i % 3 == 0,
        'bc_timestamp': datetime.now(),
        'bc_date': f'2025-10-{i % 30 + 1:02d}',
        'bc_array': None,
        'bc_json': None,
        'bc_image': None,
    }
    for i in range(50)
]

# Primes used for selecting rows to update or delete; by filtering on rows modulo a prime, we ensure a distribution
# of filters that are overlapping but not nested.
PRIMES = (23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


# Configuration options with reasonable default values; they can be tuned via -D commandline parameters.
@dataclasses.dataclass
class RandomTableOpsConfig:
    num_base_tables: int = 4  # number of base tables to maintain (they will be occasionally dropped and recreated)
    num_column_names: int = 100  # number of possible column names (c0 ... c{n-1})
    num_view_names: int = 100  # number of possible view names (view_0 ... view_{n-1})
    random_array_freq: float = 0.1  # probability of including an array when inserting a row
    random_json_freq: float = 0.1  # probability of including a JSON object when inserting a row
    random_img_freq: float = 0.1  # probability of including an image when inserting a row


class RandomTableOps:
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
    - Capture the outcome of the operation in $PIXELTABLE_HOME/random-ops.log and on the console.
    - Sleep for a short random time (0.1 to 0.5 seconds) before starting the next iteration.
    """

    logger = logging.getLogger('random_ops')

    config: RandomTableOpsConfig
    base_table_names: tuple[str, ...]
    random_ops: list[tuple[float, Callable]]

    worker_id: int

    def __init__(
        self,
        worker_id: int,
        read_only: bool,
        include_only_ops: list[str],
        exclude_ops: list[str],
        config: RandomTableOpsConfig,
    ) -> None:
        self.worker_id = worker_id
        self.read_only = read_only
        self.config = config
        self.base_table_names = tuple(f'tbl_{i}' for i in range(config.num_base_tables))

        selected_ops: set[str]
        if include_only_ops:
            selected_ops = set(include_only_ops)
        else:
            selected_ops = OP_NAMES - set(exclude_ops)

        op_weights = {
            (op_name, weight)
            for op_name, weight, is_read_op in TABLE_OPS
            if op_name in selected_ops and (is_read_op or not read_only)
        }

        # Initialize random_ops.
        self.random_ops = []
        total_weight = sum(float(weight) for _, weight in op_weights)
        cumulative_weight = 0.0
        for op_name, weight in op_weights:
            cumulative_weight += float(weight)
            self.random_ops.append((cumulative_weight / total_weight, getattr(self, op_name)))

        handler = logging.FileHandler(Config.get().home / 'random-ops.log')
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
            name = random.choice(self.base_table_names)
            # If the table does not already exist, create it and populate with some initial data
            t = pxt.create_table(name, source=INITIAL_ROWS, schema_overrides=BASIC_SCHEMA, if_exists='ignore')
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
        rows = [
            {
                'bc_string': f'str_{i}',
                'bc_int': i,
                'bc_float': float(i) * 1.1,
                'bc_bool': i % 3 == 0,
                'bc_timestamp': datetime.now(),
                'bc_date': f'2025-10-{i % 30 + 1:02d}',
                'bc_array': self.random_array(self.config.random_array_freq),
                'bc_json': self.random_json(self.config.random_json_freq),
                'bc_image': self.random_img(self.config.random_img_freq),
            }
            for i in range(i_start, i_start + num_rows)
        ]

        us = t.insert(rows)
        arrays = sum(1 for row in rows if row['bc_array'] is not None)
        jsons = sum(1 for row in rows if row['bc_json'] is not None)
        imgs = sum(1 for row in rows if row['bc_image'] is not None)
        yield (
            f'Inserted {us.row_count_stats.ins_rows} rows '
            f'(with {arrays} arrays, {jsons} jsons, {imgs} images) (total now {t.count()} rows).'
        )

    def random_array(self, freq: float) -> np.ndarray | None:
        r = random.uniform(0, 1)
        if r >= freq:
            return None

        shape = (random.randint(1, 50), random.randint(1, 50))
        dtype = random.choice((np.int64, np.float32, np.bool, np.str_))
        return np.array(np.ones(shape, dtype))

    def random_json(self, freq: float) -> Any:
        r = random.uniform(0, 1)
        if r >= freq:
            return None

        match random.randint(0, 4):
            case 0:
                return {'this': 'is', 'a': 'simple', 'json': 'dict'}
            case 1:
                return ['this', 'is', 'a', 'simple', 'json', 'list']
            case 2:
                return {'json': 'dict', 'with': 'image', 'img': self.random_img(freq=1.0)}
            case 3:
                return {'json': 'dict', 'with': 'array', 'array': self.random_array(freq=1.0)}

    def random_img(self, freq: float) -> pxt.Image | None:
        """Return an image with probability `freq`. 50% of the time it will be an RGB image, 50% RGBA."""
        r = random.uniform(0, 1)
        if r >= freq:
            return None

        if r < freq / 2:
            random_data = np.random.randint(0, 16, size=(128, 128, 3), dtype=np.uint8)
            return PIL.Image.fromarray(random_data, 'RGB')
        else:
            random_data = np.random.randint(0, 16, size=(128, 128, 4), dtype=np.uint8)
            return PIL.Image.fromarray(random_data, 'RGBA')

    def update_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(PRIMES)
        yield f'Update rows in {self.tbl_descr(t)} where bc_int % {p} == 0: '
        # TODO: We should also do updates/deletes that can be carried out without a full table scan.
        us = t.where(t.bc_int % p == 0).update(
            {'bc_string': t.bc_string + '_u', 'bc_float': t.bc_float + 1.9, 'bc_bool': ~t.bc_bool}
        )
        yield f'Updated {us.row_count_stats.upd_rows}.'

    def delete_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(PRIMES)
        yield f'Delete rows from {self.tbl_descr(t)} where bc_int % {p} == 0: '
        us = t.where(t.bc_int % p == 0).delete()
        yield f'Deleted {us.row_count_stats.del_rows} rows (total now {t.count()}).'

    def add_data_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.config.num_column_names))
        cname = f'c{n}'
        yield f'Add data column {cname!r} to {self.tbl_descr(t)}: '
        t.add_column(**{cname: pxt.String}, if_exists='ignore')
        yield 'Success.'

    def add_computed_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.config.num_column_names))
        cname = f'c{n}'
        yield f'Add computed column {cname!r} to {self.tbl_descr(t)}: '
        t.add_computed_column(**{cname: t.bc_int * random.uniform(1.0, 5.0)}, if_exists='ignore')
        yield 'Success.'

    def drop_column(self) -> Iterator[str]:
        t = self.get_random_tbl()
        yield f'Drop a column from {self.tbl_descr(t)}: '
        cnames = [
            col_name
            for col_name, col in t.get_metadata()['columns'].items()
            if col['defined_in'] == t._name and col_name not in BASIC_SCHEMA
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
        n = int(random.uniform(0, self.config.num_view_names))
        vname = f'view_{n}'  # This will occasionally lead to name collisions, which is intended
        p = random.choice(PRIMES)
        yield f'Create view {vname!r} on {self.tbl_descr(t)}: '
        # TODO: Change 'ignore' to 'replace-force' after fixing PXT-774
        pxt.create_view(vname, t.where(t.bc_int % p == 0), if_exists='ignore')
        yield 'Success.'

    def rename_view(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_base_tbl=False)  # Must be a view
        if t is None:
            yield 'No views to rename.'
            return
        n = int(random.uniform(0, self.config.num_view_names))
        if f'view_{n}' == t._name:
            n = (n + 1) % self.config.num_view_names  # Ensure new name is different
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

    def random_op(self) -> None:
        """Select a random operation and run it once."""
        r = random.uniform(0, 1)
        for cumulative_weight, op in self.random_ops:
            if r < cumulative_weight:
                self.run_op(op)
                return

    def run(self) -> None:
        """Run random table operations indefinitely."""
        while True:
            self.random_op()
            time.sleep(random.uniform(0.1, 0.5))


def init(config: RandomTableOpsConfig) -> None:
    """Initialization. This will ONLY be run once (globally), on Worker 0."""
    print(json.dumps(dataclasses.asdict(config), indent=4))
    pxt.init()


def run(
    worker_id: int, read_only: bool, include_only_ops: list[str] | None, exclude_ops: list[str] | None, config_str: str
) -> None:
    """Entrypoint for a worker process."""
    os.environ['PIXELTABLE_DB'] = 'random_ops'
    os.environ['PIXELTABLE_VERBOSITY'] = '0'
    os.environ['PXTTEST_RANDOM_TBL_OPS'] = str(worker_id)
    config = RandomTableOpsConfig(**json.loads(config_str))

    # In order to localize initialization to a single process, we call pxt.init() only from worker 0. The timings are
    # adjusted so that all workers start issuing operations at approximately the same time.
    # TODO: Do we want pxt.init() to be concurrency-safe (the first time it is called, when setting up the DB)?
    if worker_id == 0:
        t = time.monotonic()
        init(config)
        time.sleep(5 - time.monotonic() + t)  # Sleep until 5 seconds after init
    else:
        time.sleep(5)

    try:
        RandomTableOps(worker_id, read_only, include_only_ops or [], exclude_ops or [], config).run()
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
    parser.add_argument('--include-only', nargs='+', type=str, help='List of operations to include')
    parser.add_argument('--exclude', nargs='+', type=str, help='List of operations to exclude')
    parser.add_argument('-D', action='append', type=str, help='Override config parameter, e.g., -D random_img_freq=1.0')
    return parser


def main() -> None:
    # TODO: Also provide a way to adjust the relative weights via commandline.
    parser = make_parser()
    args = parser.parse_args()

    if args.workers < 1:
        print('`workers` must be at least 1')
        sys.exit(1)

    if args.read_only_workers < 0 or args.read_only_workers > args.workers:
        print('--read-only-workers must be between 0 and `workers`')
        sys.exit(1)

    if args.include_only is not None and args.exclude is not None:
        print('Cannot use both --include-only and --exclude')
        sys.exit(1)

    if args.exclude is not None:
        for op_name in args.exclude:
            if op_name not in OP_NAMES:
                print(f'--exclude: unrecognized op name: {op_name}')
                sys.exit(1)

    if args.include_only is not None:
        for op_name in args.include_only:
            if op_name not in OP_NAMES:
                print(f'--include-only: unrecognized op name: {op_name}')
                sys.exit(1)

    config = RandomTableOpsConfig()
    if args.D is not None:
        for kv in args.D:
            assert isinstance(kv, str)
            if '=' not in kv:
                print(f'-D: expected param=value format; got: {kv}')
                sys.exit(1)
            name, value = kv.split('=', 1)
            if not hasattr(config, name):
                print(f'-D: unrecognized config parameter: {name}')
                sys.exit(1)
            field = getattr(config, name)
            try:
                setattr(config, name, type(field)(value))
            except Exception as e:
                print(
                    f'-D: error setting config parameter {name!r} '
                    f'(of type `{type(field).__name__}`) to value {value!r}: {e}'
                )
                sys.exit(1)

    config_str = repr(json.dumps(dataclasses.asdict(config)))

    worker_args = [
        [
            '-c',
            'from tool.random_ops import run; '
            f'run({i}, {i >= args.workers - args.read_only_workers}, '
            f'{args.include_only}, {args.exclude}, {config_str})',
        ]
        for i in range(args.workers)
    ]

    # Remove old logfile, if one exists
    (Config.get().home / 'random-ops.log').unlink(missing_ok=True)
    run_workers(args.workers, args.duration, worker_args=worker_args)


if __name__ == '__main__':
    main()
