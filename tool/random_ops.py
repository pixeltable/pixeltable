import dataclasses
import enum
import json
import logging
import os
import random
import re
import signal
import sys
import tempfile
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable.config import Config
from tool.worker_harness import run_workers


class OpStatus(enum.Enum):
    SUCCESS = 'success'
    EXPECTED_ERROR = 'expected_error'
    UNEXPECTED_ERROR = 'unexpected_error'


@dataclasses.dataclass(frozen=True)
class OpResult:
    status: OpStatus
    msg: str


def success(msg: str) -> OpResult:
    return OpResult(OpStatus.SUCCESS, msg)


# Errors that are expected under concurrent operation; any other exception indicates a failed run.
_EXPECTED_ERROR_PATTERNS = (
    re.compile(r'That Pixeltable operation could not be completed.+'),
    re.compile(r'Table was dropped'),
    re.compile(r'Column was dropped'),
    re.compile(r"Cannot use if_exists=.+ with the same name as one of the view's own ancestors"),
)


def is_expected(e: Exception) -> tuple[bool, str]:
    """Returns whether the error is expected, and a sanitized (with variable parts removed) message for stats
    collection."""
    sanitized = str(e).replace('\n', ' ')
    if not isinstance(e, pxt.Error):
        return False, sanitized

    for pattern in _EXPECTED_ERROR_PATTERNS:
        if pattern.match(sanitized):
            return True, pattern.pattern

    return False, sanitized


# List of table operations that can be performed by RandomTableOps.
# (operation_name, weight, is_read_op)
# The probability of selecting an operation is proportional to its weight. In read-only workers, only the operations
# with is_read_op=True are considered.
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
    - Capture the outcome of the operation in $PIXELTABLE_HOME/logs/random-ops.log and on the console.
    - Sleep for a short random time (0.1 to 0.5 seconds) before starting the next iteration.
    """

    logger = logging.getLogger('pixeltable.random_ops')

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
        *,
        stats_file: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.read_only = read_only
        self.config = config
        self.stats_file = stats_file
        self._op_counts: dict[str, int] = {name: 0 for name, *_ in TABLE_OPS}
        self._err_counts: dict[str, dict[str, int]] = {}  # op_name -> {sanitized_msg -> count}
        self._last_flush_ts: float = 0.0
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

        # This is necessary to include worker_id in all (including pixeltable) log records
        class WorkerIdFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.worker_id = worker_id
                return True

        log_path = Config.get().home / 'logs' / 'random-ops.log'
        os.makedirs(log_path.parent, exist_ok=True)
        random_ops_log_handler = logging.FileHandler(log_path)
        random_ops_log_handler.setLevel(logging.DEBUG)
        random_ops_log_handler.addFilter(WorkerIdFilter())
        formatter = logging.Formatter(
            '%(asctime)s %(process)d [Worker %(worker_id)02d] %(threadName)s '
            '%(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'
        )
        random_ops_log_handler.setFormatter(formatter)

        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(random_ops_log_handler)
        self.logger.propagate = False  # prevents double logging to stdout

        logging.getLogger('pixeltable').setLevel(logging.WARNING)
        logging.getLogger('pixeltable').addHandler(random_ops_log_handler)

        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

    def _flush_stats(self, *, force: bool = False) -> None:
        if self.stats_file is None:
            return
        if not force and time.monotonic() - self._last_flush_ts < 5.0:
            return
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump({'op_counts': self._op_counts, 'err_counts': self._err_counts}, f, indent=2)
        self._last_flush_ts = time.monotonic()

    def emit(self, op: Callable, result: OpResult) -> None:
        status = result.status.value
        line = f'[{datetime.now()}] [Worker {self.worker_id:02d}] [{op.__name__:19s}] [{status}]: {result.msg}'
        print(line)
        self.logger.info(f'[{op.__name__}] [{result.status.value}]: {result.msg}')

    @classmethod
    def tbl_descr(cls, t: pxt.Table) -> str:
        return f'{t._name()!r} ({t._id.hex[:6]}...)'

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

    def query(self) -> OpResult:
        t = self.get_random_tbl()
        num_rows = int(random.uniform(50, 100))
        res = t.sample(n=num_rows).collect()
        return success(f'Collected {len(res)} rows from {self.tbl_descr(t)}.')

    def insert_rows(self) -> OpResult:
        t = self.get_random_tbl(allow_view=False)
        num_rows = int(random.uniform(20, 50))
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
        return success(
            f'Inserted {us.row_count_stats.ins_rows} rows into {self.tbl_descr(t)} '
            f'(with {arrays} arrays, {jsons} jsons, {imgs} images).'
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

    def update_rows(self) -> OpResult:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(PRIMES)
        # TODO: We should also do updates/deletes that can be carried out without a full table scan.
        us = t.where(t.bc_int % p == 0).update(
            {'bc_string': t.bc_string + '_u', 'bc_float': t.bc_float + 1.9, 'bc_bool': ~t.bc_bool}
        )
        return success(f'Updated {us.row_count_stats.upd_rows} rows in {self.tbl_descr(t)} where bc_int % {p} == 0.')

    def delete_rows(self) -> OpResult:
        t = self.get_random_tbl(allow_view=False)
        p = random.choice(PRIMES)
        us = t.where(t.bc_int % p == 0).delete()
        n_del = us.row_count_stats.del_rows
        tbl = self.tbl_descr(t)
        return success(f'Deleted {n_del} rows from {tbl} where bc_int % {p} == 0.')

    def add_data_column(self) -> OpResult:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.config.num_column_names))
        cname = f'c{n}'
        t.add_column(**{cname: pxt.String}, if_exists='ignore')
        return success(f'Added data column {cname!r} to {self.tbl_descr(t)}.')

    def add_computed_column(self) -> OpResult:
        t = self.get_random_tbl()
        n = int(random.uniform(0, self.config.num_column_names))
        cname = f'c{n}'
        t.add_computed_column(**{cname: t.bc_int * random.uniform(1.0, 5.0)}, if_exists='ignore')
        return success(f'Added computed column {cname!r} to {self.tbl_descr(t)}.')

    def drop_column(self) -> OpResult:
        t = self.get_random_tbl()
        cnames = [
            col_name
            for col_name, col in t.get_metadata()['columns'].items()
            if col['defined_in'] == t._name() and col_name not in BASIC_SCHEMA
        ]
        if len(cnames) == 0:
            return success(f'No droppable columns in {self.tbl_descr(t)}.')
        cname = random.choice(cnames)
        t.drop_column(cname, if_not_exists='ignore')
        return success(f'Dropped column {cname!r} from {self.tbl_descr(t)}.')

    def create_view(self) -> OpResult:
        t = self.get_random_tbl()  # Allows views on views
        n = int(random.uniform(0, self.config.num_view_names))
        vname = f'view_{n}'  # This will occasionally lead to name collisions, which is intended
        p = random.choice(PRIMES)
        pxt.create_view(vname, t.where(t.bc_int % p == 0), if_exists='replace_force')
        return success(f'Created view {vname!r} on {self.tbl_descr(t)}.')

    def rename_view(self) -> OpResult:
        t = self.get_random_tbl(allow_base_tbl=False)  # Must be a view
        if t is None:
            return success('No views to rename.')
        n = int(random.uniform(0, self.config.num_view_names))
        if f'view_{n}' == t._name():
            n = (n + 1) % self.config.num_view_names  # Ensure new name is different
        new_name = f'view_{n}'  # This will occasionally lead to name collisions, which is intended
        old_descr = self.tbl_descr(t)
        pxt.move(t._name(), new_name, if_exists='ignore', if_not_exists='ignore')
        return success(f'Renamed view {old_descr} to {new_name!r}.')

    def drop_view(self) -> OpResult:
        t = self.get_random_tbl(allow_base_tbl=False)
        if t is None:
            return success('No views to drop.')
        pxt.drop_table(t, force=True, if_not_exists='ignore')
        return success(f'Dropped view {self.tbl_descr(t)}.')

    def drop_table(self) -> OpResult:
        t = self.get_random_tbl(allow_view=False)
        pxt.drop_table(t, force=True, if_not_exists='ignore')
        return success(f'Dropped table {self.tbl_descr(t)}.')

    def run_op(self, op: Callable) -> None:
        """Run the given operation once. Capture any "expected" errors and fail fatally on unexpected ones."""
        fatal: Exception | None = None
        try:
            result = op()
        except Exception as e:
            expected, sanitized = is_expected(e)
            if expected:
                result = OpResult(OpStatus.EXPECTED_ERROR, sanitized)
            else:
                result = OpResult(OpStatus.UNEXPECTED_ERROR, sanitized)
                fatal = e

        self._op_counts[op.__name__] += 1
        if result.status != OpStatus.SUCCESS:
            by_msg = self._err_counts.setdefault(op.__name__, {})
            by_msg[result.msg] = by_msg.get(result.msg, 0) + 1
        self._flush_stats()

        self.emit(op, result)
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


def run(
    worker_id: int,
    read_only: bool,
    include_only_ops: list[str] | None,
    exclude_ops: list[str] | None,
    config_str: str,
    *,
    stats_file: str | None = None,
) -> None:
    """Entrypoint for a worker process."""
    os.environ['PIXELTABLE_VERBOSITY'] = '0'
    os.environ['PXTTEST_RANDOM_TBL_OPS'] = str(worker_id)
    config = RandomTableOpsConfig(**json.loads(config_str))

    ops = RandomTableOps(worker_id, read_only, include_only_ops or [], exclude_ops or [], config, stats_file=stats_file)

    def _handle_sigterm(*_: object) -> None:
        raise SystemExit(0)

    # When the worker process receives SIGTERM from the coordinator, this handler will be executed in the current
    # (main) thread. The finally block below makes sure we flush latest stats before exiting.
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        ops.run()
    except KeyboardInterrupt:
        pass
    finally:
        ops._flush_stats(force=True)


def _print_stats(stats_files: list[Path]) -> None:
    total_ops: dict[str, int] = {name: 0 for name, *_ in TABLE_OPS}
    # op_name -> {sanitized_msg -> count}
    total_err_detail: dict[str, dict[str, int]] = {}
    for path in stats_files:
        if not path.exists():
            raise FileNotFoundError(f'Stats file not found: {path}')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for op, count in data['op_counts'].items():
            total_ops[op] = total_ops.get(op, 0) + count
        for op, by_msg in data['err_counts'].items():
            dest = total_err_detail.setdefault(op, {})
            for msg, count in by_msg.items():
                dest[msg] = dest.get(msg, 0) + count

    # Table 1: success/error counts per operation
    print('\n==== Per-operation stats ====\n')
    total_errs = {op: sum(by_msg.values()) for op, by_msg in total_err_detail.items()}
    print(f'{"Operation":<22} {"Ops":>8} {"Errors":>8} {"Err%":>6}\n')
    print('-' * 48)
    grand_total = 0
    for name, _, _ in TABLE_OPS:
        n_ops = total_ops[name]
        n_errs = total_errs.get(name, 0)
        err_pct = (100.0 * n_errs / n_ops) if n_ops > 0 else 0.0
        print(f'{name:<22} {n_ops:>8} {n_errs:>8} {err_pct:>5.0f}%')
        grand_total += n_ops
    print('-' * 48)
    print(f'{"Total":<22} {grand_total:>8}\n')

    print('==== Error stats ====\n')
    # Table 2: error counts by operation and sanitized error message
    rows: list[tuple[str, str, int]] = [
        (op, msg, count) for op, by_msg in total_err_detail.items() for msg, count in by_msg.items()
    ]
    if not rows:
        print('No errors recorded\n')
        return
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))
    msg_width = min(80, max(len(r[1]) for r in rows))
    print(f'{"Operation":<22} {"Count":>8}  {"Error"}\n')
    print('-' * (34 + msg_width))
    for op, msg, count in rows:
        print(f'{op:<22} {count:>8}  {msg}')


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

    config_dict = dataclasses.asdict(config)
    print(json.dumps(config_dict, indent=4))
    config_str = repr(json.dumps(config_dict))

    # Initialize Pixeltable before the actual test
    print('Running pxt.init()...')
    pxt.init()

    with tempfile.TemporaryDirectory() as tmp_dir:
        stats_files = [Path(tmp_dir) / f'random_ops_stats_{i}.json' for i in range(args.workers)]

        worker_args = [
            [
                '-u',  # run workers with unbuffered stdio to reduce buffering-related log reordering
                '-c',
                'from tool.random_ops import run; '
                f'run({i}, {i >= args.workers - args.read_only_workers}, '
                f'{args.include_only}, {args.exclude}, {config_str}, stats_file={str(stats_files[i])!r})',
            ]
            for i in range(args.workers)
        ]

        run_workers(args.workers, args.duration, worker_args=worker_args)
        _print_stats(stats_files)


if __name__ == '__main__':
    main()
