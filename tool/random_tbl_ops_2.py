# Script that runs an infinite sequence of random directory operations.

import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Callable, ClassVar, Iterator

import pixeltable as pxt
from pixeltable.config import Config


class RandomTblOps:
    NUM_BASE_TABLES = 4
    BASE_TABLE_NAMES = tuple(f'tbl_{i}' for i in range(NUM_BASE_TABLES))
    BASIC_SCHEMA: ClassVar[dict[str, type]] = {'c0': pxt.Int, 'c1': pxt.Float, 'c2': pxt.String}
    INITIAL_ROWS: ClassVar[list[dict[str, Any]]] = [
        {'c0': i, 'c1': float(i) * 1.1, 'c2': f'str_{i}'} for i in range(50)
    ]
    PRIMES = (23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

    RANDOM_OPS_DEF = (
        ('query', 100),
        ('insert_rows', 30),
        ('update_rows', 15),
        ('delete_rows', 15),
        ('add_data_column', 5),
        ('add_computed_column', 5),
        ('drop_column', 3),
        # ('add_view', 5),
        # ('drop_view', 2),
        # ('drop_table', 0.25),
    )

    random_ops: list[tuple[float, Callable]]

    logger = logging.getLogger('random_tbl_ops')

    worker_id: int

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        # logging.basicConfig(filename='random-tbl-ops.log',
        #                 format='%(message)s',
        #                 level=logging.INFO)
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

    def get_random_tbl(self, allow_view: bool) -> pxt.Table:
        name = random.choice(self.BASE_TABLE_NAMES)
        # If the table does not already exist, create it and populate with some initial data
        t = pxt.create_table(name, source=self.INITIAL_ROWS, if_exists='ignore')
        if not allow_view:
            return t  # View not allowed
        if random.uniform(0, 1) < 0.5:
            return t  # Return base table 50% of the time
        views = t.list_views()
        if len(views) == 0:
            return t  # No views to choose from
        view = random.choice(views)
        return pxt.get_table(view)

    def query(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=True)
        num_rows = int(random.uniform(50, 100))
        yield f'Collect {num_rows} rows from {self.tbl_descr(t)}: '
        res = t.sample(n=num_rows).collect()
        yield f'Collected {len(res)} rows.'

    def insert_rows(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=False)
        num_rows = int(random.uniform(20, 50))
        yield f'Insert {num_rows} rows into {self.tbl_descr(t)}: '
        i_start = random.randint(100, 1000000000)
        us = t.insert([{'c0': i, 'c1': float(i) * 1.1, 'c2': f'str_{i}'} for i in range(i_start, i_start + num_rows)])
        yield f'Inserted {us.row_count_stats.ins_rows} rows (total now {t.count()}).'

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
        t = self.get_random_tbl(allow_view=True)
        n = int(random.uniform(0, 100))
        cname = f'c{n}'
        yield f'Add data column {cname!r} to {self.tbl_descr(t)}: '
        t.add_column(**{cname: pxt.String}, if_exists='ignore')
        yield 'Success.'

    def add_computed_column(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=True)
        n = int(random.uniform(0, 100))
        cname = f'c{n}'
        yield f'Add computed column {cname!r} to {self.tbl_descr(t)}: '
        t.add_computed_column(**{cname: t.c0 * random.uniform(1.0, 5.0)}, if_exists='ignore')
        yield 'Success.'

    def drop_column(self) -> Iterator[str]:
        t = self.get_random_tbl(allow_view=True)
        yield f'Drop a column from {self.tbl_descr(t)}: '
        cnames = [
            col_name
            for col_name, col in t.get_metadata()['columns'].items()
            if col['defined_in'] == t._name and col_name not in ('c0', 'c1', 'c2')
        ]
        if len(cnames) == 0:
            yield 'No columns to drop.'
        else:
            cname = random.choice(cnames)
            yield f'Column {cname!r}: '
            t.drop_column(cname)
            yield 'Success.'

    def run_op(self, op: Callable) -> None:
        msg_parts: list[str] = []
        fatal: Exception | None = None
        try:
            for res in op():
                msg_parts.append(res)
        except Exception as e:
            errmsg = str(e).replace('\n', ' ')
            if isinstance(e, pxt.Error) and (
                str(e)[:30]
                in (
                    # Whitelisted errors; these are expected in the current implementation.
                    'That Pixeltable operation coul',
                    'SQL error during execution of ',
                    'Column has been dropped (no re',
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
        r = random.uniform(0, 1)
        for cumulative_weight, op in self.random_ops:
            if r < cumulative_weight:
                self.run_op(op)
                return

    def run(self) -> None:
        # Initialize random_ops.
        self.random_ops = []
        total_weight = sum(float(weight) for _, weight in self.RANDOM_OPS_DEF)
        cumulative_weight = 0.0
        for op_name, weight in self.RANDOM_OPS_DEF:
            cumulative_weight += float(weight)
            self.random_ops.append((cumulative_weight / total_weight, getattr(self, op_name)))

        while True:
            self.random_tbl_op()
            time.sleep(random.uniform(0.1, 0.5))


def main() -> None:
    if len(sys.argv) == 1:
        worker_id = 0
    else:
        worker_id = int(sys.argv[1])

    os.environ['PIXELTABLE_DB'] = 'random_tbl_ops'
    os.environ['PIXELTABLE_VERBOSITY'] = '0'
    os.environ['PXTTEST_RANDOM_TBL_OPS'] = str(worker_id)

    if worker_id == 0:
        t = time.monotonic()
        pxt.init()
        time.sleep(5 - time.monotonic() + t)  # Sleep until 5 seconds after init
    else:
        time.sleep(5)

    RandomTblOps(worker_id).run()


if __name__ == '__main__':
    main()
