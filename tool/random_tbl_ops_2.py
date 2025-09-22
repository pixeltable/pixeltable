# Script that runs an infinite sequence of random directory operations.

import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Callable

import pixeltable as pxt
from pixeltable.config import Config


class RandomTblOps:
    NUM_BASE_TABLES = 4
    BASE_TABLE_NAMES = tuple(f'tbl_{i}' for i in range(NUM_BASE_TABLES))
    BASIC_SCHEMA = {'c0': pxt.Int, 'c1': pxt.Float, 'c2': pxt.String}
    INITIAL_ROWS = list({'c0': i, 'c1': float(i) * 1.1, 'c2': f'str_{i}'} for i in range(50))
    PRIMES = (23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

    RANDOM_OPS_DEF = [
        ('query', 100),
        ('insert_rows', 30),
        ('update_rows', 15),
        ('delete_rows', 15),
        ('add_data_column', 5),
        ('add_computed_column', 5),
        ('drop_column', 3),
        #('add_view', 5),
        #('drop_view', 2),
        #('drop_table', 0.25),
    ]

    RANDOM_OPS: list[tuple[float, Callable]] = []

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

    def query(self) -> str:
        t = self.get_random_tbl(allow_view=True)
        name = t._name
        res = t.sample(n=50).collect()
        return f'Queried {len(res)} rows from {name!r}'

    def insert_rows(self) -> str:
        t = self.get_random_tbl(allow_view=False)
        name = t._name
        num_rows = int(random.uniform(20, 50))
        i_start = random.randint(100, 1000000000)
        us = t.insert([{'c0': i, 'c1': float(i) * 1.1, 'c2': f'str_{i}'} for i in range(i_start, i_start + num_rows)])
        return f'Inserted {us.row_count_stats.ins_rows} rows into {name!r} (total now {t.count()})'

    def update_rows(self) -> str:
        t = self.get_random_tbl(allow_view=False)
        name = t._name
        p = random.choice(self.PRIMES)
        us = t.where(t.c0 % p == 0).update({'c1': t.c1 + 1.9, 'c2': t.c2 + '_u'})
        return f'Updated {us.row_count_stats.upd_rows} rows in {name!r}'

    def delete_rows(self) -> str:
        t = self.get_random_tbl(allow_view=False)
        name = t._name
        p = random.choice(self.PRIMES)
        us = t.where(t.c0 % p == 0).delete()
        return f'Deleted {us.row_count_stats.del_rows} rows from {name!r} (total now {t.count()})'

    def add_data_column(self) -> str:
        t = self.get_random_tbl(allow_view=True)
        name = t._name
        n = int(random.uniform(0, 100))
        cname = f'c{n}'
        t.add_column(**{cname: pxt.String}, if_exists='ignore')
        return f'Added column {cname!r} to {name!r}'

    def add_computed_column(self) -> str:
        t = self.get_random_tbl(allow_view=True)
        name = t._name
        n = int(random.uniform(0, 100))
        cname = f'c{n}'
        t.add_computed_column(**{cname: t.c0 * random.uniform(1.0, 5.0)}, if_exists='ignore')
        return f'Added computed column {cname!r} to {name!r}'

    def drop_column(self) -> None:
        t = self.get_random_tbl(allow_view=True)
        name = t._name
        cnames = list(col_name for col_name, col in t.get_metadata()['columns'].items() if col['defined_in'] == t._name and col_name not in ('c0', 'c1', 'c2'))
        if len(cnames) == 0:
            return 'No columns to drop'
        cname = random.choice(cnames)
        t.drop_column(cname)
        return f'Dropped column {cname!r} from {name!r}'

    def run_op(self, op: Callable) -> None:
        try:
            res = op()
            self.emit(op, res)
        except pxt.Error as e:
            errmsg = str(e).replace('\n', ' ')
            self.emit(op, f'pxt.Error: {errmsg}')
        except Exception as e:
            errmsg = str(e).replace('\n', ' ')
            self.emit(op, f'FATAL ERROR: {e.__class__.__qualname__}: {errmsg}')
            raise

    def random_tbl_op(self) -> None:
        r = random.uniform(0, 1)
        for cumulative_weight, op in self.RANDOM_OPS:
            if r < cumulative_weight:
                self.run_op(op)
                return

    def run(self) -> None:
        # Initialize RANDOM_OPS.
        total_weight = sum(float(weight) for _, weight in self.RANDOM_OPS_DEF)
        cumulative_weight = 0.0
        for op_name, weight in self.RANDOM_OPS_DEF:
            cumulative_weight += float(weight)
            self.RANDOM_OPS.append((cumulative_weight / total_weight, getattr(self, op_name)))

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

    if worker_id == 0:
        pxt.init()
    else:
        time.sleep(5)

    RandomTblOps(worker_id).run()


if __name__ == '__main__':
    main()
