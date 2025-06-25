# Script that runs an infinite sequence of random directory operations.

import random
import sys
import time
from typing import Any

import pixeltable as pxt
import pixeltable.functions as pxtf

ROWS_THRESHOLD = 100
COLS_THRESHOLD = 20


class RandomTblOps:
    worker_id: int
    t: pxt.Table

    def __init__(self, worker_id: int, t: pxt.Table) -> None:
        self.worker_id = worker_id
        self.t = t

    def emit(self, s: Any) -> None:
        print(f'[{self.worker_id}] {time.monotonic()}: {s}')

    def random_tbl_op(self) -> None:
        t = self.t
        cnt = t.count()
        self.emit(f'count: {cnt}')
        # is_data_op = True
        # is_data_op = random.choice([True, False])
        is_data_op = random.uniform(0, 1) < 0.5

        if is_data_op:
            # add or delete rows
            max_val = t.select(pxtf.max(t.c1)).collect()[0, 0]
            if max_val is None:
                max_val = 0
            is_update = random.uniform(0, 1) < 0.1
            if is_update:
                # update rows
                self.emit(f'updating rows: c1 > {max_val - 10}')
                status = t.where(t.c1 > max_val - 10).update({'c1': t.c1 + 1})
                self.emit(f'updated {status.num_rows} rows')
                return

            is_delete = random.uniform(0, 1) < (cnt / ROWS_THRESHOLD)
            if is_delete:
                self.emit(f'deleting rows: c1 > {max_val - 10}')
                status = t.where(t.c1 > max_val - 10).delete()
                self.emit(f'deleted {status.num_rows} rows')
            else:
                # insert rows
                self.emit(f'inserting rows: {max_val + 1} - {max_val + 10}')
                status = t.insert({'c1': max_val + 1 + i} for i in range(10))
                self.emit(f'inserted {status.num_rows} rows')

        else:
            views = t.list_views()
            is_add = random.uniform(0, 1) < 0.5
            if is_add or len(views) == 0:
                view_name = f'v{len(views)}'
                self.emit(f'CREATING VIEW {view_name}')
                v = pxt.create_view(view_name, t, additional_columns={'v_1': t.c1 + 1}, if_exists='ignore')
                self.emit(f'CREATED VIEW {v._id}')
                cnt = v.where(v.v_1 == None).count()
                assert cnt == 0, cnt
            else:
                self.emit(f'DROPPING VIEW {views[0]}')
                pxt.drop_table(views[0], if_not_exists='ignore')

        # also check a view
        v_names = t.list_views()
        if len(v_names) > 0:
            try:
                v_name = random.choice(v_names)
                self.emit(f'checking view {v_name}')
                v = pxt.get_table(v_name)
                cnt = v.where(v.v_1 == None).count()
                assert cnt == 0, cnt
            except pxt.Error as e:
                self.emit(f'ERROR: {e}')

        # TODO: add/drop columns when we do compaction, otherwise we very quickly run into the limit on the number
        # of columns of a Postgres table

        # else:
        #     # add or drop columns
        #     num_cols = len(t.get_metadata()['schema'])
        #     is_drop = random.uniform(0, 1) < (num_cols / COLS_THRESHOLD)
        #     if is_drop and num_cols > 1:
        #         # drop the last added column
        #         col_name = list(t.get_metadata()['schema'].keys())[-1]
        #         t.drop_column(col_name, if_not_exists='ignore')
        #         print(f'dropped column {col_name}')
        #     else:
        #         col_name = f'computed_{num_cols}'
        #         t.add_computed_column(**{col_name: t.c1 + num_cols}, if_exists='ignore')
        #         print(f'added column {col_name}')

    def run(self) -> None:
        while True:
            try:
                self.random_tbl_op()
            except pxt.Error as e:
                self.emit(f'ERROR: {e}')
            time.sleep(random.uniform(0.1, 0.5))


def main() -> None:
    if len(sys.argv) == 1:
        worker_id = 0
    else:
        worker_id = int(sys.argv[1])

    t = pxt.create_table('random_tbl', schema={'c1': pxt.Int}, if_exists='ignore')
    t.add_computed_column(computed1=t.c1 + 10, if_exists='ignore')

    RandomTblOps(worker_id, t).run()


if __name__ == '__main__':
    main()
