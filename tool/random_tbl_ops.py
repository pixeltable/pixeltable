# Script that runs an infinite sequence of random directory operations.

import random
import time

import pixeltable as pxt
import pixeltable.functions as pxtf

ROWS_THRESHOLD = 100
COLS_THRESHOLD = 20


def random_tbl_op(t: pxt.Table) -> None:
    cnt = t.count()
    print(f'count: {cnt}')
    is_data_op = True
    # is_data_op = random.choice([True, False])

    if is_data_op:
        # add or delete rows
        max_val = t.select(pxtf.max(t.c1)).collect()[0, 0]
        if max_val is None:
            max_val = 0
        is_update = random.uniform(0, 1) < 0.1
        if is_update:
            # update rows
            print(f'updating rows: c1 > {max_val - 10}')
            status = t.where(t.c1 > max_val - 10).update({'c1': t.c1 + 1})
            print(f'updated {status.num_rows} rows')
            return

        is_delete = random.uniform(0, 1) < (cnt / ROWS_THRESHOLD)
        if is_delete:
            print(f'deleting rows: c1 > {max_val - 10}')
            status = t.where(t.c1 > max_val - 10).delete()
            print(f'deleted {status.num_rows} rows')
        else:
            # insert rows
            print(f'inserting rows: {max_val + 1} - {max_val + 10}')
            status = t.insert({'c1': max_val + 1 + i} for i in range(10))
            print(f'inserted {status.num_rows} rows')

    else:
        # add or drop columns
        num_cols = len(t.get_metadata()['schema'])
        is_drop = random.uniform(0, 1) < (num_cols / COLS_THRESHOLD)
        if is_drop and num_cols > 1:
            # drop the last added column
            col_name = list(t.get_metadata()['schema'].keys())[-1]
            t.drop_column(col_name, if_not_exists='ignore')
            print(f'dropped column {col_name}')
        else:
            col_name = f'computed_{num_cols}'
            t.add_computed_column(**{col_name: t.c1 + num_cols}, if_exists='ignore')
            print(f'added column {col_name}')


def main() -> None:
    pxt.init()
    t = pxt.create_table('random_tbl', schema={'c1': pxt.Int}, if_exists='ignore')
    t.add_computed_column(computed1=t.c1 + 10, if_exists='ignore')

    while True:
        random_tbl_op(t)
        time.sleep(random.uniform(0.1, 0.5))


if __name__ == '__main__':
    main()
