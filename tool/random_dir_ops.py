# Script that runs an infinite sequence of random directory operations.
import os
import random
import time
from datetime import datetime
from typing import Any

import pixeltable as pxt

dirs = ['A', 'B']
names = ['x', 'y', 'z']
ospid = os.getpid()


def list_objects(dir: str) -> tuple[list[str], list[str]]:
    tbls = pxt.list_tables(dir)
    subdirs = pxt.list_dirs(dir)
    assert len(set(tbls + subdirs)) == len(tbls) + len(subdirs), f'tables: {tbls}, subdirs: {subdirs}'
    assert set(tbls + subdirs).issubset({f'{dir}.{n}' for n in names}), f'tables: {tbls}, subdirs: {subdirs}'
    return tbls, subdirs


def log_op(msg: str) -> None:
    timestamp = datetime.fromtimestamp(time.time())
    print(timestamp, ospid, msg)


def create_drop(dir_name: str, if_exists: Any) -> None:
    log_op(f'adding  : {dir_name}')
    pxt.create_dir(dir_name, if_exists=if_exists)
    time.sleep(0.05)
    log_op(f'dropping: {dir_name}')
    pxt.drop_dir(dir_name, if_not_exists=if_exists)
    log_op(f'dropped : {dir_name}')


def run_ops():
    log_op(f'listD   : {pxt.list_dirs()}')
    if_exists = 'ignore'
    try:
        create_drop('C', if_exists=if_exists)
        create_drop('A.x', if_exists=if_exists)
        create_drop('B.x', if_exists=if_exists)
    except pxt.Error as e:
        print('pxt.ERROR: ---------------', e)


def random_dir_op():
    log_op(f'listDT  : {pxt.list_dirs()} , {pxt.list_tables()}')
    dir = random.choice(dirs)
    tbls, subdirs = list_objects(dir)
    existing = [path.split('.')[1] for path in tbls + subdirs]
    log_op(f'dir     : {dir}, existing: {existing}')
    ops: list[str] = []
    if len(existing) > 0:
        ops.append('move')
        ops.append('drop')
    if len(existing) < len(names):
        ops.append('create')
    op = random.choice(ops)

    try:
        if op == 'create':
            name = random.choice(list(set(names) - set(existing)))
            path = f'{dir}.{name}'
            log_op(f'creat   : {path}')
            pxt.create_dir(path)
        elif op == 'drop':
            name = random.choice(existing)
            path = f'{dir}.{name}'
            log_op(f'drop    : {path}')
            pxt.drop_dir(path)
        elif op == 'move':
            other_dir = next(iter(set(dirs) - {dir}))
            name = random.choice(existing)
            path, other_path = f'{dir}.{name}', f'{other_dir}.{name}'
            log_op(f'move    : {path} -> {other_path}')
            pxt.move(path, other_path)
    except pxt.Error as e:
        print(e)
    log_op(f'list2   : {pxt.list_dirs()}')


def main():
    pxt.init()
    for dir in dirs:
        pxt.create_dir(dir, if_exists='ignore')

    log_op(f'listIT  : {pxt.list_dirs()} , {pxt.list_tables()}')

    while True:
        run_ops()
        random_dir_op()
        time.sleep(0.001 * random.randint(1, 10))


if __name__ == '__main__':
    main()
