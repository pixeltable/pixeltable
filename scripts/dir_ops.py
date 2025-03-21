# Script that runs an infinite sequence of random directory operations.

import random
import time

import pixeltable as pxt

dirs = ['A', 'B']
names = ['x', 'y', 'z']


def list_objects(dir: str) -> tuple[list[str], list[str]]:
    tbls = pxt.list_tables(dir)
    subdirs = pxt.list_dirs(dir)
    assert len(set(tbls + subdirs)) == len(tbls) + len(subdirs), f'tables: {tbls}, subdirs: {subdirs}'
    assert set(tbls + subdirs).issubset({f'{dir}.{n}' for n in names}), f'tables: {tbls}, subdirs: {subdirs}'
    return tbls, subdirs


def random_dir_op():
    dir = random.choice(dirs)
    tbls, subdirs = list_objects(dir)
    existing = [path.split('.')[1] for path in tbls + subdirs]
    print(f'dir: {dir}, existing: {existing}')
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
            print(f'create {path}')
            pxt.create_dir(path)
        elif op == 'drop':
            name = random.choice(existing)
            path = f'{dir}.{name}'
            print(f'drop {path}')
            pxt.drop_dir(path)
        elif op == 'move':
            other_dir = next(iter(set(dirs) - {dir}))
            name = random.choice(existing)
            path, other_path = f'{dir}.{name}', f'{other_dir}.{name}'
            print(f'move {path} -> {other_path}')
            pxt.move(path, other_path)
    except pxt.Error as e:
        print(e)
    _, _ = list_objects(dir)


def main():
    pxt.init()
    for dir in dirs:
        pxt.create_dir(dir, if_exists='ignore')

    while True:
        random_dir_op()
        time.sleep(0.25)


if __name__ == '__main__':
    main()
