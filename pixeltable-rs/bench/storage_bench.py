"""Storage benchmarks: the cost of mutations, point reads, version reads and reloads as a table grows.

Each case runs in its own process with a fresh `$PIXELTABLE_HOME` (see `run_storage.py`, which runs all of them and
prints a table). Only the public API is used, plus `pixeltable._testing.reload_catalog()` for the reload case, so the
cases also run against upstream Pixeltable (with the compat harness's `upstream_baseline/_testing.py` installed).

    python storage_bench.py CASE [--rows N]      # prints one JSON object of metrics
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

BATCH = 100


def disk_bytes(home: Path) -> int:
    total = 0
    for root, _, files in os.walk(home):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def make_rows(start: int, n: int) -> list[dict[str, Any]]:
    return [{'id': i, 'val': i % 1000, 's': f'string value {i:08d}', 'f': i * 0.5} for i in range(start, start + n)]


def schema(pxt: Any) -> dict[str, Any]:
    return {'id': pxt.Int, 'val': pxt.Int, 's': pxt.String, 'f': pxt.Float}


def avg_ms(times: list[float]) -> float:
    """The median in milliseconds (robust against other load on the machine; totals capture rare slow operations)."""
    if not times:
        return 0.0
    s = sorted(times)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return round(med * 1000, 3)


def timed(f: Callable[[], Any]) -> tuple[float, Any]:
    t = time.perf_counter()
    r = f()
    return time.perf_counter() - t, r


def create(pxt: Any, pk: bool) -> Any:
    pxt.create_dir('bench', if_exists='ignore')
    return pxt.create_table('bench.t', schema(pxt), primary_key='id' if pk else None)


def fill(t: Any, n: int) -> None:
    t.insert(make_rows(0, n))


# ------------------------------------------------------------------------------------------------ cases


def insert_batches(pxt: Any, home: Path, n: int, pk: bool) -> dict[str, Any]:
    t = create(pxt, pk)
    times = []
    for start in range(0, n, BATCH):
        rows = make_rows(start, BATCH)
        dt, _ = timed(lambda: t.insert(rows))
        times.append(dt)
    return {
        'total_s': round(sum(times), 3),
        'first10_batch_ms': avg_ms(times[:10]),
        'last10_batch_ms': avg_ms(times[-10:]),
        'disk_mb': round(disk_bytes(home) / 1e6, 2),
    }


def case_insert_batches(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    return insert_batches(pxt, home, n, pk=False)


def case_insert_batches_pk(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    return insert_batches(pxt, home, n, pk=True)


def insert_one(pxt: Any, home: Path, n: int, pk: bool) -> dict[str, Any]:
    t = create(pxt, pk)
    rows = make_rows(0, n)
    dt, _ = timed(lambda: t.insert(rows))
    return {'total_s': round(dt, 3), 'disk_mb': round(disk_bytes(home) / 1e6, 2)}


def case_insert_one(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    return insert_one(pxt, home, n, pk=False)


def case_insert_one_pk(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    return insert_one(pxt, home, n, pk=True)


def case_point_lookup(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    t = create(pxt, pk=True)
    fill(t, n)
    rng = random.Random(0)
    t.where(t.id == 0).collect()  # warm-up (loads the table)
    pk_times, filter_times, count_times = [], [], []
    for _ in range(30):
        k = rng.randrange(n)
        dt, rows = timed(lambda: t.where(t.id == k).collect())
        assert len(rows) == 1
        pk_times.append(dt)
        s = f'string value {k:08d}'
        dt, rows = timed(lambda: t.where(t.s == s).collect())
        assert len(rows) == 1
        filter_times.append(dt)
        dt, _ = timed(lambda: t.where(t.id == k).count())
        count_times.append(dt)
    return {'pk_lookup_ms': avg_ms(pk_times), 'filter_lookup_ms': avg_ms(filter_times), 'pk_count_ms': avg_ms(count_times)}


def case_update(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    t = create(pxt, pk=True)
    fill(t, n)
    t.count()  # loads the table
    before = disk_bytes(home)
    k = max(n // 100, 1)
    upd, batch, one = [], [], []
    for r in range(5):
        lo = r * 2 * k
        dt, _ = timed(lambda: t.update({'val': t.val + 1}, where=(t.id >= lo) & (t.id < lo + k)))
        upd.append(dt)
        rows = [{'id': i, 'val': -i} for i in range(lo + k, lo + 2 * k)]
        dt, _ = timed(lambda: t.batch_update(rows))
        batch.append(dt)
        dt, _ = timed(lambda: t.update({'val': 7}, where=t.id == n // 2 + r))
        one.append(dt)
    return {
        'update_1pct_ms': avg_ms(upd),
        'batch_update_1pct_ms': avg_ms(batch),
        'update_1row_ms': avg_ms(one),
        'disk_growth_mb': round((disk_bytes(home) - before) / 1e6, 2),
    }


def case_delete(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    t = create(pxt, pk=True)
    fill(t, n)
    t.count()  # loads the table
    k = max(n // 100, 1)
    pct, one = [], []
    for r in range(5):
        hi = n - r * k
        dt, _ = timed(lambda: t.delete(where=(t.id >= hi - k) & (t.id < hi)))
        pct.append(dt)
        dt, _ = timed(lambda: t.delete(where=t.id == r))
        one.append(dt)
    return {'delete_1pct_ms': avg_ms(pct), 'delete_1row_ms': avg_ms(one)}


def case_time_travel(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    t = create(pxt, pk=True)
    for b in range(10):
        t.insert(make_rows(b * (n // 10), n // 10))
    k = max(n // 100, 1)
    for i in range(20):
        t.update({'val': t.val + 1}, where=(t.id >= i * k) & (t.id < (i + 1) * k))
    # version 3 holds 3/10 of the rows; the current version (30) holds all of them
    old = pxt.get_table('bench.t:3')
    old.count()  # warm-up
    assert old.count() == 3 * (n // 10)
    v15 = pxt.get_table('bench.t:15')
    old_count = [timed(lambda: old.count())[0] for _ in range(5)]
    old_lookup = [timed(lambda: old.where(old.id == 5).collect())[0] for _ in range(5)]
    mid_filter = [timed(lambda: v15.where(v15.val > 990).count())[0] for _ in range(5)]
    cur_count = [timed(lambda: t.count())[0] for _ in range(5)]
    return {
        'old_version_count_ms': avg_ms(old_count),
        'old_version_lookup_ms': avg_ms(old_lookup),
        'mid_version_filter_ms': avg_ms(mid_filter),
        'current_count_ms': avg_ms(cur_count),
    }


def case_reload(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    from pixeltable import _testing

    if hasattr(_testing, '_initialized'):
        # the harness's hooks for upstream Pixeltable (compat/upstream_baseline) wipe the database on their first use
        _testing._initialized = True
    t = create(pxt, pk=True)
    fill(t, n)
    t.update({'val': 0}, where=t.id < 10)
    _testing.reload_catalog()
    first_s, cnt = timed(lambda: pxt.get_table('bench.t').count())
    assert cnt == n
    t2 = pxt.get_table('bench.t')
    second_s = sorted(timed(lambda: t2.count())[0] for _ in range(5))[2]
    _testing.reload_catalog()
    lookup_s, _ = timed(lambda: pxt.get_table('bench.t').where(t2.id == 5).collect())
    return {
        'reload_first_count_ms': round(first_s * 1000, 2),
        'warm_count_ms': round(second_s * 1000, 2),
        'reload_first_lookup_ms': round(lookup_s * 1000, 2),
    }


def case_many_versions(pxt: Any, home: Path, n: int) -> dict[str, Any]:
    t = create(pxt, pk=False)
    times = []
    for i in range(1000):
        row = make_rows(i, 1)
        dt, _ = timed(lambda: t.insert(row))
        times.append(dt)
    return {
        'commit_1_10_ms': avg_ms(times[:10]),
        'commit_991_1000_ms': avg_ms(times[-10:]),
        'disk_mb': round(disk_bytes(home) / 1e6, 2),
    }


CASES: dict[str, Callable[[Any, Path, int], dict[str, Any]]] = {
    name[len('case_') :]: f for name, f in globals().items() if name.startswith('case_') and callable(f)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(CASES))
    ap.add_argument('--rows', type=int, default=100_000)
    args = ap.parse_args()
    home = Path(tempfile.mkdtemp(prefix='pxt-bench-'))
    os.environ['PIXELTABLE_HOME'] = str(home)
    os.environ.setdefault('PIXELTABLE_DB', 'bench')
    try:
        import pixeltable as pxt

        out = CASES[args.case](pxt, home, args.rows)
        json.dump(out, sys.stdout)
        print()
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == '__main__':
    main()
