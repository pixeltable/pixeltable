"""Benchmark workloads for pixeltable-rs (run through `bench/run.py`, one process per benchmark).

Each `bench_<name>()` sets up its tables, then times only the operation under test and returns the elapsed seconds.
The UDFs live in this module because a UDF must be defined in an importable module.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable

import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.func import Batch

# ------------------------------------------------------------------------------------------------ UDFs


@pxt.udf
def sleep_10ms(x: int) -> int:
    time.sleep(0.01)
    return x + 1


@pxt.udf
def sleep_10ms_b(x: int) -> int:
    time.sleep(0.01)
    return x * 2


_IMG: PIL.Image.Image | None = None


def _image() -> PIL.Image.Image:
    global _IMG
    if _IMG is None:
        rng = np.random.default_rng(0)
        img = PIL.Image.fromarray(rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8))
        img.load()
        _IMG = img
    return _IMG


@pxt.udf
def pil_resize(x: int) -> int:
    out = _image().resize((256 + x % 7, 256), PIL.Image.Resampling.LANCZOS)
    return out.size[0]


_MATS: tuple[np.ndarray, np.ndarray] | None = None


@pxt.udf
def numpy_matmul(x: int) -> float:
    global _MATS
    if _MATS is None:
        rng = np.random.default_rng(1)
        _MATS = (rng.random((300, 300)), rng.random((300, 300)))
    a, b = _MATS
    return float((a @ b)[x % 300, 0])


@pxt.udf
def py_loop(x: int) -> int:
    s = 0
    for i in range(20000):
        s += i * x % 7
    return s


@pxt.udf
def py_tiny(x: int) -> int:
    return x + 1


@pxt.udf
async def async_sleep(x: int) -> int:
    await asyncio.sleep(0.05)
    return x + 1


@pxt.udf(batch_size=16)
def batched_sleep(xs: Batch[int]) -> Batch[int]:
    time.sleep(0.02)
    return [x + 1 for x in xs]


@pxt.udf
def embed(s: str) -> pxt.Array[(64,), pxt.Float]:
    h = abs(hash(s)) % (2**32)
    v = np.random.default_rng(h).random(64, dtype=np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------------------------ helpers


def _int_table(name: str, n: int) -> pxt.Table:
    t = pxt.create_table(name, {'x': pxt.Int})
    t.insert({'x': i} for i in range(n))
    return t


def _timed(f: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def _insert_with(udf: Any, n: int) -> float:
    t = pxt.create_table('t', {'x': pxt.Int})
    t.add_computed_column(y=udf(t.x))
    rows = [{'x': i} for i in range(n)]
    elapsed = _timed(lambda: t.insert(rows))
    assert t.where(t.y == None).count() == 0
    return elapsed


# ------------------------------------------------------------------------------------------------ user code


def bench_sync_sleep_insert() -> float:
    """computed column on insert: sync UDF sleeping 10 ms, 1,000 rows"""
    return _insert_with(sleep_10ms, 1000)


def bench_sync_sleep_select() -> float:
    """select: sync UDF sleeping 10 ms, 1,000 rows"""
    t = _int_table('t', 1000)
    res: list[Any] = []
    elapsed = _timed(lambda: res.append(t.select(y=sleep_10ms(t.x)).collect()))
    assert res[0]['y'] == [i + 1 for i in range(1000)]
    return elapsed


def bench_pil_resize() -> float:
    """computed column on insert: sync UDF resizing a 1024x1024 image (PIL releases the GIL), 1,000 rows"""
    _image()
    return _insert_with(pil_resize, 1000)


def bench_numpy_matmul() -> float:
    """computed column on insert: sync UDF doing a 300x300 matmul (BLAS single-threaded), 1,000 rows"""
    return _insert_with(numpy_matmul, 1000)


def bench_py_cpu_loop() -> float:
    """computed column on insert: pure-Python CPU-bound UDF (~1 ms, holds the GIL), 1,000 rows"""
    return _insert_with(py_loop, 1000)


def bench_py_cpu_tiny() -> float:
    """computed column on insert: trivial pure-Python UDF (call overhead), 100,000 rows"""
    return _insert_with(py_tiny, 100_000)


def bench_py_cpu_tiny_select() -> float:
    """select: trivial pure-Python UDF (call overhead, no storage), 100,000 rows (best of 3)"""
    import pandas as pd

    t = pxt.create_table('t', source=pd.DataFrame({'x': range(100_000)}))
    return _best_of(lambda: t.select(y=py_tiny(t.x)).collect()['y'])


def bench_async_sleep() -> float:
    """computed column on insert: async UDF awaiting asyncio.sleep(0.05), 1,000 rows"""
    return _insert_with(async_sleep, 1000)


def bench_batched_sleep() -> float:
    """computed column on insert: batched UDF (batch_size=16) sleeping 20 ms per batch, 1,000 rows"""
    return _insert_with(batched_sleep, 1000)


def bench_two_cols_insert() -> float:
    """one insert fills two independent computed columns, each a sync UDF sleeping 10 ms, 500 rows"""
    t = pxt.create_table('t', {'x': pxt.Int})
    t.add_computed_column(y=sleep_10ms(t.x))
    t.add_computed_column(z=sleep_10ms_b(t.x))
    rows = [{'x': i} for i in range(500)]
    elapsed = _timed(lambda: t.insert(rows))
    assert t.where((t.y == None) | (t.z == None)).count() == 0
    return elapsed


def bench_add_column_sleep() -> float:
    """add_computed_column over 1,000 existing rows: sync UDF sleeping 10 ms"""
    t = _int_table('t', 1000)
    return _timed(lambda: t.add_computed_column(y=sleep_10ms(t.x)))


# ------------------------------------------------------------------------------------------------ builtins

N_BIG = 1_000_000


def _best_of(f: Callable[[], Any], n: int = 3) -> float:
    return min(_timed(f) for _ in range(n))


def bench_builtin_1m() -> dict[str, float]:
    """builtin-only work over 1M rows"""
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            'a': rng.integers(0, 1_000_000, N_BIG),
            'b': rng.random(N_BIG),
            'g': rng.integers(0, 1000, N_BIG),
            's': [f'str{i % 5000}' for i in range(N_BIG)],
        }
    )
    out: dict[str, float] = {}
    tables: list[pxt.Table] = []
    out['create_1m'] = _timed(lambda: tables.append(pxt.create_table('big', source=df)))
    t = tables[0]
    d = pxt.create_table('dim', source=pd.DataFrame({'g': np.arange(1000), 'name': [f'n{i}' for i in range(1000)]}))
    # the result column is converted to Python objects (upstream's collect() materializes them, pixeltable-rs's lazily)
    out['arith_1m'] = _best_of(lambda: t.select(x=t.a * 2 + t.b / 3 - t.g).collect()['x'])
    out['string_1m'] = _best_of(lambda: t.select(u=t.s.upper() + '_x').collect()['u'])
    out['filter_1m'] = _best_of(lambda: t.where((t.b > 0.5) & (t.g < 500)).count())
    out['groupby_1m'] = _best_of(
        lambda: t.group_by(t.g).select(t.g, s=pxtf.sum(t.a), m=pxtf.mean(t.b), c=pxtf.count(t.a)).collect()
    )
    out['sort_1m'] = _best_of(lambda: t.order_by(t.b).select(t.a).limit(10).collect())
    out['join_1m'] = _best_of(lambda: t.join(d, on=t.g == d.g).select(c=pxtf.count(d.name)).collect())
    out['add_column_1m'] = _timed(lambda: t.add_computed_column(c2=t.a * 2 + t.g))
    return out


def _median_of(f: Callable[[], Any], n: int = 5) -> float:
    return sorted(_timed(f) for _ in range(n))[n // 2]


def bench_interactive_1m() -> dict[str, float]:
    """interactive operations on a 1M-row table (median of 5): head/limit, filters without an index, update/delete with
    a non-indexed predicate, first head after a reload"""
    import pandas as pd

    from pixeltable import _testing

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            'a': rng.integers(0, 1_000_000, N_BIG),
            'b': rng.random(N_BIG),
            'g': rng.integers(0, 1000, N_BIG),
            's': [f'str{i % 5000}' for i in range(N_BIG)],
        }
    )
    t = pxt.create_table('big', source=df)
    out: dict[str, float] = {}
    res: list[Any] = []
    out['head10'] = _median_of(lambda: res.append(t.head(10)))
    assert list(res[-1]['a']) == list(df['a'][:10])
    out['limit10'] = _median_of(lambda: t.select(t.a, t.s).limit(10).collect())
    out['where_eq_noindex'] = _median_of(lambda: res.append(t.where(t.a == 12345).collect()))
    assert len(res[-1]) == int((df['a'] == 12345).sum())
    out['where_g_collect'] = _median_of(lambda: res.append(t.where(t.g == 5).select(t.s, t.b).collect()))
    assert len(res[-1]) == int((df['g'] == 5).sum())
    keys = iter(range(777, 10_000))
    out['update_1row_noindex'] = _median_of(lambda: t.update({'b': 0.25}, where=t.a == next(keys)))
    keys = iter(range(20_000, 30_000))
    out['delete_few_noindex'] = _median_of(lambda: t.delete(where=t.a == next(keys)))
    if hasattr(_testing, '_initialized'):
        # the harness's hooks for upstream Pixeltable (compat/upstream_baseline) wipe the database on their first use
        _testing._initialized = True
    _testing.reload_catalog()
    out['reload_first_head'] = _timed(lambda: res.append(pxt.get_table('big').head(10)))
    assert len(res[-1]) == 10
    return out


def bench_similarity_100k() -> dict[str, float]:
    """similarity over 100k 64-d embeddings (embedding index built by a numpy UDF)"""
    import pandas as pd

    n = 100_000
    t = pxt.create_table('docs', source=pd.DataFrame({'s': [f'document {i}' for i in range(n)]}))
    out: dict[str, float] = {}
    out['embed_index_100k'] = _timed(lambda: t.add_embedding_index('s', string_embed=embed))
    sim = t.s.similarity(string='document 42')
    res: list[Any] = []
    out['similarity_100k'] = _best_of(
        lambda: res.append(t.order_by(sim, asc=False).select(t.s, sim=sim).limit(10).collect())
    )
    assert res[-1]['s'][0] == 'document 42'
    return out


BENCHMARKS: dict[str, Callable[[], Any]] = {
    name[len('bench_') :]: f for name, f in sorted(globals().items()) if name.startswith('bench_') and callable(f)
}


def run_one(name: str) -> dict[str, float]:
    """Runs one benchmark in this (fresh) process; returns {result name: seconds}."""
    f = BENCHMARKS[name]
    r = f()
    return r if isinstance(r, dict) else {name: r}


def describe(name: str) -> str:
    return (BENCHMARKS[name].__doc__ or '').strip()


def main() -> None:
    """Entry point of a benchmark process (`python -c 'import benchmarks; benchmarks.main()' <name>`)."""
    import json
    import sys

    print(json.dumps(run_one(sys.argv[1])))
    sys.stdout.flush()
    os._exit(0)
