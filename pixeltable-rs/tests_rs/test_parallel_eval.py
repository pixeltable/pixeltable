"""Data-parallel evaluation of builtin-only work (`exec::par`): tables large enough to be evaluated on several threads
give the results, in the order, and the errors of sequential evaluation; the GIL is released while queries run."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

N = 20_000  # above exec::par::MIN_PARALLEL_ROWS


@pxt.udf
def embed(s: str) -> pxt.Array[(8,), pxt.Float]:
    v = np.random.default_rng(abs(hash(s)) % (2**32)).random(8, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def big() -> tuple[pxt.Table, pd.DataFrame]:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            'a': rng.integers(-1000, 1000, N),
            'b': rng.random(N),
            'g': rng.integers(0, 50, N),
            's': [f'v{i % 97}' for i in range(N)],
            'i': np.arange(N),
        }
    )
    return pxt.create_table('big', source=df), df


class TestParallelEval:
    def test_projection_and_filter(self, big: tuple[pxt.Table, pd.DataFrame]) -> None:
        t, df = big
        res = t.select(x=t.a * 3 - t.g, u=t.s.upper() + '!').collect()
        assert res['x'] == list(df.a * 3 - df.g)
        assert res['u'] == [s.upper() + '!' for s in df.s]
        res = t.where((t.b > 0.5) & (t.g < 25)).select(t.a).collect()
        assert res['a'] == list(df.a[(df.b > 0.5) & (df.g < 25)])
        assert t.where(t.a % 7 == 0).count() == int((df.a % 7 == 0).sum())

    def test_order_by_is_stable(self, big: tuple[pxt.Table, pd.DataFrame]) -> None:
        t, df = big
        expected = list(df.sort_values('g', kind='stable').a)
        assert t.order_by(t.g).select(t.a).collect()['a'] == expected
        assert t.order_by(t.g).select(t.a).limit(25).collect()['a'] == expected[:25]
        assert t.order_by(t.g).select(t.a).limit(25, offset=100).collect()['a'] == expected[100:125]
        desc = list(df.sort_values(['g', 'b'], ascending=[False, True], kind='stable').a)
        assert t.order_by(t.g, asc=False).order_by(t.b).select(t.a).collect()['a'] == desc

    def test_group_by(self, big: tuple[pxt.Table, pd.DataFrame]) -> None:
        t, df = big
        res = t.group_by(t.g).select(
            t.g, s=pxtf.sum(t.a), c=pxtf.count(t.a), mx=pxtf.max(t.s), mn=pxtf.min(t.b), m=pxtf.mean(t.b)
        ).collect()
        grouped = df.groupby('g')
        assert res['g'] == sorted(df.g.unique())
        assert res['s'] == [int(grouped.a.sum()[g]) for g in res['g']]
        assert res['c'] == [int(grouped.a.count()[g]) for g in res['g']]
        assert res['mx'] == [grouped.s.max()[g] for g in res['g']]
        assert res['mn'] == [grouped.b.min()[g] for g in res['g']]
        assert res['m'] == pytest.approx([grouped.b.mean()[g] for g in res['g']], rel=1e-12)
        # make_list keeps row order across chunks
        lists = t.group_by(t.g).select(t.g, l=pxtf.json.make_list(t.a)).collect()
        assert lists['l'] == [list(grouped.a.apply(list)[g]) for g in lists['g']]
        # a whole-table aggregate, and one over a join
        assert t.select(s=pxtf.sum(t.a), c=pxtf.count(t.s)).collect()[0] == {'s': int(df.a.sum()), 'c': N}
        # an error in a later chunk (rows from 18000 on overflow) is the query's error
        with pytest.raises(pxt.Error, match='bigint out of range'):
            t.group_by(t.g).select(t.g, s=pxtf.sum(t.i // 18000 * 2 * 4611686018427387904)).collect()

    def test_join(self, big: tuple[pxt.Table, pd.DataFrame]) -> None:
        t, df = big
        d = pxt.create_table('dim', source=pd.DataFrame({'g': np.arange(0, 50, 2), 'name': [f'n{i}' for i in range(0, 50, 2)]}))
        res = t.join(d, on=t.g == d.g).select(t.a, d.name).collect()
        kept = df[df.g % 2 == 0]
        assert res['a'] == list(kept.a)
        assert res['name'] == [f'n{g}' for g in kept.g]
        res = t.join(d, on=t.g == d.g, how='left').select(t.a, d.name).collect()
        assert res['name'] == [f'n{g}' if g % 2 == 0 else None for g in df.g]

    def test_first_error_in_row_order(self, big: tuple[pxt.Table, pd.DataFrame]) -> None:
        t, _ = big
        # every row but those with a == 0 or a == 1 overflows; the error is the same as sequential evaluation's
        with pytest.raises(pxt.Error, match='bigint out of range'):
            t.select(t.a * 9223372036854775807).collect()

    def test_similarity(self) -> None:
        n = 6000
        t = pxt.create_table('docs', source=pd.DataFrame({'s': [f'doc {i}' for i in range(n)]}))
        t.add_embedding_index('s', string_embed=embed)
        sim = t.s.similarity(string='doc 4242')
        res = t.order_by(sim, asc=False).select(t.s, sim=sim).limit(3).collect()
        assert res['s'][0] == 'doc 4242'
        assert res['sim'][0] == pytest.approx(1.0, abs=1e-5)
        assert res['sim'] == sorted(res['sim'], reverse=True)

    def test_computed_columns(self) -> None:
        t = pxt.create_table('cc', {'a': pxt.Int})
        t.add_computed_column(b=t.a * 2)
        t.add_computed_column(c=(t.b + 1) // (t.a % 5))  # ZeroDivisionError where a % 5 == 0
        status = t.insert(({'a': i} for i in range(N)), on_error='ignore')
        assert status.num_excs == N // 5
        assert status.updated_cols == [] and status.cols_with_excs == ['cc.c']
        res = t.order_by(t.a).select(t.a, t.b, t.c, err=t.c.errortype).collect()
        assert res['b'] == [2 * a for a in range(N)]
        assert res['c'] == [None if a % 5 == 0 else (2 * a + 1) // (a % 5) for a in range(N)]
        assert res['err'] == ['ZeroDivisionError' if a % 5 == 0 else None for a in range(N)]
        with pytest.raises(pxt.Error, match='ZeroDivisionError'):
            t.insert(({'a': i} for i in range(1, N)), on_error='abort')
        assert t.count() == N
        status = t.add_computed_column(d=t.a + t.b, on_error='abort')
        assert status.num_excs == 0
        assert t.order_by(t.a).select(t.d).collect()['d'] == [3 * a for a in range(N)]
        with pytest.raises(pxt.Error, match="Error while evaluating computed column 'e'"):
            t.add_computed_column(e=t.a // (t.a % 3), on_error='abort')
        assert 'e' not in t.columns()
        t.update({'a': t.a + 1}, where=t.a < 10)
        assert t.order_by(t.a).select(t.d).limit(3).collect()['d'] == [3, 6, 9]

    def test_gil_released_during_query(self) -> None:
        n = 400_000
        t = pxt.create_table('gil', source=pd.DataFrame({'a': np.arange(n), 'b': np.arange(n) % 13}))
        ticks = [0]
        done = threading.Event()

        def spin() -> None:
            while not done.is_set():
                ticks[0] += 1
                time.sleep(0)

        thread = threading.Thread(target=spin)
        thread.start()
        try:
            start = time.monotonic()
            before = ticks[0]
            for _ in range(3):
                t.order_by(t.b * 7 - t.a).select(t.a).collect()
            elapsed = time.monotonic() - start
            progressed = ticks[0] - before
        finally:
            done.set()
            thread.join()
        # the other thread ran while the queries held no GIL (with the GIL held it would barely move)
        assert progressed > 1000 * elapsed
