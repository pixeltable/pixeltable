"""The fast query paths give the results of row-by-row evaluation: key-ordered scans of tables with several segments
and updated rows (`TableData::scan_order()`), limit pushdown, filters over typed chunk columns (`exec::fast::VPred`),
typed aggregation and its parallel merge, integer-key joins, numeric top-k and sorts, update/delete row selection.

Each query is compared with the same query written so that it takes the generic path (a column `c` as `c + 0`, which
is evaluated row by row), on data with NULLs, NaNs, mixed signs and timestamps, large enough for several storage chunks
(8,192 rows), parallel chunks (16,384 rows) and segments."""

import datetime
import math
from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

N = 40_000
BATCH = 8_000
BASE = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


def row(i: int) -> dict[str, Any]:
    b: float | None = ((i * 7919) % 2003 - 1000) / 37.0
    if i % 97 == 0:
        b = math.nan
    elif i % 89 == 0:
        b = None
    return {
        'id': i,
        'a': None if i % 13 == 0 else (i * 31) % 1001 - 500,
        'b': b,
        'g': None if i % 101 == 0 else i % 37,
        'ts': None if i % 17 == 0 else BASE + datetime.timedelta(hours=(i * 13) % 5000),
        's': f's{i % 211}',
    }


@pytest.fixture
def tbl() -> pxt.Table:
    # built once and kept across this module's tests (unless another module's test reset the database meanwhile)
    if 'scan_t' in pxt.list_tables():
        return pxt.get_table('scan_t')
    t = pxt.create_table(
        'scan_t',
        {'id': pxt.Int, 'a': pxt.Int | None, 'b': pxt.Float | None, 'g': pxt.Int | None, 'ts': pxt.Timestamp | None, 's': pxt.String},
    )
    for start in range(0, N, BATCH):
        t.insert(row(i) for i in range(start, start + BATCH))
    # new versions of scattered rows land in later segments; deleted rows disappear
    t.update({'a': t.a + 1}, where=(t.id % 251 == 3))
    t.update({'b': t.b * 2.0}, where=(t.id % 509 == 7))
    t.delete(where=(t.id % 307 == 5))
    return t


@pytest.fixture(autouse=True)
def reset_db() -> None:
    # (overrides the per-test reset of conftest: the module's table is shared)
    pass


def norm(v: Any) -> Any:
    """NaN compares equal to NaN."""
    if isinstance(v, float) and math.isnan(v):
        return 'nan'
    return v


def rows(rs: pxt.ResultSet) -> list[tuple]:
    return [tuple(norm(x) for x in r.values()) for r in rs]


def deleted(i: int) -> bool:
    return i % 307 == 5


class TestScanPaths:
    def test_scan_order_and_limits(self, tbl: pxt.Table) -> None:
        t = tbl
        ids = [i for i in range(N) if not deleted(i)]
        assert t.select(t.id).collect()['id'] == ids
        assert t.count() == len(ids)
        assert list(t.head(25)['id']) == ids[:25]
        assert list(t.tail(25)['id']) == ids[-25:]
        assert t.select(t.id).limit(30, offset=8190).collect()['id'] == ids[8190:8220]
        filtered = [i for i in ids if i % 5 == 1]
        assert t.where(t.id % 5 == 1).select(t.id).limit(40, offset=3000).collect()['id'] == filtered[3000:3040]
        assert t.where(t.id > N - 300).select(t.id).limit(1000).collect()['id'] == [i for i in ids if i > N - 300]

    @pytest.mark.parametrize(
        'pred',
        [
            lambda c: c['a'] > 17,
            lambda c: c['a'] == 100,
            lambda c: c['a'] != 100,
            lambda c: c['b'] <= 3.5,
            lambda c: c['b'] > -2,
            lambda c: c['a'] < 2.5,
            lambda c: (c['a'] >= -100) & (c['b'] < 10.0),
            lambda c: (c['a'] > 400) | (c['b'] < -20.0),
            lambda c: ~(c['g'] == 3),
            lambda c: c['b'] == None,  # noqa: E711
            lambda c: (c['a'] == None) | (c['g'] > 30),  # noqa: E711
            lambda c: c['ts'] <= BASE + datetime.timedelta(hours=2000),
            lambda c: c['ts'] > datetime.date(2024, 3, 1),
        ],
    )
    def test_filters(self, tbl: pxt.Table, pred: Any) -> None:
        t = tbl
        cols = {'a': t.a, 'b': t.b, 'g': t.g, 'ts': t.ts}
        generic = {'a': t.a + 0, 'b': t.b + 0.0, 'g': t.g + 0, 'ts': t.ts}
        fast = t.where(pred(cols)).select(t.id).collect()['id']
        slow = t.where(pred(generic)).select(t.id).collect()['id']
        assert fast == slow
        assert t.where(pred(cols)).count() == len(fast)

    def test_group_by(self, tbl: pxt.Table) -> None:
        t = tbl

        def q(g: Any, a: Any, b: Any) -> list[tuple]:
            return rows(
                t.group_by(g)
                .select(g, s=pxtf.sum(a), c=pxtf.count(a), m=pxtf.mean(b), mn=pxtf.min(a), mx=pxtf.max(b), sb=pxtf.sum(b))
                .collect()
            )

        assert q(t.g, t.a, t.b) == q(t.g + 0, t.a + 0, t.b + 0.0)
        whole = lambda a, b: rows(t.select(s=pxtf.sum(a), c=pxtf.count(b), m=pxtf.mean(b)).collect())  # noqa: E731
        assert whole(t.a, t.b) == whole(t.a + 0, t.b + 0.0)
        flt = lambda g, a: rows(t.where(t.id % 3 == 0).group_by(g).select(g, s=pxtf.sum(a)).collect())  # noqa: E731
        assert flt(t.g, t.a) == flt(t.g + 0, t.a + 0)

    @pytest.mark.parametrize('limit', [None, 7, 20_000])
    @pytest.mark.parametrize('asc', [True, False])
    def test_order_by(self, tbl: pxt.Table, limit: int | None, asc: bool) -> None:
        t = tbl

        def q(key: Any) -> list[tuple]:
            query = t.order_by(key, asc=asc).select(t.id, t.b)
            return rows((query.limit(limit) if limit else query).collect())

        assert q(t.b) == q(t.b + 0.0)
        assert q(t.a) == q(t.a + 0)

    def test_join(self, tbl: pxt.Table) -> None:
        t = tbl
        d = pxt.create_table('scan_d', {'g': pxt.Int | None, 'name': pxt.String}, if_exists='replace')
        d.insert([{'g': g, 'name': f'n{g}'} for g in range(0, 40, 3)] + [{'g': None, 'name': 'null'}, {'g': 6, 'name': 'six'}])
        for how in ['inner', 'left']:
            fast = rows(t.join(d, on=t.g == d.g, how=how).select(t.id, d.name).collect())
            slow = rows(t.join(d, on=(t.g + 0) == d.g, how=how).select(t.id, d.name).collect())
            assert fast == slow
        agg = lambda on: rows(t.join(d, on=on).group_by(d.name).select(d.name, c=pxtf.count(t.id)).collect())  # noqa: E731
        assert agg(t.g == d.g) == agg((t.g + 0) == d.g)

    def test_update_delete_selection(self) -> None:
        t = pxt.create_table('scan_u', {'id': pxt.Int, 'b': pxt.Float | None}, if_exists='replace')
        t.insert({'id': i, 'b': row(i)['b']} for i in range(N))
        # NaN sorts after every other float (Postgres' order), so it is greater than 20.0
        expected = {i for i in range(N) if (b := row(i)['b']) is not None and (math.isnan(b) or b > 20.0)}
        status = t.update({'id': t.id + N}, where=t.b > 20.0)
        assert status.num_rows == len(expected)
        assert set(t.where(t.id >= N).select(t.id).collect()['id']) == {i + N for i in expected}
        status = t.delete(where=(t.b < -20.0) | (t.b == None))  # noqa: E711
        gone = {i for i in range(N) if (b := row(i)['b']) is None or (not math.isnan(b) and b < -20.0)}
        assert status.num_rows == len(gone)
        assert t.count() == N - len(gone)

    def test_limit_stops_the_filter(self, tbl: pxt.Table) -> None:
        t = tbl
        # rows from id 30000 on overflow; a limit that the first rows satisfy never evaluates them (as SQL's LIMIT)
        boom = t.id // 30000 * 4611686018427387904 * 2
        assert t.where(boom >= 0).select(t.id).limit(5).collect()['id'] == [0, 1, 2, 3, 4]
        with pytest.raises(pxt.Error, match='bigint out of range'):
            t.where(boom >= 0).select(t.id).collect()
