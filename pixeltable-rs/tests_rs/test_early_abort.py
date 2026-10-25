"""A failing UDF call stops evaluation at its row: with `on_error='abort'` (and in queries), calls for later rows are
not started, and calls that evaluation still needs are made in row order in the thread that runs the operation, never
on the data-parallel threads (`exec::prefetch`, `exec::par`). Sizes cover both the sequential path and the parallel
one (`par::MIN_PARALLEL_ROWS` = 4,096 rows, `par::CHUNK_ROWS` = 16,384 rows per chunk)."""

import threading
from typing import Iterator

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

LOCK = threading.Lock()
CALLS: list[tuple[int, int]] = []  # (argument, thread id) of the calls after the failing one
IN_FLIGHT = [0, 0]  # current, max


@pytest.fixture(autouse=True)
def clear_calls() -> Iterator[None]:
    CALLS.clear()
    IN_FLIGHT[0] = IN_FLIGHT[1] = 0
    yield


def _record(x: int) -> None:
    with LOCK:
        CALLS.append((x, threading.get_ident()))


@pxt.udf
def fails_on_0(x: int) -> int:
    if x == 0:
        raise ValueError('bad 0')
    _record(x)
    return x


def _serial_fails_on_0(x: int) -> int:
    if x == 0:
        raise ValueError('bad 0')
    with LOCK:
        IN_FLIGHT[0] += 1
        IN_FLIGHT[1] = max(IN_FLIGHT[1], IN_FLIGHT[0])
    _record(x)
    with LOCK:
        IN_FLIGHT[0] -= 1
    return x


serial_fails_on_0 = pxt.udf(max_concurrency=1)(_serial_fails_on_0)


def int_table(name: str, n: int) -> pxt.Table:
    t = pxt.create_table(name, {'x': pxt.Int, 'g': pxt.Int})
    t.insert({'x': i, 'g': i % 10} for i in range(n))
    return t


def assert_no_calls() -> None:
    assert len(CALLS) == 0, f'{len(CALLS)} calls after the failing row'


class TestEarlyAbort:
    @pytest.mark.parametrize('n', [1_000, 5_000, 20_000])
    def test_select(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.select(y=fails_on_0(t.x)).collect()
        assert_no_calls()

    @pytest.mark.parametrize('n', [1_000, 20_000])
    def test_where(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.where(fails_on_0(t.x) > 3).collect()
        assert_no_calls()

    @pytest.mark.parametrize('n', [1_000, 20_000])
    def test_order_by(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.order_by(fails_on_0(t.x)).select(t.x).collect()
        assert_no_calls()
        with pytest.raises(pxt.Error, match='bad 0'):
            t.order_by(fails_on_0(t.x)).select(t.x).limit(5).collect()
        assert_no_calls()

    @pytest.mark.parametrize('n', [1_000, 40_000])
    def test_group_by(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.group_by(t.g).select(t.g, s=pxtf.sum(fails_on_0(t.x))).collect()
        assert_no_calls()

    @pytest.mark.parametrize('n', [1_000, 5_000])
    def test_insert(self, n: int) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=fails_on_0(t.x))
        with pytest.raises(pxt.Error, match='bad 0'):
            t.insert(({'x': i} for i in range(n)), on_error='abort')
        assert_no_calls()
        assert t.count() == 0

    @pytest.mark.parametrize('n', [1_000, 5_000])
    def test_add_computed_column(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.add_computed_column(y=fails_on_0(t.x), on_error='abort')
        assert_no_calls()
        assert 'y' not in t.columns()

    @pytest.mark.parametrize('n', [1_000, 5_000])
    def test_max_concurrency_one(self, n: int) -> None:
        t = int_table('t', n)
        with pytest.raises(pxt.Error, match='bad 0'):
            t.select(y=serial_fails_on_0(t.x)).collect()
        assert_no_calls()
        # without a failure, the calls run one at a time, in row order, in this thread
        res = t.where(t.x > 0).select(y=serial_fails_on_0(t.x)).collect()
        assert res['y'] == list(range(1, n))
        assert [x for x, _ in CALLS] == list(range(1, n))
        assert {tid for _, tid in CALLS} == {threading.get_ident()}
        assert IN_FLIGHT[1] == 1

    @pytest.mark.parametrize('n', [1_000, 20_000])
    def test_unconsulted_failure(self, n: int) -> None:
        # row 0's failure is not needed (`|` is decided by its first operand), so the row passes and evaluation goes on:
        # the later rows need the calls that were skipped ahead, and row evaluation makes them, in row order in this
        # thread
        t = int_table('t', n)
        res = t.where((t.x == 0) | (fails_on_0(t.x) > 0)).select(t.x).collect()
        assert res['x'] == list(range(n))
        assert [x for x, _ in CALLS] == list(range(1, n))
        assert {tid for _, tid in CALLS} == {threading.get_ident()}
