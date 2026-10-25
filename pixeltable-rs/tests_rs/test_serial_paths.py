"""Paths that used to make their user-function calls one at a time or base row by base row: `Table.compute()` (calls of a
batch dispatched together, `exec::compute_rows`), component views (the rows of many base rows evaluated together,
`exec::propagate`), UDF arguments of many rows evaluated in parallel (`exec::prefetch`). Results, row order and the
error of `on_error='abort'` stay those of sequential evaluation."""

import threading
import time
from typing import Iterator, TypedDict

import pytest

import pixeltable as pxt

LOCK = threading.Lock()
CALLS: list[int] = []


@pytest.fixture(autouse=True)
def clear_calls() -> Iterator[None]:
    CALLS.clear()
    yield


@pxt.udf
def slow_square(x: int) -> int:
    with LOCK:
        CALLS.append(x)
    time.sleep(0.02)
    if x in (7, 150):
        raise ValueError(f'bad {x}')
    return x * x


@pxt.udf
def tag(s: str) -> str:
    with LOCK:
        CALLS.append(len(s))
    time.sleep(0.01)
    return f'<{s}>'


@pxt.udf
def record(x: int) -> int:
    with LOCK:
        CALLS.append(x)
    return x


class _Piece(TypedDict):
    piece: str


@pxt.iterator
def split3(text: str) -> Iterator[_Piece]:
    for i in range(3):
        yield {'piece': f'{text}/{i}'}


class TestSerialPaths:
    def test_compute_dispatches_together(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=slow_square(t.x))
        n = 300
        start = time.monotonic()
        res = t.compute([{'x': i} for i in range(n)], on_error='ignore')
        elapsed = time.monotonic() - start
        # sequentially the calls take 6 s
        assert elapsed < 3.0
        assert [r['y'] for r in res] == [None if i in (7, 150) else i * i for i in range(n)]
        assert sorted(CALLS) == list(range(n))
        assert t.count() == 0

    def test_compute_abort_is_first_failing_row(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=slow_square(t.x))
        with pytest.raises(pxt.Error, match='bad 7'):
            t.compute([{'x': i} for i in range(300)])
        # calls for rows after the failing one are not started once it failed
        assert len(CALLS) < 150

    def test_component_view_rows_evaluated_together(self) -> None:
        t = pxt.create_table('docs', {'text': pxt.String})
        v = pxt.create_view('pieces', t, iterator=split3(text=t.text))
        v.add_computed_column(tagged=tag(v.piece))
        n = 200
        start = time.monotonic()
        t.insert({'text': f'd{i}'} for i in range(n))
        elapsed = time.monotonic() - start
        # base row by base row, three calls at a time, the 600 calls would take 2 s
        assert elapsed < 1.5
        res = v.select(v.pos, v.tagged).collect()
        assert res['tagged'] == [f'<d{i}/{p}>' for i in range(n) for p in range(3)]
        assert res['pos'] == [p for _ in range(n) for p in range(3)]

    def test_parallel_arguments(self) -> None:
        n = 20_000
        t = pxt.create_table('t', {'x': pxt.Int, 's': pxt.String})
        t.insert({'x': i, 's': f'v{i % 7}'} for i in range(n))
        res = t.select(y=record(t.x * 3 - 1)).collect()
        assert res['y'] == [i * 3 - 1 for i in range(n)]
        CALLS.clear()
        # the argument (0 before row 15000) fails from row 15000 on: the error of that row, and no call for a later row
        with pytest.raises(pxt.Error, match='bigint out of range'):
            t.select(y=record(t.x // 15000 * 4611686018427387904 * 2)).collect()
        assert len(CALLS) <= 15000 and set(CALLS) <= {0}
