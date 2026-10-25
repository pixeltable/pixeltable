"""Concurrent execution of UDF calls (`exec::dispatch`, `exec::prefetch`): results are placed by row, errors are
recorded per row, `on_error='abort'` reports the first failing row in row order, interrupts abort, `max_concurrency`
limits, async calls overlap, batches, resource pools."""

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable._testing
from pixeltable.func import Batch

LOCK = threading.Lock()
CALLS: list[tuple[int, int]] = []  # (argument, thread id) of recorded calls


def record(x: int) -> None:
    with LOCK:
        CALLS.append((x, threading.get_ident()))


@pytest.fixture(autouse=True)
def clear_calls() -> Iterator[None]:
    CALLS.clear()
    yield


@pxt.udf
def slow_inc(x: int) -> int:
    time.sleep(0.02)
    return x + 1


@pxt.udf
def slow_double(x: int) -> int:
    time.sleep(0.02)
    return x * 2


@pxt.udf
def sleep_200ms(x: int) -> int:
    time.sleep(0.2)
    return x


@pxt.udf
def recorded(x: int) -> int:
    record(x)
    time.sleep(0.002)
    return x


@pxt.udf
def fails_on_3_mod_7(x: int) -> int:
    time.sleep(0.002)
    if x % 7 == 3:
        raise ValueError(f'bad {x}')
    return x


@pxt.udf
def late_failure(x: int) -> int:
    # row 5 fails last in time, row 40 first: an abort reports row 5
    if x == 5:
        time.sleep(0.3)
        raise ValueError('bad 5')
    if x == 40:
        raise ValueError('bad 40')
    time.sleep(0.01)
    return x


@pxt.udf
def first_row_fails(x: int) -> int:
    if x == 0:
        raise ValueError('bad 0')
    record(x)
    time.sleep(0.02)
    return x


@pxt.udf(max_concurrency=1)
def serial_udf(x: int) -> int:
    record(x)
    time.sleep(0.001)
    return x


@pxt.udf
def contended(x: int) -> float:
    # numpy releases and retakes the GIL several times per call: on several threads, the calls wait for each other
    total = 0.0
    for i in range(8):
        total += float(np.random.default_rng(x * 8 + i).random(64).sum())
    record(x)
    return total


@pxt.udf
def interrupting(x: int) -> int:
    time.sleep(0.005)
    if x == 30:
        raise KeyboardInterrupt
    return x


IN_FLIGHT = [0, 0]  # current, max


async def _track(x: int) -> int:
    IN_FLIGHT[0] += 1
    IN_FLIGHT[1] = max(IN_FLIGHT[1], IN_FLIGHT[0])
    await asyncio.sleep(0.05)
    IN_FLIGHT[0] -= 1
    return x * 2


@pxt.udf
async def async_double(x: int) -> int:
    return await _track(x)


@pxt.udf(max_concurrency=3)
async def async_limited(x: int) -> int:
    return await _track(x)


@pxt.udf(batch_size=4)
def batched(xs: Batch[int]) -> Batch[int]:
    time.sleep(0.01)
    if 13 in xs:
        raise ValueError('bad batch')
    return [x * 10 for x in xs]


RATE_CALLS: list[float] = []


@pxt.udf(resource_pool='request-rate:pxtrs_test_rate')
async def rate_limited(x: int) -> int:
    RATE_CALLS.append(time.monotonic())
    return x


ATTEMPTS: dict[int, int] = {}


@pxt.udf(resource_pool='request-rate:pxtrs_test_retry')
async def flaky(x: int) -> int:
    ATTEMPTS[x] = ATTEMPTS.get(x, 0) + 1
    if ATTEMPTS[x] == 1:
        raise RuntimeError('rate limit exceeded')
    return x


@pxt.udf
def count_rows(n: int) -> int:
    t = pxt.get_table('nested_src')
    return len(t.where(t.x < n).select(y=slow_inc(t.x)).collect())


def int_table(name: str, n: int) -> pxt.Table:
    t = pxt.create_table(name, {'x': pxt.Int})
    t.insert({'x': i} for i in range(n))
    return t


class TestSyncDispatch:
    def test_insert_results_by_row(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=slow_inc(t.x))
        start = time.monotonic()
        t.insert({'x': i} for i in range(200))
        elapsed = time.monotonic() - start
        res = t.order_by(t.x).select(t.x, t.y).collect()
        assert res['y'] == [x + 1 for x in res['x']]
        # sequentially, 200 calls of 20 ms take 4 s
        assert elapsed < 2.0

    def test_select_keeps_row_order(self) -> None:
        t = int_table('t', 150)
        res = t.select(t.x, y=slow_inc(t.x)).collect()
        assert res['x'] == list(range(150))
        assert res['y'] == list(range(1, 151))

    def test_where_and_order_by(self) -> None:
        t = int_table('t', 60)
        res = t.where(slow_inc(t.x) % 3 == 0).order_by(slow_double(t.x), asc=False).select(t.x).collect()
        assert res['x'] == sorted((x for x in range(60) if (x + 1) % 3 == 0), reverse=True)

    def test_two_columns_run_concurrently(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=slow_inc(t.x))
        t.add_computed_column(z=slow_double(t.x))
        start = time.monotonic()
        t.insert({'x': i} for i in range(100))
        elapsed = time.monotonic() - start
        res = t.order_by(t.x).select(t.x, t.y, t.z).collect()
        assert res['y'] == [x + 1 for x in res['x']] and res['z'] == [2 * x for x in res['x']]
        assert elapsed < 1.5  # sequentially 4 s

    def test_dependent_columns(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=slow_inc(t.x))
        t.add_computed_column(z=slow_double(t.y), stored=False)
        t.add_computed_column(w=slow_inc(t.z))
        t.insert({'x': i} for i in range(50))
        res = t.order_by(t.x).select(t.x, t.w).collect()
        assert res['w'] == [2 * (x + 1) + 1 for x in res['x']]

    def test_unstored_column_is_called_once_per_row(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(u=recorded(t.x), stored=False)
        t.add_computed_column(v=slow_inc(t.u))
        CALLS.clear()
        t.insert({'x': i} for i in range(40))
        assert sorted(x for x, _ in CALLS) == list(range(40))
        assert t.order_by(t.x).select(t.v).collect()['v'] == list(range(1, 41))

    def test_update_with_udfs(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int, 'y': pxt.Int | None})
        t.add_computed_column(z=slow_double(t.y))
        t.insert({'x': i, 'y': None} for i in range(100))
        start = time.monotonic()
        status = t.update({'y': slow_inc(t.x)}, where=t.x % 2 == 1)
        elapsed = time.monotonic() - start
        assert status.num_rows == 50
        res = t.order_by(t.x).select(t.x, t.y, t.z).collect()
        assert res['y'] == [x + 1 if x % 2 == 1 else None for x in res['x']]
        assert res['z'] == [2 * (x + 1) if x % 2 == 1 else None for x in res['x']]
        # sequentially: 50 calls for the values and 50 for z, of 20 ms each
        assert elapsed < 1.0
        # the first failing row's exception, raised as is (as before calls were made ahead)
        with pytest.raises(ValueError, match='bad 5'):
            t.update({'y': late_failure(t.x)})
        assert t.where(t.y != None).count() == 50

    def test_errors_per_row(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=fails_on_3_mod_7(t.x))
        status = t.insert(({'x': i} for i in range(100)), on_error='ignore')
        assert status.num_excs == len([i for i in range(100) if i % 7 == 3])
        res = t.order_by(t.x).select(t.x, t.y, err=t.y.errormsg, et=t.y.errortype).collect()
        for x, y, err, et in zip(res['x'], res['y'], res['err'], res['et']):
            if x % 7 == 3:
                assert (y, err, et) == (None, f'bad {x}', 'ValueError')
            else:
                assert (y, err, et) == (x, None, None)

    def test_abort_reports_first_failing_row(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=late_failure(t.x))
        with pytest.raises(pxt.Error) as exc_info:
            t.insert(({'x': i} for i in range(60)), on_error='abort')
        assert 'bad 5' in str(exc_info.value)
        assert t.count() == 0

    def test_add_column_abort_reports_first_failing_row(self) -> None:
        t = int_table('t', 60)
        with pytest.raises(pxt.Error) as exc_info:
            t.add_computed_column(y=late_failure(t.x), on_error='abort')
        assert "Error while evaluating computed column 'y':\nbad 5" in str(exc_info.value)
        assert 'y' not in t.columns()

    def test_query_error_is_first_failing_row(self) -> None:
        t = int_table('t', 60)
        with pytest.raises(pxt.Error) as exc_info:
            t.select(late_failure(t.x)).collect()
        assert 'bad 5' in str(exc_info.value)

    def test_abort_cancels_later_rows(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=first_row_fails(t.x))
        with pytest.raises(pxt.Error, match='bad 0'):
            t.insert(({'x': i} for i in range(400)), on_error='abort')
        # only calls that had started when row 0 failed ran
        assert len(CALLS) < 100

    def test_keyboard_interrupt_aborts(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=interrupting(t.x))
        with pytest.raises(KeyboardInterrupt):
            t.insert(({'x': i} for i in range(100)), on_error='ignore')
        assert t.count() == 0
        t2 = int_table('t2', 100)
        with pytest.raises(KeyboardInterrupt):
            t2.add_computed_column(y=interrupting(t2.x), on_error='ignore')
        assert 'y' not in t2.columns()

    def test_sigint_aborts(self, tmp_path: Path) -> None:
        # a KeyboardInterrupt delivered to the main thread while it waits for the workers aborts the insert
        script = f"""
import os, sys
os.environ['PIXELTABLE_HOME'] = {str(tmp_path)!r}
os.environ['PIXELTABLE_VERBOSITY'] = '0'  # no console messages: the test reads stdout
sys.path.insert(0, {str(Path(__file__).parent)!r})
import pixeltable as pxt
import test_concurrency as tc
t = pxt.create_table('t', {{'x': pxt.Int}})
t.add_computed_column(y=tc.sleep_200ms(t.x))
print('ready', flush=True)
try:
    t.insert({{'x': i}} for i in range(400))
    print('finished', flush=True)
except KeyboardInterrupt:
    print('interrupted', t.count(), flush=True)
"""
        proc = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True, cwd=tmp_path)
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == 'ready'
        time.sleep(0.8)
        sent = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
        assert out.strip() == 'interrupted 0'
        assert time.monotonic() - sent < 3.0

    def test_max_concurrency_one_runs_in_order_in_caller(self) -> None:
        t = int_table('t', 50)
        CALLS.clear()
        assert t.select(y=serial_udf(t.x)).collect()['y'] == list(range(50))
        assert [x for x, _ in CALLS] == list(range(50))
        assert {tid for _, tid in CALLS} == {threading.get_ident()}

    def test_one_thread_runs_in_order_in_caller(self) -> None:
        t = int_table('t', 30)
        prev, _ = pixeltable._testing.configure_udf_dispatch(threads=1)
        try:
            CALLS.clear()
            assert t.select(y=recorded(t.x)).collect()['y'] == list(range(30))
            assert [x for x, _ in CALLS] == list(range(30))
            assert {tid for _, tid in CALLS} == {threading.get_ident()}
        finally:
            pixeltable._testing.configure_udf_dispatch(threads=prev)

    @pytest.mark.skipif(not getattr(sys, '_is_gil_enabled', lambda: True)(), reason='without a GIL the calls overlap')
    def test_calls_contending_for_the_gil_run_in_caller(self) -> None:
        # slow enough for workers, but slower on several threads than one after the other: the probes (the first one
        # and a few more after 64, 128, ... calls) send the calls back to the dispatching thread
        t = int_table('t', 4000)
        CALLS.clear()
        assert len(t.select(y=contended(t.x)).collect()) == 4000
        in_caller = sum(tid == threading.get_ident() for _, tid in CALLS)
        assert in_caller > 0.75 * len(CALLS)

    def test_nested_evaluation_in_udf(self) -> None:
        int_table('nested_src', 20)
        t = int_table('t', 8)
        assert t.select(n=count_rows(t.x)).collect()['n'] == list(range(8))

    def test_batched(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=batched(t.x))
        status = t.insert(({'x': i} for i in range(40)), on_error='ignore')
        # the batch [12, 13, 14, 15] fails, each of its rows records the error
        assert status.num_excs == 4
        res = t.order_by(t.x).select(t.x, t.y, err=t.y.errormsg).collect()
        for x, y, err in zip(res['x'], res['y'], res['err']):
            assert (y, err) == ((None, 'bad batch') if 12 <= x <= 15 else (10 * x, None))


class TestAsyncDispatch:
    def test_calls_overlap(self) -> None:
        IN_FLIGHT[:] = [0, 0]
        t = pxt.create_table('t', {'x': pxt.Int})
        t.add_computed_column(y=async_double(t.x))
        start = time.monotonic()
        t.insert({'x': i} for i in range(200))
        elapsed = time.monotonic() - start
        assert t.order_by(t.x).select(t.y).collect()['y'] == [2 * i for i in range(200)]
        assert elapsed < 2.0  # sequentially 10 s
        assert IN_FLIGHT[1] > 50

    def test_max_concurrency(self) -> None:
        IN_FLIGHT[:] = [0, 0]
        t = int_table('t', 20)
        assert t.select(y=async_limited(t.x)).collect()['y'] == [2 * i for i in range(20)]
        assert 2 <= IN_FLIGHT[1] <= 3

    def test_request_rate_pool(self) -> None:
        os.environ['PXTRS_TEST_RATE_RATE_LIMIT'] = '1200'  # 20 requests per second
        RATE_CALLS.clear()
        t = int_table('t', 6)
        assert t.select(y=rate_limited(t.x)).collect()['y'] == list(range(6))
        assert len(RATE_CALLS) == 6
        assert max(RATE_CALLS) - min(RATE_CALLS) >= 0.2

    def test_retry_in_resource_pool(self) -> None:
        ATTEMPTS.clear()
        t = int_table('t', 3)
        assert t.select(y=flaky(t.x)).collect()['y'] == [0, 1, 2]
        assert ATTEMPTS == {0: 2, 1: 2, 2: 2}
