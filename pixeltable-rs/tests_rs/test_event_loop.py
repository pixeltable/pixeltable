"""The event loop of async UDFs (`pixeltable.func._runtime`, `exec::dispatch`): a coroutine's BaseException reaches the
caller instead of stopping the loop, a stopped loop is detected, nested evaluation from the loop thread does not
deadlock, and the loop is closed when the runtime is reset and at exit. Cases that used to hang run in a subprocess
with a timeout."""

import asyncio
import gc
import os
import subprocess
import sys
import textwrap
import time

import pytest

import pixeltable as pxt
import pixeltable._testing
from pixeltable.func import _runtime

TABLES = {'k2': 'k2', 'k3': 'k3'}


@pxt.udf
async def async_raises(x: int, what: str) -> int:
    await asyncio.sleep(0.001)
    if x == 5:
        if what == 'kbi':
            raise KeyboardInterrupt()
        if what == 'exit':
            raise SystemExit(3)
        if what == 'cancelled':
            raise asyncio.CancelledError()
    return x


@pxt.udf
async def async_plus1(x: int) -> int:
    await asyncio.sleep(0.001)
    return x + 1


@pxt.udf
async def stops_the_loop(x: int) -> int:
    if x == 0:
        # the loop's thread ends (as if the loop had died) while the dispatcher waits for this call
        _runtime._loop = None
        asyncio.get_running_loop().stop()
    await asyncio.sleep(0.01)
    return x


@pxt.udf
def sync_runs_async_query(x: int) -> int:
    time.sleep(0.001)
    t3 = pxt.get_table(TABLES['k3'])
    return sum(t3.where(t3.k < 3).select(y=async_plus1(t3.k)).collect()['y']) + x


@pxt.udf
async def async_runs_sync_query(x: int) -> int:
    t2 = pxt.get_table(TABLES['k2'])
    return sum(t2.select(y=sync_runs_async_query(t2.k)).collect()['y']) + x


def run_case(body: str, tmp_path, timeout: float = 60) -> list[str]:
    """Runs `body` in a fresh interpreter with its own store, the names of this module imported; returns its output
    lines (it fails if the process hangs or exits with an error)."""
    code = 'import asyncio\nimport pixeltable as pxt\nfrom test_event_loop import *\n' + textwrap.dedent(body)
    env = dict(os.environ, PIXELTABLE_HOME=str(tmp_path / 'home'))
    r = subprocess.run(
        [sys.executable, '-c', code], env=env, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True,
        text=True, timeout=timeout,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.strip().splitlines()


QUERY_THEN_RECOVER = """
    t = pxt.create_table('t', {'x': pxt.Int})
    t.insert({'x': i} for i in range(%d))
    try:
        t.select(y=async_raises(t.x, %r)).collect()
        print('no exception')
    except BaseException as e:
        print('raised', type(e).__name__, getattr(e, 'code', ''))
    # the loop still serves async UDFs
    print('after', t.order_by(t.x).select(y=async_plus1(t.x)).collect()['y'][:3])
"""


class TestBaseExceptions:
    @pytest.mark.parametrize('n', [100, 6], ids=['batch', 'few-rows'])
    def test_keyboard_interrupt(self, tmp_path, n: int) -> None:
        assert run_case(QUERY_THEN_RECOVER % (n, 'kbi'), tmp_path) == ['raised KeyboardInterrupt ', 'after [1, 2, 3]']

    def test_system_exit(self, tmp_path) -> None:
        assert run_case(QUERY_THEN_RECOVER % (100, 'exit'), tmp_path) == ['raised SystemExit 3', 'after [1, 2, 3]']

    def test_cancelled_error(self, tmp_path) -> None:
        # a CancelledError the UDF raises itself (not a cancellation of its task) reaches the caller
        assert run_case(QUERY_THEN_RECOVER % (100, 'cancelled'), tmp_path) == ['raised CancelledError ', 'after [1, 2, 3]']

    def test_single_coroutine(self, tmp_path) -> None:
        # a call made outside a batch (`_runtime.run_coroutine()`)
        out = run_case(
            """
            try:
                async_raises.exec([5, 'kbi'], {})
            except BaseException as e:
                print('raised', type(e).__name__)
            print(async_plus1.exec([1], {}))
            """,
            tmp_path,
        )
        assert out == ['raised KeyboardInterrupt', '2']

    def test_insert_is_not_committed(self, tmp_path) -> None:
        out = run_case(
            """
            t = pxt.create_table('t', {'x': pxt.Int})
            t.add_computed_column(y=async_raises(t.x, 'kbi'))
            try:
                t.insert(({'x': i} for i in range(20)), on_error='ignore')
            except BaseException as e:
                print('raised', type(e).__name__)
            print('rows', t.count())
            """,
            tmp_path,
        )
        assert out == ['raised KeyboardInterrupt', 'rows 0']


def test_stopped_loop_is_detected(tmp_path) -> None:
    out = run_case(
        """
        t = pxt.create_table('t', {'x': pxt.Int})
        t.insert({'x': i} for i in range(20))
        try:
            t.select(y=stops_the_loop(t.x)).collect()
            print('no exception')
        except pxt.Error as e:
            print('raised', type(e).__name__, 'stopped' in str(e))
        print('after', t.order_by(t.x).select(y=async_plus1(t.x)).collect()['y'][:3])
        """,
        tmp_path,
    )
    assert out == ['raised InternalError True', 'after [1, 2, 3]']


def test_nested_async_sync_async(tmp_path) -> None:
    # an async UDF runs a query whose sync UDF runs a query with an async UDF: the loop thread evaluates the nested
    # queries itself (it used to hand the sync calls to workers, whose async calls then waited for the busy loop)
    out = run_case(
        """
        k3 = pxt.create_table('k3', {'k': pxt.Int})
        k3.insert({'k': i} for i in range(10))
        k2 = pxt.create_table('k2', {'k': pxt.Int})
        k2.insert({'k': i} for i in range(20))
        t = pxt.create_table('t', {'x': pxt.Int})
        t.insert({'x': i} for i in range(4))
        print(t.order_by(t.x).select(y=async_runs_sync_query(t.x)).collect()['y'])
        """,
        tmp_path,
    )
    # sync_runs_async_query(k) = (1 + 2 + 3) + k; summed over k2: 6 * 20 + 190 = 310
    assert out == ['[310, 311, 312, 313]']


def open_loops() -> list[asyncio.AbstractEventLoop]:
    gc.collect()
    return [o for o in gc.get_objects() if isinstance(o, asyncio.AbstractEventLoop) and not o.is_closed()]


class TestLoopLifecycle:
    def test_closed_by_reset(self) -> None:
        t = pxt.create_table('t', {'x': pxt.Int})
        t.insert({'x': i} for i in range(10))
        assert t.select(y=async_plus1(t.x)).collect()['y'] == list(range(1, 11))
        assert len(open_loops()) >= 1
        pixeltable._testing.reload_catalog()
        assert open_loops() == []
        # a later call starts a new loop
        assert t.select(y=async_plus1(t.x)).collect()['y'] == list(range(1, 11))
        pixeltable._testing.reset_db()
        assert open_loops() == []

    def test_closed_at_exit(self, tmp_path) -> None:
        # an exit handler closes the loop (a leftover one would report "unclosed event loop" when collected)
        out = run_case(
            """
            import atexit, gc
            t = pxt.create_table('t', {'x': pxt.Int})
            t.insert({'x': i} for i in range(10))
            print(t.select(y=async_plus1(t.x)).collect()['y'][:3])
            atexit._run_exitfuncs()
            gc.collect()
            print('open loops', sum(1 for o in gc.get_objects() if isinstance(o, asyncio.AbstractEventLoop) and not o.is_closed()))
            """,
            tmp_path,
        )
        assert out == ['[1, 2, 3]', 'open loops 0']

    def test_busy_loop_is_not_closed(self) -> None:
        assert _runtime.run_coroutine(asyncio.sleep(0, result=7)) == 7
        fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5), _runtime.get_loop())
        # reset hooks close only an idle loop
        assert _runtime.shutdown(only_if_idle=True) is False
        fut.result(timeout=5)
        assert _runtime.shutdown(only_if_idle=True) is True
        assert open_loops() == []
