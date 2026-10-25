"""Concurrent writers (`exec::txn`): an operation that computes rows without the catalog lock commits them only if what
it read is unchanged, else it is computed again; operations of one process on the same tables wait for each other.
Each test forces the interleaving that lost data before (a UDF waits until the competing operation has had its
chance), so that the outcome does not depend on timing."""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import exceptions as excs

# how long a UDF waits for a competing operation that, with writer locks, cannot run until it is done
WAIT_S = 1.0

STARTED = threading.Event()
GO = threading.Event()
BARRIER = threading.Barrier(2)
CALLS = [0]


@pytest.fixture(autouse=True)
def reset_sync() -> None:
    global BARRIER
    STARTED.clear()
    GO.clear()
    BARRIER = threading.Barrier(2)
    CALLS[0] = 0


@pxt.udf
def meet_inc(x: int) -> int:
    # two updates meet here if they compute at the same time (both read the same value)
    try:
        BARRIER.wait(timeout=WAIT_S)
    except threading.BrokenBarrierError:
        pass
    return x + 1


@pxt.udf
def gated_inc(x: int) -> int:
    # signals that the operation is computing, then gives a competing operation the chance to commit
    STARTED.set()
    GO.wait(timeout=WAIT_S)
    return x + 1


@pxt.udf
def file_gated_inc(x: int) -> int:
    # signals that the operation is computing, then waits until another process has committed
    open(os.environ['GATE_READY'], 'w').close()
    while not os.path.exists(os.environ['GATE_GO']):
        time.sleep(0.01)
    return x + 1


@pxt.udf
def slow_inc(x: int) -> int:
    time.sleep(0.001)
    return x + 1


@pxt.udf
def appends_to_own_table(x: int) -> int:
    CALLS[0] += 1
    pxt.get_table('own').insert([{'x': 1000 + x}])
    return x


def run_in_thread(f) -> tuple[threading.Thread, dict]:
    out: dict = {}

    def target() -> None:
        try:
            out['result'] = f()
        except BaseException as e:  # noqa: BLE001
            out['error'] = e

    th = threading.Thread(target=target)
    th.start()
    return th, out


class TestThreads:
    def test_concurrent_updates_with_udf(self) -> None:
        # both updates read val=0; the second one to commit must not overwrite the first one's increment
        t = pxt.create_table('c', {'id': pxt.Int, 'val': pxt.Int}, primary_key='id')
        t.insert([{'id': 0, 'val': 0}])
        ths = [run_in_thread(lambda: t.update({'val': meet_inc(t.val)}, where=t.id == 0)) for _ in range(2)]
        for th, out in ths:
            th.join(30)
            assert 'error' not in out, out
        assert t.collect()['val'] == [2]

    def test_many_concurrent_updates(self) -> None:
        t = pxt.create_table('c', {'id': pxt.Int, 'val': pxt.Int}, primary_key='id')
        t.insert({'id': i, 'val': 0} for i in range(4))

        def work(udf: bool) -> None:
            for j in range(25):
                t.update({'val': slow_inc(t.val) if udf else t.val + 1}, where=t.id == j % 4)

        ths = [run_in_thread(lambda udf=(p % 2 == 0): work(udf)) for p in range(4)]
        for th, out in ths:
            th.join(120)
            assert 'error' not in out, out
        assert t.select(s=pxtf.sum(t.val)).collect()[0]['s'] == 100

    def test_update_does_not_resurrect_deleted_rows(self) -> None:
        t = pxt.create_table('d', {'id': pxt.Int, 'val': pxt.Int}, primary_key='id')
        t.insert({'id': i, 'val': 0} for i in range(20))
        upd, upd_out = run_in_thread(lambda: t.update({'val': gated_inc(t.val)}).num_rows)
        assert STARTED.wait(30)
        # with the update computing, the delete waits for it (before, it committed and the update brought the rows
        # back from its snapshot)
        dele, del_out = run_in_thread(lambda: t.delete(where=t.id < 10).num_rows)
        dele.join(WAIT_S / 4)
        GO.set()
        upd.join(30)
        dele.join(30)
        assert upd_out == {'result': 20} and del_out == {'result': 10}
        assert t.count() == 10
        assert t.order_by(t.id).collect()['id'] == list(range(10, 20))
        assert set(t.collect()['val']) == {1}

    def test_insert_during_add_computed_column(self) -> None:
        t = pxt.create_table('e', {'x': pxt.Int})
        t.insert({'x': i} for i in range(20))
        add, add_out = run_in_thread(lambda: t.add_computed_column(y=gated_inc(t.x)).num_rows)
        assert STARTED.wait(30)
        ins, ins_out = run_in_thread(lambda: t.insert({'x': i} for i in range(20, 30)).num_rows)
        ins.join(WAIT_S / 4)
        GO.set()
        add.join(30)
        ins.join(30)
        assert add_out == {'result': 20} and ins_out == {'result': 10}
        assert t.count() == 30
        assert t.where(t.y == None).count() == 0
        assert t.order_by(t.x).collect()['y'] == [x + 1 for x in range(30)]

    def test_udf_writing_its_own_table(self) -> None:
        # the column's computation keeps being invalidated by the inserts its own UDF makes: fail instead of looping
        t = pxt.create_table('own', {'x': pxt.Int})
        t.insert({'x': i} for i in range(3))
        with pytest.raises(excs.ConcurrencyError, match='changed by an operation started by the user code'):
            t.add_computed_column(y=appends_to_own_table(t.x))
        assert 'y' not in t.columns()
        assert CALLS[0] < 50  # a few attempts, not MAX_RETRIES


def run_python(code: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Runs `code` in a fresh interpreter on the same store as this process."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)], env=dict(os.environ), cwd=cwd, capture_output=True, text=True,
        timeout=timeout,
    )


def wait_for(path: Path, proc: subprocess.Popen | None = None, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f'the other process ended: {proc.communicate()[1][-2000:]}')
        assert time.monotonic() < deadline, f'{path} did not appear'
        time.sleep(0.01)


class TestProcesses:
    """Another process commits while this one computes: the commit finds that and the operation is computed again."""

    def gate(self, tmp_path: Path) -> tuple[Path, Path]:
        ready, go = tmp_path / 'ready', tmp_path / 'go'
        os.environ['GATE_READY'], os.environ['GATE_GO'] = str(ready), str(go)
        return ready, go

    def test_update_against_other_process(self, tmp_path: Path) -> None:
        ready, go = self.gate(tmp_path)
        t = pxt.create_table('c', {'id': pxt.Int, 'val': pxt.Int}, primary_key='id')
        t.insert({'id': i, 'val': 0} for i in range(10))
        try:
            proc = subprocess.Popen(
                [sys.executable, '-c', textwrap.dedent("""
                    import pixeltable as pxt
                    from test_writers import file_gated_inc
                    t = pxt.get_table('c')
                    print(t.update({'val': file_gated_inc(t.val)}).num_rows)
                    """)],
                env=dict(os.environ), cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            wait_for(ready, proc)
            t.delete(where=t.id < 5)
            t.update({'val': t.val + 10})
            go.touch()
            out, err = proc.communicate(timeout=120)
        finally:
            del os.environ['GATE_READY'], os.environ['GATE_GO']
        assert proc.returncode == 0, err[-2000:]
        assert out.strip() == '5'
        assert t.order_by(t.id).collect()['id'] == list(range(5, 10))
        assert set(t.collect()['val']) == {11}

    def test_add_computed_column_against_other_process(self, tmp_path: Path) -> None:
        ready, go = self.gate(tmp_path)
        t = pxt.create_table('e', {'x': pxt.Int})
        t.insert({'x': i} for i in range(10))

        def other_process() -> subprocess.CompletedProcess:
            wait_for(ready)
            r = run_python("""
                import pixeltable as pxt
                pxt.get_table('e').insert({'x': i} for i in range(10, 15))
                """)
            go.touch()
            return r

        th, out = run_in_thread(other_process)
        try:
            status = t.add_computed_column(y=file_gated_inc(t.x))
        finally:
            th.join(120)
            del os.environ['GATE_READY'], os.environ['GATE_GO']
        assert out['result'].returncode == 0, out['result'].stderr[-2000:]
        assert status.num_rows == 15
        assert t.count() == 15
        assert t.order_by(t.x).collect()['y'] == [x + 1 for x in range(15)]

    def test_insert_against_other_process(self, tmp_path: Path) -> None:
        # another process changes the schema while this one computes an insert: the insert is planned and built again
        ready, go = self.gate(tmp_path)
        t = pxt.create_table('e', {'x': pxt.Int})
        t.add_computed_column(y=file_gated_inc(t.x))

        def other_process() -> subprocess.CompletedProcess:
            wait_for(ready)
            r = run_python("""
                import pixeltable as pxt
                pxt.get_table('e').add_column(z=pxt.String | None)
                """)
            go.touch()
            return r

        th, out = run_in_thread(other_process)
        try:
            status = t.insert({'x': i} for i in range(5))
        finally:
            th.join(120)
            del os.environ['GATE_READY'], os.environ['GATE_GO']
        assert out['result'].returncode == 0, out['result'].stderr[-2000:]
        assert status.num_rows == 5
        assert t.columns() == ['x', 'y', 'z']
        assert t.order_by(t.x).collect()['y'] == [x + 1 for x in range(5)]
