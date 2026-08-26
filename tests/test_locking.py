"""Tests for the table locking protocol (op-tbl-locking.md).

The point of these tests is to pin down the two blocking matrices of §1.2 and §1.3 -- all 16
(in-flight, arriving) cells per table kind -- and the freshness property of §1.4 that the whole design rests on.
Asserting that conflicting operations exclude each other is only half of it: a protocol that made everything block
would pass a conflict-only suite while destroying the MVCC property these matrices exist to protect.
"""

import enum
import itertools
import threading
from typing import Any, Callable

import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.store import StoreBase
from pixeltable.utils.fault_injection import FaultLocation
from tests.coordinator import MultiThreadedScenario
from tests.fault_injection import BlockFault
from tests.utils import pxt_raises, reload_catalog


def store_name(t: pxt.Table) -> str:
    """The store table backing t."""
    tvp = t._tbl_version_path
    return StoreBase.storage_name(tvp.tbl_id, is_view=tvp.is_view())  # type: ignore[arg-type]


def _own_connection() -> sql.Connection:
    """A connection outside any Pixeltable transaction, for probing pg_locks."""
    return Env.get().engine.connect()


def wait_until_blocked_on(store_tbl_name: str, timeout: float = 8.0) -> None:
    """Block until some backend is waiting for a lock on store_tbl_name; raise TimeoutError if none does.

    This is what makes "waits" a positive observation rather than a stall: without it a scenario asserting that an
    operation blocks would simply hang, and could not tell blocking apart from being slow.
    """
    q = sql.text(
        'SELECT count(*) FROM pg_locks '
        "WHERE locktype = 'relation' AND relation = CAST(:name AS regclass) AND NOT granted"
    )
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        with _own_connection() as conn:
            while not deadline.is_set():
                if conn.execute(q, {'name': store_tbl_name}).scalar() > 0:
                    return
                conn.rollback()
                deadline.wait(0.05)
        raise TimeoutError(f'no backend is waiting for a lock on {store_tbl_name}')
    finally:
        timer.cancel()


def held_lock_modes(store_tbl_name: str) -> set[str]:
    """The relation-lock modes currently held on store_tbl_name by any backend."""
    q = sql.text(
        'SELECT DISTINCT mode FROM pg_locks '
        "WHERE locktype = 'relation' AND relation = CAST(:name AS regclass) AND granted"
    )
    with _own_connection() as conn:
        return {r[0] for r in conn.execute(q, {'name': store_tbl_name})}


class BackgroundOp:
    """An operation running on its own thread, so that a scenario step can observe it blocking.

    A scenario step runs to completion before the next one starts, so an operation that is supposed to *wait*
    cannot be a step: it has to run alongside one.
    """

    done: threading.Event
    exc: BaseException | None

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.done = threading.Event()
        self.exc = None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except BaseException as e:
            self.exc = e
        finally:
            self.done.set()

    def join(self, timeout: float = 20.0) -> None:
        assert self.done.wait(timeout), 'background operation did not finish'
        if self.exc is not None:
            raise self.exc


class OpClass(enum.Enum):
    READ = 'read'
    WRITE = 'write'
    MD_UPDATE = 'md_update'
    FINALIZE = 'finalize'


class Outcome(enum.Enum):
    PROCEEDS = 'proceeds'
    WAITS = 'waits'
    FAILS_FAST = 'fails_fast'


_R, _W, _M, _F = OpClass.READ, OpClass.WRITE, OpClass.MD_UPDATE, OpClass.FINALIZE
_P, _WT, _FF = Outcome.PROCEEDS, Outcome.WAITS, Outcome.FAILS_FAST

# (in-flight, arriving) -> outcome. Mirrors §1.2 and §1.3 cell for cell. The FINALIZE column is the *helper*
# variant (NOWAIT); the owner variant is covered separately.
MATRIX: dict[bool, dict[tuple[OpClass, OpClass], Outcome]] = {
    # is_data_versioned=False -- §1.2
    False: {
        (_R, _R): _P,
        (_R, _W): _P,
        (_R, _M): _WT,
        (_R, _F): _FF,
        (_W, _R): _P,
        (_W, _W): _P,
        (_W, _M): _WT,
        (_W, _F): _FF,
        (_M, _R): _FF,
        (_M, _W): _FF,
        (_M, _M): _WT,
        (_M, _F): _FF,
        (_F, _R): _FF,
        (_F, _W): _FF,
        (_F, _M): _WT,
        (_F, _F): _FF,
    },
    # is_data_versioned=True -- §1.3; differs in exactly the five cells marked below
    True: {
        (_R, _R): _P,
        (_R, _W): _P,
        (_R, _M): _WT,
        (_R, _F): _FF,
        (_W, _R): _P,
        (_W, _W): _WT,  # <- differs
        (_W, _M): _WT,
        (_W, _F): _FF,
        (_M, _R): _WT,  # <- differs
        (_M, _W): _WT,  # <- differs
        (_M, _M): _WT,
        (_M, _F): _FF,
        (_F, _R): _WT,  # <- differs
        (_F, _W): _WT,  # <- differs
        (_F, _M): _WT,
        (_F, _F): _FF,
    },
}

# the lock each operation class takes, by table kind; §1.1's mode table expressed as a test
EXPECTED_MODES: dict[tuple[OpClass, bool], str] = {
    (_R, False): 'AccessShareLock',
    (_R, True): 'AccessShareLock',
    (_W, False): 'RowExclusiveLock',
    (_W, True): 'ExclusiveLock',
    (_M, False): 'AccessExclusiveLock',
    (_M, True): 'AccessExclusiveLock',
}


@pytest.mark.local('fault-injection/concurrency tests against the in-process catalog internals')
class TestLocking:
    @staticmethod
    def _make_tbl(name: str, is_data_versioned: bool) -> pxt.Table:
        t = pxt.create_table(name, {'a': pxt.Int | None}, _is_data_versioned=is_data_versioned)
        t.insert([{'a': 1}])
        return t

    @staticmethod
    def _op(op_class: OpClass, t: pxt.Table, tag: str) -> Callable[[], Any]:
        """The operation each class is exercised with. `tag` keeps concurrent md updates from colliding by name."""
        if op_class is OpClass.READ:
            return lambda: t.select(t.a).collect()
        if op_class is OpClass.WRITE:
            return lambda: t.insert([{'a': 2}])
        if op_class is OpClass.MD_UPDATE:
            return lambda: t.add_column(**{f'c_{tag}': pxt.Int | None})
        raise AssertionError(op_class)

    # ------------------------------------------------------------------ §8.2 the blocking matrix

    @pytest.mark.parametrize(
        'in_flight,arriving', list(itertools.product([_R, _W, _M], [_R, _W, _M])), ids=lambda c: c.value
    )
    def test_blocking_matrix(
        self, is_data_versioned: bool, in_flight: OpClass, arriving: OpClass, uses_db: None, fault_injection: None
    ) -> None:
        """One cell of §1.2/§1.3: what an arriving operation does against one already holding its locks.

        The FINALIZE row and column are covered separately; every other cell of both matrices runs here.
        """
        expected = MATRIX[is_data_versioned][in_flight, arriving]
        t = self._make_tbl('test', is_data_versioned)
        name = store_name(t)
        fault = BlockFault()
        op: list[BackgroundOp] = []

        def observe() -> None:
            if expected is Outcome.PROCEEDS:
                # completing while the holder is parked is exactly what "does not block" means
                op[0].join(timeout=10.0)
            elif expected is Outcome.FAILS_FAST:
                with pxt_raises(excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS):
                    op[0].join(timeout=10.0)
            else:
                # a positive observation of waiting, so a regression to PROCEEDS cannot pass
                wait_until_blocked_on(name)
                assert not op[0].done.is_set()

        def finish() -> None:
            if expected is Outcome.WAITS:
                op[0].join()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='hold', event=fault.reached, fn=self._op(in_flight, t, 'hold'))
            .then_run(
                thread_id=1, name='start arriving', fn=lambda: op.append(BackgroundOp(self._op(arriving, t, 'arr')))
            )
            .then_run(thread_id=1, name='observe', fn=observe)
            .then_unblock(thread_id=1, fault=fault)
            .then_run(thread_id=1, name='finish', fn=finish)
            .execute(timeout=60.0)
        )

        if (in_flight, arriving) == (_W, _W):
            # both writes must land, whether they shared the table (operational) or serialized (versioned)
            assert t.count() == 3

    # ------------------------------------------------------------------ §8.6 mode assertions

    @pytest.mark.parametrize('op_class', [OpClass.READ, OpClass.WRITE, OpClass.MD_UPDATE])
    def test_lock_modes(self, is_data_versioned: bool, op_class: OpClass, uses_db: None, fault_injection: None) -> None:
        """Each operation class holds exactly the mode §1.1 prescribes for the table kind."""
        t = self._make_tbl('test', is_data_versioned)
        name = store_name(t)
        observed: dict[str, set[str]] = {}
        fault = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='hold', event=fault.reached, fn=self._op(op_class, t, 'x'))
            .then_run(thread_id=1, name='probe', fn=lambda: observed.setdefault('modes', held_lock_modes(name)))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        expected = EXPECTED_MODES[op_class, is_data_versioned]
        assert expected in observed['modes'], (op_class, is_data_versioned, observed['modes'])

    def test_md_only_reads_take_no_locks(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """list/describe/get_table read catalog metadata, not rows: they neither lock nor wait (§4.1)."""
        t = self._make_tbl('test', is_data_versioned)
        fault = BlockFault()
        results: dict[str, Any] = {}

        def md_reads() -> None:
            results['tables'] = pxt.list_tables()
            results['tbl'] = pxt.get_table('test')
            results['ls'] = pxt.ls()

        (
            MultiThreadedScenario()
            # park a schema change with ACCESS EXCLUSIVE held
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='schema change', event=fault.reached, fn=lambda: t.add_column(b=pxt.Int | None)
            )
            .then_run(thread_id=1, name='md reads', fn=md_reads)
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert 'test' in results['tables']
        assert results['tbl'] is not None
        assert len(results['ls']) == 1

    # ------------------------------------------------------------------ §8.4 the freshness regression test

    def test_write_blocked_by_schema_change_sees_new_schema(self, uses_db: None, fault_injection: None) -> None:
        """A write that waited for a schema change must see the schema that landed while it waited.

        This is the whole reason for LOCK TABLE over advisory locks (§1.4). With a snapshot pinned before the wait,
        the insert would silently leave the new computed column NULL. Written for a versioned table, because that is
        where the arriving write waits rather than failing fast.
        """
        t = pxt.create_table('test', {'a': pxt.Int | None}, _is_data_versioned=True)
        fault = BlockFault()
        insert: list[BackgroundOp] = []

        (
            MultiThreadedScenario()
            # Thread 0: add a computed column, parked inside its transaction with ACCESS EXCLUSIVE held
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=t.a + 1)
            )
            # Thread 1: start an insert that must block on the lock, and confirm it is actually waiting
            .then_run(
                thread_id=1, name='start insert', fn=lambda: insert.append(BackgroundOp(lambda: t.insert([{'a': 1}])))
            )
            .then_run(thread_id=1, name='confirm blocked', fn=lambda: wait_until_blocked_on(store_name(t)))
            .then_unblock(thread_id=1, fault=fault)
            .then_run(thread_id=1, name='join insert', fn=lambda: insert[0].join())
            .execute(timeout=40.0)
        )

        # the row was inserted after the schema change, so the computed column must be populated
        assert t.select(t.a, t.b).collect()['b'] == [2]

    # ------------------------------------------------------------------ §8.3 pending-op finalization
    #
    # The FINALIZE row and column of §1.2/§1.3 are not separately observable with the fault points available: a
    # helper only reaches its NOWAIT finalize after acquiring its own locks, and by then the owner cannot be
    # holding ACCESS EXCLUSIVE. What a helper does when it finds the owner between transactions is covered by
    # test_catalog.py::test_drop_view_concurrent_insert; the owner variant is below.

    def test_owner_finalize_blocks(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """The owner variant of finalize blocks for the lock rather than failing (§4.4).

        A schema change whose md transaction is parked, and which then finalizes its own pending ops, has to come out
        the other side with the change applied -- it cannot hand its own work back.
        """
        t = self._make_tbl('test', is_data_versioned)
        fault = BlockFault()
        op: list[BackgroundOp] = []

        (
            MultiThreadedScenario()
            # a read holding ACCESS SHARE, so the owner's finalize has something to wait for
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='hold read', event=fault.reached, fn=lambda: t.select(t.a).collect())
            .then_run(
                thread_id=1,
                name='start schema change',
                fn=lambda: op.append(BackgroundOp(lambda: t.add_column(b=pxt.Int | None))),
            )
            .then_run(thread_id=1, name='confirm blocked', fn=lambda: wait_until_blocked_on(store_name(t)))
            .then_unblock(thread_id=1, fault=fault)
            .then_run(thread_id=1, name='join', fn=lambda: op[0].join())
            .execute(timeout=60.0)
        )

        assert 'b' in t.columns()

    # ------------------------------------------------------------------ §8.3 stale lock-set guesses

    def test_view_added_behind_the_cache(self, uses_db: None, fault_injection: None) -> None:
        """A view created after this thread cached the tree must still end up in the lock set (§2.1 step 4).

        The guess under-locks, so the validation that runs once the locks are held has to catch it and restart.
        """
        base = self._make_tbl('base', True)
        # warm this thread's cache on a base with no views
        base.insert([{'a': 2}])
        cat = get_runtime().catalog
        corrections_before = cat._num_lock_set_mismatches + cat._num_lock_set_warm_ups

        # create the view from another thread, which has its own catalog cache; this thread's cached tree goes stale
        BackgroundOp(lambda: pxt.create_view('v', base)).join()

        base.insert([{'a': 3}])

        # the write propagated, and the lock set was corrected on the way -- either the stale guess was caught by
        # validation, or the entry had been evicted and the warm-up supplied the current tree
        assert pxt.get_table('v').count() == 3
        assert cat._num_lock_set_mismatches + cat._num_lock_set_warm_ups > corrections_before

    def test_view_dropped_behind_the_cache(self, uses_db: None, fault_injection: None) -> None:
        """A view dropped after this thread cached the tree: LOCK TABLE hits a missing relation (§2.4).

        The write must recover as a stale guess -- succeeding, never surfacing UndefinedTable and never reporting
        that its own table was dropped.
        """
        base = self._make_tbl('base', True)
        view = pxt.create_view('v', base)
        base.insert([{'a': 2}])  # warms the tree, view included
        BackgroundOp(lambda: pxt.drop_table('v')).join()

        # succeeds, and never surfaces UndefinedTable or claims that base itself was dropped
        base.insert([{'a': 3}])

        assert base.count() == 3
        assert pxt.get_table('v', if_not_exists='ignore') is None
        del view

    def test_cold_cache_takes_the_right_mode(self, uses_db: None, fault_injection: None) -> None:
        """A first touch of a versioned table holds EXCLUSIVE, not ROW EXCLUSIVE, on its first attempt (§8.3).

        That is what distinguishes the warm-up of §2.3 from guessing a default mode and correcting it afterwards:
        after the warm-up the table's kind is known, so the wait policy and mode applied are the real ones.
        """
        t = self._make_tbl('test', True)
        name = store_name(t)
        reload_catalog()
        t = pxt.get_table('test')
        observed: dict[str, set[str]] = {}
        fault = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='insert', event=fault.reached, fn=lambda: t.insert([{'a': 2}]))
            .then_run(thread_id=1, name='probe', fn=lambda: observed.setdefault('modes', held_lock_modes(name)))
            .then_unblock(thread_id=1, fault=fault)
            .execute(timeout=30.0)
        )

        assert 'ExclusiveLock' in observed['modes'], observed['modes']
        assert 'RowExclusiveLock' not in observed['modes'], observed['modes']

    # ------------------------------------------------------------------ §8.3 no false sharing

    def test_no_false_sharing(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """A metadata update on table A must not block a read or a write on unrelated table B."""
        a = self._make_tbl('a', is_data_versioned)
        b = self._make_tbl('b', is_data_versioned)
        fault = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='md update on a', event=fault.reached, fn=lambda: a.add_column(c=pxt.Int | None)
            )
            .then_run(thread_id=1, name='read b', fn=lambda: b.select(b.a).collect())
            .then_run(thread_id=1, name='write b', fn=lambda: b.insert([{'a': 3}]))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert b.count() == 2

    def test_view_tree_is_locked(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """A write to a base with lock_mutable_tree=True excludes a metadata update on its mutable view."""
        if not is_data_versioned:
            pytest.skip('views of operational tables are not supported yet [PXT-1101]')
        base = self._make_tbl('base', is_data_versioned)
        view = pxt.create_view('v', base)
        unrelated = self._make_tbl('unrelated', is_data_versioned)
        fault = BlockFault()
        view_op: list[BackgroundOp] = []

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='write base', event=fault.reached, fn=lambda: base.insert([{'a': 5}]))
            # the view is part of the base write's lock set, so a schema change on it has to wait
            .then_run(
                thread_id=1,
                name='start view md update',
                fn=lambda: view_op.append(BackgroundOp(lambda: view.add_column(c=pxt.Int | None))),
            )
            .then_run(thread_id=1, name='confirm view locked', fn=lambda: wait_until_blocked_on(store_name(view)))
            # ... while an unrelated table is unaffected
            .then_run(thread_id=1, name='read unrelated', fn=lambda: unrelated.select(unrelated.a).collect())
            .then_unblock(thread_id=1, fault=fault)
            .then_run(thread_id=1, name='join view md update', fn=lambda: view_op[0].join())
            .execute(timeout=40.0)
        )

        assert base.count() == 2
