import contextlib
import enum
import itertools
import logging
from typing import Any, Callable

import psycopg
import pytest
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog.catalog import _store_tbl_name, _TblOpClass
from pixeltable.catalog.view import View
from pixeltable.env import Env, store_app_name
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation

from .coordinator import MultiThreadedScenario
from .fault_injection import BlockFault, ExceptionFault
from .utils import DatabaseRoot, pxt_raises, reload_catalog

_logger = logging.getLogger('pixeltable_test')


class TestCatalog:
    """Tests for miscellanous catalog functions."""

    def test_json_reserved_key(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        # JSON cell values are user data and may contain a key that collides with the proxy protocol's reserved
        # tag; inserting and reading such values back must round-trip rather than be rejected.
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('json_tbl'), {'id': pxt.Int | None, 'data': pxt.Json | None}, _is_data_versioned=is_data_versioned
        )
        rows = [
            {'id': 0, 'data': {'$pxt': 1}},  # collides at the top level
            {'id': 1, 'data': {'a': {'$pxt': [1, 2]}, 'b': 3}},  # collides while nested
            {'id': 2, 'data': {'$pxt': 'UUID', 'v': 'not-a-uuid'}},  # mimics a real type tag
            {'id': 3, 'data': {'ok': 1, 'nested': {'plain': True}}},  # no collision, unaffected
        ]
        t.insert(rows)
        result = t.order_by(t.id).select(t.data).collect()['data']
        assert result == [row['data'] for row in rows]

    def test_proxy_move_cross_db(self, init_env: None) -> None:
        # cross-catalog moves are rejected before any RPC (no daemon needed)
        with pytest.raises(excs.Error, match='same catalog'):
            pxt.move('pxt://local:db1/t', 'pxt://local:db2/t')
        with pytest.raises(excs.Error, match='same catalog'):
            pxt.move('pxt://local:db/t', 'local_t')  # hosted -> local

    def test_ls(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        pxt.create_dir(p('test_dir'))
        pxt.create_dir(p('test_dir/subdir'))

        tbl_name = p('test_dir/tbl')
        t = pxt.create_table(tbl_name, {'a': pxt.Int | None})
        t.insert(a=3)
        v1_name = p('view1')
        v1 = pxt.create_view(v1_name, t)
        t.insert(a=5)
        v1.add_column(b=pxt.Int | None)
        _s1 = pxt.create_snapshot(p('test_dir/snapshot1'), v1)
        t.insert(a=22)
        v2_name = p('test_dir/view2')
        v2 = pxt.create_view(v2_name, t)
        _s2 = pxt.create_snapshot(p('test_dir/snapshot2'), v2, additional_columns={'c': pxt.String | None})
        t.insert(a=4171780)
        df = pxt.ls(p('test_dir'))
        expected = f"""
            Name Kind Version Base
            snapshot1 snapshot {v1_name}:2
            snapshot2 snapshot {v2_name}:0
            subdir dir
            tbl table 4
            view2 view 1 {tbl_name}
        """

        def tokens(s: str) -> list[list[str]]:
            return [line.split() for line in s.splitlines() if line.split()]

        # compare contents, not repr(): pandas truncates one this wide. Empty cells (a dir has no version
        # or base) are dropped to match the expected tokens.
        actual = [list(df.columns), *([v for v in row if v != ''] for row in df.itertuples(index=False))]
        assert actual == tokens(expected)

    def test_cross_type_replacement(self, db_root: DatabaseRoot) -> None:
        """Test that tables, views, and snapshots can replace each other with if_exists='replace'.

        This tests the path collision handling logic: dirs can only collide with dirs,
        but all table subtypes (table, view, snapshot) can collide with each other.
        """
        p = db_root.make_catalog_path
        base_table = pxt.create_table(p('base'), {'c1': pxt.Int | None})

        # One lambda per create_x with expected columns
        creators = {
            'table': (lambda: pxt.create_table(p('target'), {'c2': pxt.String | None}, if_exists='replace'), ['c2']),
            'view': (
                lambda: pxt.create_view(
                    p('target'), base_table, additional_columns={'c3': pxt.String | None}, if_exists='replace'
                ),
                ['c3', 'c1'],
            ),
            'snapshot': (lambda: pxt.create_snapshot(p('target'), base_table, if_exists='replace'), ['c1']),
        }

        # Test all permutations: each table subtype can replace any table subtype
        for existing_creator, _ in creators.values():
            for replacing_creator, expected_cols in creators.values():
                existing_creator()
                assert p('target') in pxt.list_tables(p(''))
                result = replacing_creator()
                assert p('target') in pxt.list_tables(p(''))
                assert result.columns() == expected_cols

        # Verify cross-type replacement is blocked in both directions for every table subtype
        pxt.drop_table(p('target'))
        pxt.create_dir(p('target'))
        for creator, _ in creators.values():
            # dirs cannot be replaced by table subtypes
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a table, view or snapshot'):
                creator()
            # table subtypes cannot be replaced by dirs
            pxt.drop_dir(p('target'))
            creator()
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a directory'):
                pxt.create_dir(p('target'), if_exists='replace')
            pxt.drop_table(p('target'))
            pxt.create_dir(p('target'))

    def test_table_op_from_dict_needs_xact(self) -> None:
        """Verifies that a TableOp can be correctly deserialized from a dict that includes the legacy 'needs_xact'
        field"""
        from pixeltable.catalog.tbl_ops import CreateTableMdOp, TableOp

        # notice needs_xact that is no longer included in the output of to_dict
        # however, for backward compatibility it needs to continue to be accepted
        op = TableOp.from_dict(
            {
                'op_sn': 0,
                'status': 0,
                'tbl_id': 'b8037eea-404d-47c9-97fc-b4976bbb5466',
                'num_ops': 2,
                '_classname': 'CreateTableMdOp',
                'is_view': False,
                'needs_xact': True,
            }
        )
        assert isinstance(op, CreateTableMdOp)
        assert op.needs_xact  # now a ClassVar
        assert 'needs_xact' not in op.to_dict()

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_finalize_pending_ops_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int | None})
        exc = sql_exc.DBAPIError('', {}, orig=psycopg.errors.SerializationFailure())
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)
        t.add_column(b=pxt.Int | None)
        fault.assert_count(1)
        _ = t.select(t.b).collect()

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_finalize_pending_ops_non_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int | None})
        # Inject a non-retriable error into LoadViewOp. LoadViewOp is the last of 3 ops that constitute a view creation.
        # Upon catching the injected error, the catalog should abort view creation, and undo the first two ops that
        # were already executed.
        exc = Exception('injected')
        fault = ExceptionFault(exc, recurring=True)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_LOAD_VIEW_OP_EXEC, fault)

        with pxt_raises(code=excs.ErrorCode.INTERNAL_ERROR, match=str(exc)):
            _ = pxt.create_view('view', t.where(t.a > 0))
        fault.assert_count(1)

        # Check that view is not in catalog
        ls = pxt.ls()
        assert len(ls) == 1, ls
        assert ls['Name'].iloc[0] == 'test', ls

    @pytest.mark.db_roots('local', reason='recovers transparently when the server drops the pooled db connections')
    def test_dropped_connection(self, uses_db: None) -> None:
        if not Env.get().is_local:
            # the way this test drops connections (pg_terminate_backend on the pixeltable db) is specific to pgserver
            pytest.skip('requires pgserver')
        pxt.create_dir('d')
        t = pxt.create_table('d/t', {'a': pxt.Int | None})
        t.insert([{'a': 1}])

        def kill_connections() -> None:
            # Terminate this worker's backends out from under the pooled connections. The terminator runs on a
            # separate connection outside the engine's pool (and the statement excludes only its own backend),
            # so every connection the engine has pooled is killed.
            term_engine = sql.create_engine(Env.get().db_url, poolclass=sql.pool.NullPool)
            try:
                with term_engine.connect() as term:
                    killed = term.execute(
                        sql.text(
                            'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
                            'WHERE datname = current_database() AND application_name = :app'
                        ),
                        {'app': store_app_name()},
                    ).fetchall()
                    assert len(killed) > 0
                    term.commit()
            finally:
                term_engine.dispose()

        # each operation kind reconnects and succeeds instead of raising the dropped-connection error:
        # a catalog-metadata read, a data query, and a write
        kill_connections()
        assert 'd/t' in pxt.list_tables('d')

        kill_connections()
        assert t.count() == 1

        kill_connections()
        assert t.select(t.a).collect()['a'] == [1]

        kill_connections()
        t.insert([{'a': 2}])
        assert t.count() == 2

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_concurrent_add_column_insert(self, uses_db: None, fault_injection: None, is_data_versioned: bool) -> None:
        """Concurrent insert while add_column is blocked mid-finalize"""
        t = pxt.create_table('test', {'a': pxt.Int | None}, _is_data_versioned=is_data_versioned)
        fault = BlockFault()

        (
            MultiThreadedScenario()
            # Thread 0: arm the fault in pending table ops finalization
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=fault)
            # Thread 0: start adding a computed column, this will block at the fault point
            .then_run_until(
                thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=t.a + 1)
            )
            # Thread 1: run an insert concurrently with add column in thread 0 once thread 0 is in finalize pending ops
            # point
            .then_run(thread_id=1, name='insert', fn=lambda: t.insert([{'a': 1}]))
            # Unblock thread 0
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        # Both operations should have completed successfully.
        result = t.select(t.a, t.b).collect()
        assert len(result) == 1
        assert result[0] == {'a': 1, 'b': 2}

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_create_view_stale_base_tv_after_txn_failure(self, uses_db: None, fault_injection: None) -> None:
        """
        Verifies bug fix: due to an error in view creation, Catalog would fail to invalidate a modified but not
        persisted TableVersion. Later that would result in Pixeltable acting on that stale TableVersion, which can cause
        various sorts of issues including a data corruption.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})

        injected_exc = Exception('injected error')

        def create_view() -> None:
            with pytest.raises(Exception, match='injected'):
                pxt.create_view('va', base)

        (
            MultiThreadedScenario()
            # Thread 0: Warm up its catalog so base's tv is cached.
            .then_run(thread_id=0, name='warm up cache', fn=lambda: pxt.get_table('base'))
            # Thread 0: Arm a non-retriable exception fault inside create_view.
            .then_inject_fault(
                thread_id=0,
                loc=FaultLocation.CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED,
                fault=ExceptionFault(injected_exc),
            )
            # Thread 0: Run create_view (va) that fails. Before the fix, base_tv was not added to _modified_tvs, so it
            # stays in cache with stale in-memory state, i.e. with view_sn=v+1
            .then_run(thread_id=0, name='create view that fails', fn=create_view)
            # Thread 1: Create view vb on the same base. This also advances the persisted view_sn to v+1, which
            # automatically matches thread 0's stale cached value.
            .then_run(thread_id=1, name='create view', fn=lambda: pxt.create_view('vb', base))
            # Thread 0: insert into base table. Before the fix, Catalog would observe that the cached TableVersion's
            # version and view_sn match the stored values, and based on that decide to skip reloading the table.
            # The outcome of that is the write is not propagated to vb.
            .then_run(thread_id=0, name='insert into base (stale cache)', fn=lambda: base.insert([{'a': 42}]))
            .execute()
        )

        assert base.count() == 1
        # Verify that the insert was propagated to vb.
        assert pxt.get_table('vb').count() == 1

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_drop_view_concurrent_insert(self, uses_db: None, fault_injection: None) -> None:
        """
        Start with a base table and a view. Thread 0 begins to drop the view, but pauses inside finalize pending ops
        (without the exclusive lock). Thread 1 swoops in in the meantime to insert a row into the base table, and
        finalizes view drop as a side effect. Before the fix, this would result in the insert failing with "table not
        found" error.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        _ = pxt.create_view('v', base)
        block_in_finalize = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=block_in_finalize
            )
            # Thread 0: drop v but block mid-finalize
            .then_run_until(
                thread_id=0, name='drop view', event=block_in_finalize.reached, fn=lambda: pxt.drop_table('v')
            )
            # Thread 1: insert into base
            .then_run(thread_id=1, name='insert into base', fn=lambda: base.insert([{'a': 1}]))
            .then_unblock(thread_id=1, fault=block_in_finalize)
            .execute()
        )

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None


class Outcome(enum.Enum):
    """What an arriving operation does when it meets one already in flight."""

    PROCEEDS = 'proceeds'
    WAITS = 'waits'
    FAILS_FAST = 'fails_fast'


# abbreviations for the two blocking matrices below, which are wide enough without the enum name on every entry
_READ, _WRITE, _MD_UPDATE, _FINALIZE = (
    _TblOpClass.DATA_READ,
    _TblOpClass.DATA_WRITE,
    _TblOpClass.MD_WRITE,
    _TblOpClass.FINALIZE,
)
_PROCEEDS, _WAITS, _FAILS_FAST = Outcome.PROCEEDS, Outcome.WAITS, Outcome.FAILS_FAST

# (in-flight, arriving) -> outcome, on an operational table. There is no arriving-FINALIZE column: that is not a
# cell a scenario can construct, since an arriving FINALIZE is not an operation a test issues but a detour some
# other operation takes after running into pending ops, and it can only do that once it holds its own lock. Where
# the in-flight operation excludes that acquisition, the arriving one never reaches the finalization. A finalization
# always waits, which the FINALIZE rows assert from the other side and
# test_md_only_read_waits_for_a_finalize asserts directly.
OP_MATRIX: dict[tuple[_TblOpClass, _TblOpClass], Outcome] = {
    (_READ, _READ): _PROCEEDS,
    (_READ, _WRITE): _PROCEEDS,
    (_READ, _MD_UPDATE): _WAITS,
    (_WRITE, _READ): _PROCEEDS,
    (_WRITE, _WRITE): _PROCEEDS,
    (_WRITE, _MD_UPDATE): _WAITS,
    (_MD_UPDATE, _READ): _FAILS_FAST,
    (_MD_UPDATE, _WRITE): _FAILS_FAST,
    (_MD_UPDATE, _MD_UPDATE): _WAITS,
    (_FINALIZE, _READ): _FAILS_FAST,
    (_FINALIZE, _WRITE): _FAILS_FAST,
    (_FINALIZE, _MD_UPDATE): _WAITS,
}

# the same, on a data-versioned table. The five cells marked below are the whole difference: writers serialize
# instead of sharing the table, and an arriving read or write waits for a schema change instead of being refused.
VERSIONED_MATRIX: dict[tuple[_TblOpClass, _TblOpClass], Outcome] = {
    (_READ, _READ): _PROCEEDS,
    (_READ, _WRITE): _PROCEEDS,
    (_READ, _MD_UPDATE): _WAITS,
    (_WRITE, _READ): _PROCEEDS,
    (_WRITE, _WRITE): _WAITS,  # differs
    (_WRITE, _MD_UPDATE): _WAITS,
    (_MD_UPDATE, _READ): _WAITS,  # differs
    (_MD_UPDATE, _WRITE): _WAITS,  # differs
    (_MD_UPDATE, _MD_UPDATE): _WAITS,
    (_FINALIZE, _READ): _WAITS,  # differs
    (_FINALIZE, _WRITE): _WAITS,  # differs
    (_FINALIZE, _MD_UPDATE): _WAITS,
}

# MD_READ is left out of both roles: it takes no locks, so it can neither hold one nor be excluded by one, and the
# fault point the holder parks at fires only for a transaction that locked something. That it proceeds regardless is
# what test_md_only_reads_never_block asserts.
_LOCKING_CLASSES = tuple(c for c in _TblOpClass if c is not _TblOpClass.MD_READ)

assert (
    OP_MATRIX.keys()
    == VERSIONED_MATRIX.keys()
    == {
        (in_flight, arriving)
        for in_flight, arriving in itertools.product(_LOCKING_CLASSES, (_READ, _WRITE, _MD_UPDATE))
    }
), 'each matrix must cover every in-flight class against every arriving class it is written for'


@pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
class TestLocking:
    """Deterministic interleavings of the table-locking protocol (docs/design/catalog-locking.md).

    Each scenario parks one transaction at a fault point with its locks held, and asserts what a second
    operation does when it arrives.

    Scenarios that name a view are not parameterized over the table kind: create_view() requires a
    data-versioned base (PXT-1101), so an operational tree has a single node.
    """

    @staticmethod
    def store_tbl_name(t: pxt.Table) -> str:
        """The name of the store table backing `t`."""
        return _store_tbl_name(t._id, is_view=isinstance(t, View))

    @staticmethod
    def is_blocked_on(store_tbl_name: str) -> bool:
        """Whether some backend is waiting for a lock on store_tbl_name."""
        q = sql.text(
            'SELECT count(*) FROM pg_locks '
            "WHERE locktype = 'relation' AND relation = CAST(:name AS regclass) AND NOT granted"
        )
        with Env.get().engine.connect() as conn:
            return conn.execute(q, {'name': store_tbl_name}).scalar() > 0

    @staticmethod
    def make_tbl(name: str = 'tbl', *, is_data_versioned: bool) -> pxt.Table:
        t = pxt.create_table(name, {'a': pxt.Int | None}, _is_data_versioned=is_data_versioned)
        t.insert([{'a': 1}])
        return t

    @staticmethod
    def op_fn(op_class: _TblOpClass, t: pxt.Table, tag: str) -> Callable[[], Any]:
        """An operation of `op_class` against `t`. tag disambiguates the column an MD_UPDATE adds.

        The query and the column expression are built here rather than inside the returned callable. Building either
        reads metadata in a transaction of its own, and that transaction is not the one under test: it would be
        refused by whatever the scenario has queued on the table, which is correct protocol behavior but not what the
        cell measures.
        """
        if op_class is _TblOpClass.DATA_READ:
            return t.select().collect
        if op_class is _TblOpClass.DATA_WRITE:
            return lambda: t.insert([{'a': 2}])
        if op_class is _TblOpClass.MD_WRITE:
            value_expr = t.a + 1
            return lambda: t.add_computed_column(**{f'c_{tag}': value_expr})
        # a FINALIZE is only reachable through a schema change that leaves pending ops, and add_computed_column() is
        # not one: it runs its ALTER TABLE inside the md transaction. add_column() rolls its store DDL forward as a
        # pending op, so its second transaction is the finalization the fault filter picks out.
        assert op_class is _TblOpClass.FINALIZE
        return lambda: t.add_column(**{f'c_{tag}': pxt.Int | None})

    @staticmethod
    def assert_fails_fast(fn: Callable[[], Any]) -> None:
        """The operation must report a schema change in progress rather than waiting for the lock.

        Nothing here asserts that retry_loop() did not retry, because it cannot silently do so: _MAX_RETRIES is -1,
        so a retried operation never returns and the scenario fails on its timeout instead.
        """
        with pxt_raises(excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS):
            fn()

    @pytest.mark.parametrize('in_flight,arriving', list(OP_MATRIX))
    def test_blocking_matrix(
        self,
        is_data_versioned: bool,
        in_flight: _TblOpClass,
        arriving: _TblOpClass,
        uses_db: None,
        fault_injection: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """What an arriving operation does while another is in flight, for every pair of operation classes.

        The point is not only that conflicting classes exclude each other, but that non-conflicting ones do not:
        an implementation that made everything block would pass a conflict-only suite.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        outcome = (VERSIONED_MATRIX if is_data_versioned else OP_MATRIX)[in_flight, arriving]
        # the holder's transaction is parked with its locks held; the filter picks which of its transactions,
        # since a schema change runs an MD_UPDATE and then one FINALIZE per pending op
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is in_flight)
        holder = self.op_fn(in_flight, t, 'holder')
        arriving_fn = self.op_fn(arriving, t, 'arriving')
        arriving_result: list[Any] = []

        def arriver() -> None:
            arriving_result.append(arriving_fn())

        scenario = (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name=f'in-flight {in_flight.value}', event=fault.reached, fn=holder)
        )
        if outcome is Outcome.WAITS:
            (
                scenario.then_run_until(
                    thread_id=1,
                    name=f'arriving {arriving.value} waits',
                    poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(t)),
                    fn=arriver,
                ).then_unblock(thread_id=2, fault=fault)
            )
        else:
            fn = arriver if outcome is Outcome.PROCEEDS else (lambda: self.assert_fails_fast(arriver))
            scenario.then_run(thread_id=1, name=f'arriving {arriving.value}', fn=fn).then_unblock(
                thread_id=1, fault=fault
            )
        # the versioned (write, write) cell must be wait-then-succeed, not the fail-then-redo it replaces,
        # so the second write must not have been retried. Capturing is confined to that cell: at DEBUG the catalog
        # also runs its consistency check on every transaction, which the other 23 cells have no use for.
        checks_retries = is_data_versioned and in_flight is arriving is _TblOpClass.DATA_WRITE
        with contextlib.ExitStack() as stack:
            if checks_retries:
                stack.enter_context(caplog.at_level(logging.DEBUG, logger='pixeltable.catalog.catalog'))
            scenario.execute()
        if checks_retries:
            retries = [r.message for r in caplog.records if 'Retriable error' in r.message or 'Retrying (' in r.message]
            assert retries == [], retries

        # Every write that ran to completion added a row to the one make_tbl() inserted, and one that was
        # refused added none -- so the row count says which operations actually did their work, where the outcome
        # alone only says how they were scheduled. A write that waited still lands its rows, which is why the test
        # below is "not refused" rather than "proceeded": on a data-versioned table an arriving write waits.
        # In the (DATA_WRITE, DATA_WRITE) cell operational writers share the table, so both sets of rows land.
        expected_rows = 1
        if in_flight is _TblOpClass.DATA_WRITE:
            expected_rows += 1
        if arriving is _TblOpClass.DATA_WRITE and outcome is not Outcome.FAILS_FAST:
            expected_rows += 1
        assert t.count() == expected_rows

        # an operation that waited for the schema change must have run against the schema that landed while
        # it waited, which the outcome alone does not show. Only the versioned cells assert it: on an operational
        # table both are refused rather than made to wait.
        if in_flight is _TblOpClass.MD_WRITE and is_data_versioned:
            if arriving is _TblOpClass.DATA_WRITE:
                # re-fetch: this thread's handle predates the schema change
                t2 = pxt.get_table('tbl')
                assert t2.where(t2.a == 2).select(t2.c_holder).collect()['c_holder'] == [3]
            if arriving is _TblOpClass.DATA_READ:
                assert arriving_result[0].schema.keys() == {'a', 'c_holder'}, arriving_result[0].schema

    def test_schema_change_blocked_by_schema_change_sees_new_schema(
        self, is_data_versioned: bool, uses_db: None, fault_injection: None
    ) -> None:
        """An operation that waited for a lock must see the schema that landed while it waited.

        This is the property that makes LOCK TABLE the mechanism rather than an advisory lock: the
        statement is snapshot-exempt, so the waiter's snapshot is pinned when the lock is granted, not before.
        With a snapshot pinned before the wait, the second column below would be computed from a schema that no
        longer exists.

        A schema change is the arriving operation here because that is the class that waits on both kinds; the
        test below is the data-versioned insert form, which also exercises the wait policy.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)
        # both expressions are built here, outside the operations that use them: see self.op_fn()
        b_expr, c_expr = t.a + 1, t.a + 100

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add column b', event=fault.reached, fn=lambda: t.add_computed_column(b=b_expr)
            )
            .then_run_until(
                thread_id=1,
                name='add column c, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(t)),
                fn=lambda: t.add_computed_column(c=c_expr),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        # the waiter read the metadata that thread 0 committed, so both columns exist and both are populated.
        # Re-fetch: this thread's handle predates both schema changes.
        t2 = pxt.get_table('tbl')
        res = t2.select(t2.a, t2.b, t2.c).collect()
        assert len(res) == 1 and res[0]['b'] == 2 and res[0]['c'] == 101, res

    def test_insert_blocked_by_schema_change_sees_new_schema(self, uses_db: None, fault_injection: None) -> None:
        """A data-versioned insert that waited for a schema change must run against the new schema.

        The versioned form of the test above, and the one that also exercises the wait policy: an insert waits
        only on a data-versioned table. With a snapshot pinned before the wait, the insert would compute its rows
        against the pre-change schema and silently leave the new computed column NULL.
        """
        t = pxt.create_table('tbl', {'a': pxt.Int | None}, _is_data_versioned=True)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)
        b_expr = t.a + 1

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add computed column', event=fault.reached, fn=lambda: t.add_computed_column(b=b_expr)
            )
            .then_run_until(
                thread_id=1,
                name='insert, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(t)),
                fn=lambda: t.insert([{'a': 1}]),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        # the row was inserted after the schema change, so the computed column must be populated
        t2 = pxt.get_table('tbl')
        res = t2.select(t2.a, t2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_no_false_sharing(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """A schema change on one table must not exclude operations on an unrelated one.

        This is what catches an over-broad lock set; the matrix above cannot, since it only ever names one table.
        """
        a = self.make_tbl('a', is_data_versioned=is_data_versioned)
        b = self.make_tbl('b', is_data_versioned=is_data_versioned)
        x_expr = a.a + 1
        read_b = b.select(b.a).collect
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add column to a', event=fault.reached, fn=lambda: a.add_computed_column(x=x_expr)
            )
            .then_run(thread_id=1, name='read b', fn=read_b)
            .then_run(thread_id=1, name='write b', fn=lambda: b.insert([{'a': 2}]))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )
        assert b.count() == 2

    def test_md_only_reads_never_block(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """Listing and describing a catalog take no locks, so they work during a schema change.

        These read catalog metadata and no table data, so they may observe a schema that is about to change
        rather than failing or blocking.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        b_expr = t.a + 1
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)

        def md_only_reads() -> None:
            assert 'tbl' in pxt.list_tables()
            assert pxt.get_table('tbl') is not None
            assert 'tbl' in pxt.get_dir_contents()['tables']

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=b_expr)
            )
            .then_run(thread_id=1, name='metadata-only reads', fn=md_only_reads)
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

    def test_md_only_read_waits_for_a_finalize(
        self, is_data_versioned: bool, uses_db: None, fault_injection: None
    ) -> None:
        """A finalization waits for its lock, so an operation that runs into pending ops is never refused.

        A finalization that failed fast would refuse the read below over a conflict that is not its own: the read
        takes no locks at all. The refusal would come from the detour, which asks for ACCESS EXCLUSIVE and so can
        be refused by anything, a plain reader included. tool/random_ops.py hits this within seconds of starting,
        as a read of a table another worker is mid-change on.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        block_on_finalize = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.FINALIZE)
        store_tbl = self.store_tbl_name(t)
        result: list[pxt.Table] = []

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=block_on_finalize)
            .then_run_until(
                thread_id=0,
                name='add_column, parked inside its finalization with the lock held',
                event=block_on_finalize.reached,
                fn=lambda: t.add_column(b=pxt.Int | None),
            )
            .then_run_until(
                thread_id=1,
                name='metadata-only read, which waits',
                poll_condition=lambda: self.is_blocked_on(store_tbl),
                fn=lambda: result.append(pxt.get_table('tbl')),
            )
            .then_unblock(thread_id=2, fault=block_on_finalize)
            .execute()
        )

        assert len(result) == 1
        assert 'b' in result[0].columns()

    def test_base_write_excludes_view_schema_change(self, uses_db: None, fault_injection: None) -> None:
        """A write to a base locks its mutable tree, so it excludes a schema change on a view of it.

        The write propagates to the view, so the view's store table is part of the write's lock set even though
        the caller never named it. The unrelated view in the same scenario is what keeps the set from being read as
        "lock everything": it must not be excluded.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        v = pxt.create_view('v', base)
        other_base = pxt.create_table('other_base', {'a': pxt.Int | None})
        other_v = pxt.create_view('other_v', other_base)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.DATA_WRITE)
        b_expr, other_expr = v.a + 1, other_v.a + 1

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='insert into base', event=fault.reached, fn=lambda: base.insert([{'a': 1}])
            )
            .then_run(
                thread_id=1,
                name='schema change on an unrelated view',
                fn=lambda: other_v.add_computed_column(c=other_expr),
            )
            .then_run_until(
                thread_id=1,
                name='schema change on the view, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(v)),
                fn=lambda: v.add_computed_column(b=b_expr),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        v2 = pxt.get_table('v')
        res = v2.select(v2.a, v2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_create_view_excluded_by_base_write(self, uses_db: None, fault_injection: None) -> None:
        """create_view locks the base's store table, so it waits for a write in flight there.

        Adding a mutable view changes how writes to the base propagate, which is why the base is in the lock set
        of an operation that only names the view. The row written while create_view waited must be in the view.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.DATA_WRITE)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='insert into base', event=fault.reached, fn=lambda: base.insert([{'a': 1}])
            )
            .then_run_until(
                thread_id=1,
                name='create_view, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(base)),
                fn=lambda: pxt.create_view('v', base),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        assert pxt.get_table('v').count() == 1

    def test_stale_guess_view_added(self, uses_db: None, fault_injection: None) -> None:
        """A base write whose lock set predates a new view must restart and lock the view too.

        The window the fault opens is the one a guess is exposed to: the set has been resolved and the locks are
        not yet held, so a view created now is in the tree the write propagates to but not in what it locked.
        Warming a thread's cache and then creating the view is not enough to reach that window -- every operation
        opens with a metadata read of its own, which discovers the view before the lock set is computed.

        The corrected set is observed rather than counted: propagating to the view means the write ran against
        metadata that has it, and doing so without the view's store table locked trips assert_rows_write_locked()
        in the store, so a write that skipped the restart fails here rather than passing quietly.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.DATA_WRITE)

        def create_view() -> None:
            v = pxt.create_view('v', base)
            v.add_computed_column(b=v.a + 1)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_BEFORE_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0,
                name='write to the base, parked before it locks',
                event=fault.reached,
                fn=lambda: base.insert([{'a': 1}]),
            )
            .then_run(thread_id=1, name='create a view with a computed column', fn=create_view)
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        # the row reached the view, and its computed column was evaluated -- both only possible against metadata
        # that names the view, which is what the restart went back for
        v2 = pxt.get_table('v')
        res = v2.select(v2.a, v2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_stale_guess_view_dropped(self, uses_db: None, fault_injection: None) -> None:
        """A base write whose lock set names a dropped view must recover, not surface UndefinedTable.

        The guess is wrong in the other direction here, and it is the lock statement itself that fails, because a
        store table it names is gone. That is a stale guess like any other: the write's own table is still there,
        so the set is rebuilt from the store and the write retried, rather than reported as a dropped table.
        Without the StaleLockSetError recovery this test fails with a raw UndefinedTable.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        pxt.create_view('v', base)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.DATA_WRITE)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_BEFORE_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0,
                name='write to the base, parked before it locks',
                event=fault.reached,
                fn=lambda: base.insert([{'a': 1}]),
            )
            .then_run(thread_id=1, name='drop the view', fn=lambda: pxt.drop_table('v'))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None

    def test_cold_cache_write_takes_versioned_mode(self, uses_db: None, fault_injection: None) -> None:
        """A first-touch write on a data-versioned table takes the versioned mode on its first attempt.

        The kind selects the mode, and a cold cache does not know it, which is why the lock set is read from the
        store. Both writers are cold here, because that is what separates the two modes: EXCLUSIVE conflicts with
        itself, so the second write is excluded, where two ROW EXCLUSIVE writers would share the table and both
        proceed. A default guessed and then corrected would show as the second write waiting anyway, so each
        thread does its first touch of the table inside the scenario, with no attempt of its own to correct.
        """
        t = self.make_tbl(is_data_versioned=True)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.DATA_WRITE)

        def cold_write(a: int) -> None:
            reload_catalog()
            pxt.get_table('tbl').insert([{'a': a}])

        (
            MultiThreadedScenario()
            # the reload precedes the fault: it replaces the thread's Runtime, and the fault manager with it
            .then_run(thread_id=0, name='drop thread 0 catalog', fn=reload_catalog)
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0,
                name='cache-cold write, holding its lock',
                event=fault.reached,
                fn=lambda: pxt.get_table('tbl').insert([{'a': 2}]),
            )
            .then_run_until(
                thread_id=1,
                name='second cache-cold write, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(t)),
                fn=lambda: cold_write(3),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        assert t.count() == 3

    def test_cold_cache_read_does_not_fail_fast(self, uses_db: None, fault_injection: None) -> None:
        """A first-touch read of a data-versioned table waits for a schema change rather than failing.

        The wait policy also follows from the kind, so a cold cache could apply the operational one and refuse the
        Reading the lock set from the store settles the kind, so the policy applied is the real one.
        """
        t = self.make_tbl(is_data_versioned=True)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)
        b_expr = t.a + 1

        def cold_read() -> None:
            reload_catalog()
            t2 = pxt.get_table('tbl')
            assert t2.select(t2.a, t2.b).collect()[0]['b'] == 2

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=b_expr)
            )
            .then_run_until(
                thread_id=1,
                name='cache-cold read, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(t)),
                fn=cold_read,
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

    def test_tbl_created_under_dropping_dir(self, uses_db: None, fault_injection: None) -> None:
        """A table created under a directory after the drop resolved its lock set must still be dropped.

        The drop is parked between its store table locks and its directory lock, which is the window in which the
        new table can appear. The directory lock does not catch it, since that is taken too late; the post-lock
        re-resolution of the path does, restarting the operation on a corrected set.
        """
        pxt.create_dir('d')
        victim = pxt.create_table('d/victim', {'a': pxt.Int | None}, _is_data_versioned=False)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='drop dir', event=fault.reached, fn=lambda: pxt.drop_dir('d', force=True))
            .then_run(
                thread_id=1,
                name='create a table under it',
                fn=lambda: pxt.create_table('d/latecomer', {'a': pxt.Int | None}, _is_data_versioned=False),
            )
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert pxt.get_dir_contents() == {'dirs': [], 'tables': []}
        assert pxt.list_tables() == []
        del victim

    def test_concurrent_drop_dir_and_create_view(self, uses_db: None, fault_injection: None) -> None:
        """A drop_dir and a create_view that cross directories must not wedge each other.

        The drop is parked between the two lock phases, holding the base's store table and no directory row;
        the create_view it excludes lives in the other directory. Under an order that held a directory row while
        waiting for a store table this pair could deadlock. Whichever operation loses must say so as a Pixeltable
        error rather than as a deadlock or a raw store error.
        """
        pxt.create_dir('d1')
        pxt.create_dir('d2')
        base = pxt.create_table('d1/base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)

        def create_view_after_the_drop() -> None:
            # the drop takes the base with it, so the view has nothing left to be a view of
            with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
                pxt.create_view('d2/v', base)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='drop d1', event=fault.reached, fn=lambda: pxt.drop_dir('d1', force=True))
            .then_run_until(
                thread_id=1,
                name='create a view in d2, which waits',
                poll_condition=lambda: self.is_blocked_on(self.store_tbl_name(base)),
                fn=create_view_after_the_drop,
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        assert pxt.get_dir_contents()['dirs'] == ['d2']
        assert pxt.list_tables() == []

    def test_view_invisible_until_its_creator_commits(self, uses_db: None, fault_injection: None) -> None:
        """A table is never visible without a store table to lock.

        The creator is parked inside the transaction that writes the view's metadata and creates its store table,
        so this is the widest window in which the two could disagree. Nothing outside may see the view yet; once
        the transaction commits it is there in full.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        base.insert([{'a': 1}])
        fault = BlockFault()

        def assert_invisible() -> None:
            # a metadata-only read, so it takes no locks and is not excluded by the creator
            assert pxt.get_table('v', if_not_exists='ignore') is None
            assert pxt.list_tables() == ['base']

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED, fault=fault)
            .then_run_until(
                thread_id=0, name='create a view', event=fault.reached, fn=lambda: pxt.create_view('v', base)
            )
            .then_run(thread_id=1, name='the view is not visible yet', fn=assert_invisible)
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        v = pxt.get_table('v')
        assert v.count() == 1

    def test_write_finishes_an_interrupted_drop(self, uses_db: None, fault_injection: None) -> None:
        """A write that meets a half-finished drop finalizes it and then proceeds.

        The drop is parked between two of its ops, which is a gap where no lock is held: the view's md row and
        store table are both still there, and its pending ops are not resolved. Since a committed `tables` row
        implies an existing store table, this is the only mid-drop state a locker can observe, which is why the
        StaleLockSetError recovery has no branch for a missing store table with an md row still in place.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        pxt.create_view('v', base)
        fault = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=fault)
            .then_run_until(thread_id=0, name='drop the view', event=fault.reached, fn=lambda: pxt.drop_table('v'))
            .then_run(thread_id=1, name='write to the base', fn=lambda: base.insert([{'a': 1}]))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None

    def test_ancestor_dropped_under_a_live_handle(self, uses_db: None) -> None:
        """A handle to a view whose base was dropped reports table_was_dropped, never UndefinedTable."""
        base = pxt.create_table('base', {'a': pxt.Int | None})
        v = pxt.create_view('v', base)
        pxt.drop_table('base', force=True)
        with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
            _ = v.select(v.a).collect()

    def test_dropped_under_a_live_handle(self, is_data_versioned: bool, uses_db: None) -> None:
        """A handle to a table another thread dropped reports table_was_dropped, never UndefinedTable."""
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        pxt.drop_table('tbl')
        with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
            _ = t.select(t.a).collect()

    def test_lock_on_a_dropped_store_tbl(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """LOCK TABLE on a store table that is gone recovers rather than surfacing UndefinedTable.

        The operation is parked on its way *into* the lock, the table is dropped in full, and the lock statement
        then names a store table that no longer exists. Postgres raises the same UndefinedTable whether it was
        already gone or was dropped while we waited for it, so this covers the waiter case too, and
        deterministically, which the waiter form is not: a waiter is granted the lock as soon as the drop's md
        transaction commits, which is several transactions before the store table actually goes away.

        What must happen: the lock set is rebuilt from the store, which no longer names the table, and the
        operation reports that the table is gone instead of a raw UndefinedTable or an endless retry.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        b_expr = t.a + 1
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is _TblOpClass.MD_WRITE)

        def expect_dropped() -> None:
            with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
                t.add_computed_column(b=b_expr)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_BEFORE_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='schema change, pre-lock', event=fault.reached, fn=expect_dropped)
            .then_run(thread_id=1, name='drop the table', fn=lambda: pxt.drop_table('tbl'))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )
        assert pxt.list_tables() == []
