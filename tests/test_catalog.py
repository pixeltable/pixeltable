import enum
import logging
from typing import Any, Callable

import psycopg
import pytest
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import telemetry_schemas
from pixeltable.catalog.locking import TblOpClass, get_store_tbl_name
from pixeltable.catalog.view import View
from pixeltable.env import Env, store_app_name
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation

from .coordinator import MultiThreadedScenario
from .fault_injection import BlockFault, ExceptionFault
from .utils import DatabaseRoot, RecordingSubscriber, pxt_raises, reload_catalog

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
        # Fail the last operation of view creation. The catalog must abort creation and undo the completed operations.
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
        Thread 0 drops a view and pauses during finalization while holding no locks. Thread 1 inserts into the base
        table, finishing the pending view drop before inserting. The insert must succeed; it previously failed
        with a "table not found" error.
        """
        base = pxt.create_table('base', {'a': pxt.Int | None})
        _ = pxt.create_view('v', base)
        block_in_finalize = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=block_in_finalize
            )
            .then_run_until(
                thread_id=0, name='drop view', event=block_in_finalize.reached, fn=lambda: pxt.drop_table('v')
            )
            .then_run(thread_id=1, name='insert into base', fn=lambda: base.insert([{'a': 1}]))
            .then_unblock(thread_id=1, fault=block_in_finalize)
            .execute()
        )

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None

    @pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
    def test_drop_base_finalized_before_view(self, uses_db: None, fault_injection: None) -> None:
        """
        The view chain is root -> base -> v. Thread 0 drops base and v, then pauses while finalizing v's drop,
        holding no locks. Thread 1 inserts into root and finishes dropping base, leaving v's drop unfinished.
        Thread 0 must then finish dropping v even though base is gone.
        """
        root = pxt.create_table('root', {'a': pxt.Int | None})
        base = pxt.create_view('base', root)
        _ = pxt.create_view('v', base)
        block_in_finalize = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=block_in_finalize
            )
            .then_run_until(
                thread_id=0,
                name='drop base and view',
                event=block_in_finalize.reached,
                fn=lambda: pxt.drop_table('base', force=True),
            )
            .then_run(thread_id=1, name='insert into root', fn=lambda: root.insert([{'a': 1}]))
            .then_unblock(thread_id=1, fault=block_in_finalize)
            .execute()
        )

        assert pxt.list_tables() == ['root']
        assert root.list_views() == []


class Outcome(enum.Enum):
    """Expected result when an operation starts while another holds its locks."""

    PROCEEDS = 'proceeds'
    WAITS = 'waits'
    FAILS_FAST = 'fails_fast'


_READ = TblOpClass.DATA_READ
_WRITE = TblOpClass.DATA_WRITE
_MD_WRITE = TblOpClass.MD_WRITE
_FINALIZE = TblOpClass.FINALIZE
_PROCEEDS, _WAITS, _FAILS_FAST = Outcome.PROCEEDS, Outcome.WAITS, Outcome.FAILS_FAST

# (Lock holder, arriving operation) -> expected outcome for operational tables.
OP_MATRIX: dict[tuple[TblOpClass, TblOpClass], Outcome] = {
    (_READ, _READ): _PROCEEDS,
    (_READ, _WRITE): _PROCEEDS,
    (_READ, _MD_WRITE): _WAITS,
    (_WRITE, _READ): _PROCEEDS,
    (_WRITE, _WRITE): _PROCEEDS,
    (_WRITE, _MD_WRITE): _WAITS,
    (_MD_WRITE, _READ): _FAILS_FAST,
    (_MD_WRITE, _WRITE): _FAILS_FAST,
    (_MD_WRITE, _MD_WRITE): _WAITS,
    (_FINALIZE, _READ): _FAILS_FAST,
    (_FINALIZE, _WRITE): _FAILS_FAST,
    (_FINALIZE, _MD_WRITE): _WAITS,
}

# Data-versioned tables serialize writers and wait for schema changes.
VERSIONED_MATRIX: dict[tuple[TblOpClass, TblOpClass], Outcome] = {
    (_READ, _READ): _PROCEEDS,
    (_READ, _WRITE): _PROCEEDS,
    (_READ, _MD_WRITE): _WAITS,
    (_WRITE, _READ): _PROCEEDS,
    (_WRITE, _WRITE): _WAITS,  # differs from operational
    (_WRITE, _MD_WRITE): _WAITS,
    (_MD_WRITE, _READ): _WAITS,  # differs from operational
    (_MD_WRITE, _WRITE): _WAITS,  # differs from operational
    (_MD_WRITE, _MD_WRITE): _WAITS,
    (_FINALIZE, _READ): _WAITS,  # differs from operational
    (_FINALIZE, _WRITE): _WAITS,  # differs from operational
    (_FINALIZE, _MD_WRITE): _WAITS,
}


@pytest.mark.db_roots('local', reason='fault-injection/concurrency test against the in-process catalog internals')
class TestLocking:
    """Test table locking with controlled thread scheduling; see docs/design/catalog-locking.md.

    Pause one transaction at a fault point and check whether a second operation proceeds, waits, or fails fast.

    Unlike most tests, these depend on catalog internals, because the public API cannot express the scenarios:
    - The holder must be paused inside its transaction while it holds its locks. Only a fault point can do that, and
      TblOpClass selects which of an operation's transactions to pause.
    - "Waits" has no public signal: a waiting operation and a slow one look the same until both return. The tests
      detect a wait by an ungranted request in pg_locks, which is keyed by the store table name. A timeout instead
      would be slow and flaky.
    """

    @staticmethod
    def is_blocked_on_tbl_lock(t: pxt.Table) -> bool:
        """Whether some backend is waiting for a lock on the store table backing `t`."""
        q = sql.text(
            'SELECT count(*) FROM pg_locks '
            "WHERE locktype = 'relation' AND relation = CAST(:name AS regclass) AND NOT granted"
        )
        store_tbl_name = get_store_tbl_name(t._id, is_view=isinstance(t, View))
        with Env.get().engine.connect() as conn:
            return conn.execute(q, {'name': store_tbl_name}).scalar() > 0

    @staticmethod
    def make_tbl(name: str = 'tbl', *, is_data_versioned: bool) -> pxt.Table:
        t = pxt.create_table(name, {'a': pxt.Int | None}, _is_data_versioned=is_data_versioned)
        t.insert([{'a': 1}])
        return t

    @staticmethod
    def op_fn(op_class: TblOpClass, t: pxt.Table, col_name: str) -> Callable[[], Any]:
        """Build a callable for `op_class` on `t`. MD_WRITE and FINALIZE add a column named `col_name`.

        Build queries and column expressions before starting the scenario. Their metadata reads open separate
        transactions and would otherwise interfere with the operation being tested.
        """
        if op_class is TblOpClass.DATA_READ:
            return t.select().collect
        if op_class is TblOpClass.DATA_WRITE:
            return lambda: t.insert([{'a': 2}])
        if op_class is TblOpClass.MD_WRITE:
            value_expr = t.a + 1
            return lambda: t.add_computed_column(**{col_name: value_expr})
        # add_column() leaves pending DDL for a finalization transaction, which the fault filter can pause.
        # add_computed_column() runs ALTER TABLE in the metadata transaction instead.
        assert op_class is TblOpClass.FINALIZE
        return lambda: t.add_column(**{col_name: pxt.Int | None})

    @staticmethod
    def assert_fails_fast(fn: Callable[[], Any]) -> None:
        """Check that the operation reports SCHEMA_CHANGE_IN_PROGRESS.

        An unintended retry would wait indefinitely while the holder is paused, causing the scenario to time out.
        """
        with pxt_raises(excs.ErrorCode.SCHEMA_CHANGE_IN_PROGRESS):
            fn()

    @pytest.mark.parametrize('in_flight,arriving', list(OP_MATRIX))
    def test_blocking_matrix(
        self, is_data_versioned: bool, in_flight: TblOpClass, arriving: TblOpClass, uses_db: None, fault_injection: None
    ) -> None:
        """Check every supported pair of concurrent operation classes.

        Test both conflicts and compatible operations: an implementation that blocks everything must fail too.
        """
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        outcome = (VERSIONED_MATRIX if is_data_versioned else OP_MATRIX)[in_flight, arriving]
        # A schema change can open several transactions; pause the one matching the class under test.
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is in_flight)
        holder = self.op_fn(in_flight, t, 'c_holder')
        arriving_fn = self.op_fn(arriving, t, 'c_arriving')
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
                    poll_condition=lambda: self.is_blocked_on_tbl_lock(t),
                    fn=arriver,
                ).then_unblock(thread_id=2, fault=fault)
            )
        else:
            fn = arriver if outcome is Outcome.PROCEEDS else (lambda: self.assert_fails_fast(arriver))
            scenario.then_run(thread_id=1, name=f'arriving {arriving.value}', fn=fn).then_unblock(
                thread_id=1, fault=fault
            )
        scenario.execute()

        expected_rows = 1
        if in_flight is TblOpClass.DATA_WRITE:
            expected_rows += 1
        if arriving is TblOpClass.DATA_WRITE and outcome is not Outcome.FAILS_FAST:
            expected_rows += 1
        assert t.count() == expected_rows

        if in_flight is TblOpClass.MD_WRITE and is_data_versioned:
            if arriving is TblOpClass.DATA_WRITE:
                # re-fetch: this thread's handle predates the schema change
                t2 = pxt.get_table('tbl')
                assert t2.where(t2.a == 2).select(t2.c_holder).collect()['c_holder'] == [3]
            if arriving is TblOpClass.DATA_READ:
                assert arriving_result[0].schema.keys() == {'a', 'c_holder'}, arriving_result[0].schema

    def test_concurrent_schema_changes(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """A waiting schema change must see metadata committed by the preceding schema change."""
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)
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
                poll_condition=lambda: self.is_blocked_on_tbl_lock(t),
                fn=lambda: t.add_computed_column(c=c_expr),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        # Re-fetch: this thread's handle predates both schema changes.
        t2 = pxt.get_table('tbl')
        res = t2.select(t2.a, t2.b, t2.c).collect()
        assert len(res) == 1 and res[0]['b'] == 2 and res[0]['c'] == 101, res

    def test_concurrent_insert_schema_change(self, uses_db: None, fault_injection: None) -> None:
        """An insert waiting for a schema change must populate the newly added computed column."""
        t = pxt.create_table('tbl', {'a': pxt.Int | None}, _is_data_versioned=True)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)
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
                poll_condition=lambda: self.is_blocked_on_tbl_lock(t),
                fn=lambda: t.insert([{'a': 1}]),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        t2 = pxt.get_table('tbl')
        res = t2.select(t2.a, t2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_unrelated_table_not_locked(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """A schema change must allow reads and writes on an unrelated table."""
        a = self.make_tbl('a', is_data_versioned=is_data_versioned)
        b = self.make_tbl('b', is_data_versioned=is_data_versioned)
        x_expr = a.a + 1
        read_b = b.select(b.a).collect
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)

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

    def test_md_only_reads_do_not_block(self, is_data_versioned: bool, uses_db: None, fault_injection: None) -> None:
        """Catalog listings and table descriptions remain available while a schema change holds its lock, as long as
        there are no pending ops. add_computed_column, add_embedding_index, add_btree_index are examples of long-running
        schema changes that do not use pending table ops, therefore do not block concurrent reads."""
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        b_expr = t.a + 1
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)

        def md_only_reads() -> None:
            assert 'tbl' in pxt.list_tables()
            assert 'tbl' in pxt.get_dir_contents()['tables']
            t2 = pxt.get_table('tbl')
            assert 'tbl' in repr(t2)
            assert t2.list_views() == []

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
        """A metadata read that encounters pending operations must wait for finalization."""
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        block_on_finalize = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.FINALIZE)
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
                poll_condition=lambda: self.is_blocked_on_tbl_lock(t),
                fn=lambda: result.append(pxt.get_table('tbl')),
            )
            .then_unblock(thread_id=2, fault=block_on_finalize)
            .execute()
        )

        assert len(result) == 1
        assert 'b' in result[0].columns()

    def test_base_write_blocks_view_schema_change(self, uses_db: None, fault_injection: None) -> None:
        """A base-table write must block schema changes on its mutable views."""
        base = pxt.create_table('base', {'a': pxt.Int | None})
        v = pxt.create_view('v', base)
        other_base = pxt.create_table('other_base', {'a': pxt.Int | None})
        other_v = pxt.create_view('other_v', other_base)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.DATA_WRITE)
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
                poll_condition=lambda: self.is_blocked_on_tbl_lock(v),
                fn=lambda: v.add_computed_column(b=b_expr),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        v2 = pxt.get_table('v')
        res = v2.select(v2.a, v2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_base_tabl_write_blocks_create_view(self, uses_db: None, fault_injection: None) -> None:
        """create_view() must wait for a base-table write and include the inserted row."""
        base = pxt.create_table('base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.DATA_WRITE)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(
                thread_id=0, name='insert into base', event=fault.reached, fn=lambda: base.insert([{'a': 1}])
            )
            .then_run_until(
                thread_id=1,
                name='create_view, which waits',
                poll_condition=lambda: self.is_blocked_on_tbl_lock(base),
                fn=lambda: pxt.create_view('v', base),
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        assert pxt.get_table('v').count() == 1

    def test_stale_lock_set_view_added(
        self, uses_db: None, fault_injection: None, telemetry_recorder: RecordingSubscriber
    ) -> None:
        """A base table write's tentative lock set is invalidated by a concurrent view creation."""
        base = pxt.create_table('base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.DATA_WRITE)

        def create_view() -> None:
            v = pxt.create_view('v', base)
            v.add_computed_column(b=v.a + 1)

        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 0
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
        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 1

        v2 = pxt.get_table('v')
        res = v2.select(v2.a, v2.b).collect()
        assert len(res) == 1 and res[0]['b'] == 2, res

    def test_stale_lock_set_view_dropped(
        self, uses_db: None, fault_injection: None, telemetry_recorder: RecordingSubscriber
    ) -> None:
        """A base write must retry if a view in its tentative lock set is dropped before lock acquisition."""
        base = pxt.create_table('base', {'a': pxt.Int | None})
        pxt.create_view('v', base)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.DATA_WRITE)

        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 0
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
        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 1

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None

    def test_cold_cache(self, uses_db: None, telemetry_recorder: RecordingSubscriber) -> None:
        """Operations with an empty cache must lock and reach every affected table."""
        pxt.create_dir('d1')
        pxt.create_dir('d2')
        base = pxt.create_table('d1/base', {'a': pxt.Int | None})
        base.insert([{'a': 1}])
        v1 = pxt.create_view('d2/v1', base, additional_columns={'b': base.a + 1})
        pxt.create_view('d2/v2', v1, additional_columns={'c': v1.b + 1})
        pxt.create_snapshot('d2/s', base, additional_columns={'d': base.a + 10})

        # a write through a handle obtained before the cache was emptied, so not even the target is cached
        reload_catalog()
        base.insert([{'a': 2}])
        v2 = pxt.get_table('d2/v2')
        res = v2.select(v2.a, v2.b, v2.c).order_by(v2.a).collect()
        assert list(zip(res['a'], res['b'], res['c'])) == [(1, 2, 3), (2, 3, 4)]
        s = pxt.get_table('d2/s')
        assert s.select(s.d).collect()['d'] == [11]

        # a schema change on the base
        reload_catalog()
        t = pxt.get_table('d1/base')
        t.add_computed_column(x=t.a * 10)
        v2 = pxt.get_table('d2/v2')
        res = v2.select(v2.a, v2.x).order_by(v2.a).collect()
        assert list(zip(res['a'], res['x'])) == [(1, 10), (2, 20)]

        # dropping the base drops its views and snapshot
        reload_catalog()
        pxt.drop_table('d1/base', force=True)
        assert pxt.list_tables() == []
        assert sorted(pxt.list_dirs()) == ['d1', 'd2']
        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 0

    def test_tbl_created_under_dropping_dir(
        self, uses_db: None, fault_injection: None, telemetry_recorder: RecordingSubscriber
    ) -> None:
        """A directory drop must include a table created after its lock set was built."""
        pxt.create_dir('d')
        _ = pxt.create_table('d/victim', {'a': pxt.Int | None}, _is_data_versioned=False)
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)

        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 0
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
        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 1

        assert pxt.get_dir_contents() == {'dirs': [], 'tables': []}
        assert pxt.list_tables() == []

    def test_concurrent_drop_dir_and_create_view(self, uses_db: None, fault_injection: None) -> None:
        """Dropping a base's directory while creating a view elsewhere."""
        pxt.create_dir('d1')
        pxt.create_dir('d2')
        base = pxt.create_table('d1/base', {'a': pxt.Int | None})
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)

        def create_view_after_the_drop() -> None:
            with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
                pxt.create_view('d2/v', base)

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_AFTER_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='drop d1', event=fault.reached, fn=lambda: pxt.drop_dir('d1', force=True))
            .then_run_until(
                thread_id=1,
                name='create a view in d2, which waits',
                poll_condition=lambda: self.is_blocked_on_tbl_lock(base),
                fn=create_view_after_the_drop,
            )
            .then_unblock(thread_id=2, fault=fault)
            .execute()
        )

        assert pxt.get_dir_contents()['dirs'] == ['d2']
        assert pxt.list_tables() == []

    def test_concurrent_create_dir_and_parents(self, uses_db: None, fault_injection: None) -> None:
        """Concurrent create_dir(parents=True) calls must create a single directory chain."""
        fault = BlockFault()

        def create_chain() -> None:
            pxt.create_dir('a/b/c', parents=True, if_exists='ignore')

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_CREATE_DIR_BEFORE_PARENT_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='create a/b/c (paused)', event=fault.reached, fn=create_chain)
            .then_run(thread_id=1, name='create a/b/c', fn=create_chain)
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        assert pxt.list_dirs() == ['a', 'a/b', 'a/b/c']
        assert pxt.list_dirs('a') == ['a/b', 'a/b/c']

    def test_write_finishes_an_interrupted_drop(self, uses_db: None, fault_injection: None) -> None:
        """A base write must complete a view's pending drop before proceeding."""
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

    def test_tentative_lock_set_with_dropped_tbl(
        self, is_data_versioned: bool, uses_db: None, fault_injection: None, telemetry_recorder: RecordingSubscriber
    ) -> None:
        """A missing store table during lock acquisition must result in TABLE_NOT_FOUND."""
        t = self.make_tbl(is_data_versioned=is_data_versioned)
        b_expr = t.a + 1
        fault = BlockFault(filter=lambda kwargs: kwargs['op_class'] is TblOpClass.MD_WRITE)

        def expect_dropped() -> None:
            with pxt_raises(excs.ErrorCode.TABLE_NOT_FOUND):
                t.add_computed_column(b=b_expr)

        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 0
        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_BEFORE_TBL_LOCK, fault=fault)
            .then_run_until(thread_id=0, name='schema change, pre-lock', event=fault.reached, fn=expect_dropped)
            .then_run(thread_id=1, name='drop the table', fn=lambda: pxt.drop_table('tbl'))
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )
        assert telemetry_recorder.counter_total(telemetry_schemas.lock_set_restarts) == 1
        assert pxt.list_tables() == []
