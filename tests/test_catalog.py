from typing import Callable

import psycopg
import pytest
import sqlalchemy as sql
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.coordinator import MultiThreadedScenario
from tests.fault_injection import BlockFault, ExceptionFault
from tests.utils import pxt_raises


class TestCatalog:
    """Tests for miscellanous catalog functions."""

    def test_json_reserved_key(self, make_catalog_path: Callable[[str], str]) -> None:
        # JSON cell values are user data and may contain a key that collides with the proxy protocol's reserved
        # tag; inserting and reading such values back must round-trip rather than be rejected.
        p = make_catalog_path
        t = pxt.create_table(p('json_tbl'), {'id': pxt.Int, 'data': pxt.Json})
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

    def test_ls(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('test_dir'))
        pxt.create_dir(p('test_dir/subdir'))

        tbl_name = p('test_dir/tbl')
        t = pxt.create_table(tbl_name, {'a': pxt.Int})
        t.insert(a=3)
        v1_name = p('view1')
        v1 = pxt.create_view(v1_name, t)
        t.insert(a=5)
        v1.add_column(b=pxt.Int)
        _s1 = pxt.create_snapshot(p('test_dir/snapshot1'), v1)
        t.insert(a=22)
        v2_name = p('test_dir/view2')
        v2 = pxt.create_view(v2_name, t)
        _s2 = pxt.create_snapshot(p('test_dir/snapshot2'), v2, additional_columns={'c': pxt.String})
        t.insert(a=4171780)
        df = pxt.ls(p('test_dir'))
        # a hosted (proxy) table's Base shows its full catalog uri, which widens the column vs local; compare row
        # tokens so the assertion checks content (including the uris) independent of column padding.
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

        assert tokens(repr(df)) == tokens(expected)

    def test_cross_type_replacement(self, make_catalog_path: Callable[[str], str]) -> None:
        """Test that tables, views, and snapshots can replace each other with if_exists='replace'.

        This tests the path collision handling logic: dirs can only collide with dirs,
        but all table subtypes (table, view, snapshot) can collide with each other.
        """
        p = make_catalog_path
        base_table = pxt.create_table(p('base'), {'c1': pxt.Int})

        # One lambda per create_x with expected columns
        creators = {
            'table': (lambda: pxt.create_table(p('target'), {'c2': pxt.String}, if_exists='replace'), ['c2']),
            'view': (
                lambda: pxt.create_view(
                    p('target'), base_table, additional_columns={'c3': pxt.String}, if_exists='replace'
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
                'needs_xact': True,
            }
        )
        assert isinstance(op, CreateTableMdOp)
        assert op.needs_xact  # now a ClassVar
        assert 'needs_xact' not in op.to_dict()

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_finalize_pending_ops_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
        exc = sql_exc.DBAPIError('', {}, orig=psycopg.errors.SerializationFailure())
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)
        t.add_column(b=pxt.Int)
        fault.assert_count(1)
        _ = t.select(t.b).collect()

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_finalize_pending_ops_non_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
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

    @pytest.mark.local('recovers transparently when the server drops the pooled db connections')
    def test_dropped_connection(self, uses_db: None) -> None:
        if not Env.get().is_local:
            # the way this test drops connections (pg_terminate_backend on the pixeltable db) is specific to pgserver
            pytest.skip('requires pgserver')
        pxt.create_dir('d')
        t = pxt.create_table('d/t', {'a': pxt.Int})
        t.insert([{'a': 1}])

        def kill_connections() -> None:
            # Terminate this worker's backends out from under the pooled connections. The terminator runs on a
            # separate connection outside the engine's pool (and the statement excludes only its own backend),
            # so every connection the engine has pooled is killed.
            term_engine = sql.create_engine(Env.get().db_url, poolclass=sql.pool.NullPool)
            try:
                with term_engine.connect() as term:
                    term.execute(sql.text(Env.get()._pgserver_terminate_connections_stmt()))
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

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_concurrent_add_column_insert(self, uses_db: None, fault_injection: None) -> None:
        """Concurrent insert while add_column is blocked mid-finalize"""
        t = pxt.create_table('test', {'a': pxt.Int})
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

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_create_view_stale_base_tv_after_txn_failure(self, uses_db: None, fault_injection: None) -> None:
        """
        Verifies bug fix: due to an error in view creation, Catalog would fail to invalidate a modified but not
        persisted TableVersion. Later that would result in Pixeltable acting on that stale TableVersion, which can cause
        various sorts of issues including a data corruption.
        """
        base = pxt.create_table('base', {'a': pxt.Int})

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

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_load_view_concurrent_drop_view(self, uses_db: None, fault_injection: None) -> None:
        """
        Start with a base table and a view. Thread 0 loads the view md, and is about to initialize it when Thread 1
        drops it. Thread 0 then proceeds to initialize the view, which involves loading the base table. At READ
        COMMITTED isolation level, this scenario results in an AssertionError because the base table and
        the view are inconsistent with one another.
        """
        base = pxt.create_table('base', {'a': pxt.Int})
        v = pxt.create_view('v', base)
        block_before_init = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_LOAD_TBL_VERSION_BEFORE_INIT, fault=block_before_init
            )
            .then_run_until(thread_id=0, name='access column', event=block_before_init.reached, fn=lambda: v.a)
            # Thread 1: drop v while Thread 0 is waiting to initialize it
            .then_run(thread_id=1, name='drop view', fn=lambda: pxt.drop_table('v'))
            # unblock Thread 0 to continue with v initialization that also loads base
            .then_unblock(thread_id=1, fault=block_before_init)
            .execute()
        )

        assert pxt.get_table('v', if_not_exists='ignore') is None
        base.insert([{'a': 1}])
        assert base.count() == 1

    @pytest.mark.local('fault-injection/concurrency test against the in-process catalog internals')
    def test_drop_view_concurrent_insert(self, uses_db: None, fault_injection: None) -> None:
        """
        Start with a base table and a view. Thread 0 begins to drop the view, but pauses inside finalize pending ops
        (without the exclusive lock). Thread 1 swoops in in the meantime to insert a row into the base table, and
        finalizes view drop as a side effect. Before the fix, this would result in the insert failing with "table not
        found" error.
        """
        base = pxt.create_table('base', {'a': pxt.Int})
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
