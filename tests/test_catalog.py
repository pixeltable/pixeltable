from textwrap import dedent

import psycopg
import pytest
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Path, is_valid_identifier
from pixeltable.runtime import get_runtime
from pixeltable.share.packager import TablePackager, TableRestorer
from pixeltable.utils.fault_injection import BlockFault, ExceptionFault, FaultLocation
from tests.conftest import clean_db
from tests.coordinator import MultiThreadedScenario
from tests.utils import pxt_raises, reload_catalog


class TestCatalog:
    """Tests for miscellanous catalog functions."""

    def test_valid_identifier(self) -> None:
        valid_ids = ['a', 'a1', 'a_1', 'a_']
        invalid_ids = ['', '_', '__', '_a', '1a', 'a.b', '.a', 'a-b']
        for valid_id in valid_ids:
            assert is_valid_identifier(valid_id), valid_ids

        for invalid_id in invalid_ids:
            assert not is_valid_identifier(invalid_id), invalid_ids

    def test_valid_path(self) -> None:
        """Test path validation using Path.parse()."""
        # Test empty path
        Path.parse('', allow_empty_path=True)  # Should succeed
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('', allow_empty_path=False)  # Should fail

        valid_paths = ['a', 'a_.b_', 'a.b.c', 'a.b.c.d']
        invalid_paths = ['.', '..', 'a.', '.a', 'a..b']

        for valid_path in valid_paths:
            # Should succeed with both empty_is_valid settings
            Path.parse(valid_path, allow_empty_path=False)
            Path.parse(valid_path, allow_empty_path=True)

        for invalid_path in invalid_paths:
            # Should fail regardless of empty_is_valid setting
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(invalid_path, allow_empty_path=False)
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(invalid_path, allow_empty_path=True)

    def test_path_parse(self) -> None:
        """Test Path.parse() with '/' delimiter and backward compatibility with '.'."""
        # Test valid paths with SLASH
        valid_slash_paths = ['a', 'a_', 'a/b', 'a/b/c', 'a_/b_']
        for valid_path in valid_slash_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == tuple(valid_path.split('/'))
            assert str(parsed) == valid_path

        # Test backward compatibility with DOT delimiter
        valid_dot_paths = ['a', 'a_', 'a.b', 'a.b.c', 'a_.b_']
        for valid_path in valid_dot_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == tuple(valid_path.split('.'))
            # String representation always uses SLASH
            assert str(parsed) == valid_path.replace('.', '/')

        # Test empty path
        empty_parsed = Path.parse('', allow_empty_path=True)
        assert empty_parsed.components == ('',)
        assert str(empty_parsed) == ''

        # Test versioned paths with SLASH
        versioned = Path.parse('a/b/c:5', allow_versioned_path=True)
        assert versioned.components == ('a', 'b', 'c')
        assert versioned.version == 5
        assert str(versioned) == 'a/b/c:5'

        # Test versioned paths with DOT (backward compatibility)
        versioned_dot = Path.parse('a.b.c:5', allow_versioned_path=True)
        assert versioned_dot.components == ('a', 'b', 'c')
        assert versioned_dot.version == 5
        assert str(versioned_dot) == 'a/b/c:5'

    @pytest.mark.parametrize('path_str', ['a.b.c', 'a/b/c'])
    def test_path_ancestors(self, path_str: str) -> None:
        # Test with both dot and slash paths (both result in '/' representation)
        # multiple ancestors in path
        path = Path.parse(path_str)
        expected_ancestors = [Path(('',), None), Path(('a',), None), Path(('a', 'b'), None)]
        assert path.ancestors() == expected_ancestors

        # single element in path
        path = Path.parse('a')
        assert path.ancestors() == [Path(('',), None)]

        # root
        path = Path.parse('', allow_empty_path=True)
        assert path.ancestors() == []

    def test_path_delimiter_str_hash_compare(self) -> None:
        """Test that paths with different input delimiters but same components compare equal and hash the same."""
        # Parse with DOT delimiter (backward compatibility)
        dotted_path = Path.parse('a.b.c')

        # Parse with SLASH delimiter
        unix_path = Path.parse('a/b/c')

        assert dotted_path.components == unix_path.components == ('a', 'b', 'c')

        # String representation always uses SLASH
        assert str(dotted_path) == 'a/b/c'
        assert str(unix_path) == 'a/b/c'

        # both paths should be equal
        assert dotted_path == unix_path

        # both paths should have the same hash
        assert hash(dotted_path) == hash(unix_path)

        # Test with versioned paths
        dotted_versioned = Path.parse('a.b.c:5', allow_versioned_path=True)
        unix_versioned = Path.parse('a/b/c:5', allow_versioned_path=True)

        assert dotted_versioned.components == unix_versioned.components == ('a', 'b', 'c')
        assert dotted_versioned.version == unix_versioned.version == 5
        assert str(dotted_versioned) == 'a/b/c:5'
        assert str(unix_versioned) == 'a/b/c:5'
        assert dotted_versioned == unix_versioned
        assert hash(dotted_versioned) == hash(unix_versioned)

        # Test parent always uses SLASH
        dotted_parent = dotted_path.parent
        unix_parent = unix_path.parent
        assert str(dotted_parent) == 'a/b'
        assert str(unix_parent) == 'a/b'
        assert dotted_parent == unix_parent

        # Test append always uses SLASH
        dotted_appended = dotted_path.append('d')
        unix_appended = unix_path.append('d')
        assert str(dotted_appended) == 'a/b/c/d'
        assert str(unix_appended) == 'a/b/c/d'
        assert dotted_appended.components == unix_appended.components == ('a', 'b', 'c', 'd')

    def test_ls(self, uses_db: None) -> None:
        t = pxt.create_table('tbl_for_replica', {'a': pxt.Int})
        snapshot = pxt.create_snapshot('snapshot_for_replica', t)
        packager = TablePackager(snapshot)
        bundle_path = packager.package()
        clean_db()
        reload_catalog()

        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir/subdir')

        restorer = TableRestorer('test_dir/replica1')
        restorer.restore(bundle_path)

        t = pxt.create_table('test_dir/tbl', {'a': pxt.Int})
        t.insert(a=3)
        v1 = pxt.create_view('view1', t)
        t.insert(a=5)
        v1.add_column(b=pxt.Int)
        _s1 = pxt.create_snapshot('test_dir/snapshot1', v1)
        t.insert(a=22)
        v2 = pxt.create_view('test_dir/view2', t)
        _s2 = pxt.create_snapshot('test_dir/snapshot2', v2, additional_columns={'c': pxt.String})
        t.insert(a=4171780)
        df = pxt.ls('test_dir')
        print(repr(df))
        assert dedent(repr(df)) == dedent(
            '''
                 Name      Kind Version                      Base
             replica1   replica       0  <anonymous base table>:0
            snapshot1  snapshot                           view1:2
            snapshot2  snapshot                  test_dir/view2:0
               subdir       dir                                  |
                  tbl     table       4                          |
                view2      view       1              test_dir/tbl
            '''
        ).strip('\n').replace('|', '')  # fmt: skip

    def test_cross_type_replacement(self, uses_db: None) -> None:
        """Test that tables, views, and snapshots can replace each other with if_exists='replace'.

        This tests the path collision handling logic: dirs can only collide with dirs,
        but all table subtypes (table, view, snapshot) can collide with each other.
        """
        base_table = pxt.create_table('base', {'c1': pxt.Int})

        # One lambda per create_x with expected columns
        creators = {
            'table': (lambda: pxt.create_table('target', {'c2': pxt.String}, if_exists='replace'), ['c2']),
            'view': (
                lambda: pxt.create_view(
                    'target', base_table, additional_columns={'c3': pxt.String}, if_exists='replace'
                ),
                ['c3', 'c1'],
            ),
            'snapshot': (lambda: pxt.create_snapshot('target', base_table, if_exists='replace'), ['c1']),
        }

        # Test all permutations: each table subtype can replace any table subtype
        for existing_creator, _ in creators.values():
            for replacing_creator, expected_cols in creators.values():
                existing_creator()
                assert 'target' in pxt.list_tables()
                result = replacing_creator()
                assert 'target' in pxt.list_tables()
                assert result.columns() == expected_cols

        # Verify cross-type replacement is blocked in both directions for every table subtype
        pxt.drop_table('target')
        pxt.create_dir('target')
        for creator, _ in creators.values():
            # dirs cannot be replaced by table subtypes
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a table, view or snapshot'):
                creator()
            # table subtypes cannot be replaced by dirs
            pxt.drop_dir('target')
            creator()
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a directory'):
                pxt.create_dir('target', if_exists='replace')
            pxt.drop_table('target')
            pxt.create_dir('target')

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

    def test_finalize_pending_ops_retriable_error(self, uses_db: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
        exc = sql_exc.DBAPIError('', {}, orig=psycopg.errors.SerializationFailure())
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)
        t.add_column(b=pxt.Int)
        fault.assert_count(1)
        _ = t.select(t.b).collect()

    def test_finalize_pending_ops_non_retriable_error(self, uses_db: None) -> None:
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
        assert ls['Name'][0] == 'test', ls

    def test_concurrent_add_column_insert(self, uses_db: None) -> None:
        """Concurrent insert while add_column is blocked mid-finalize"""
        t = pxt.create_table('test', {'a': pxt.Int})
        fault = BlockFault()

        def t0_inject_block() -> None:
            get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)

        def t0_add_column() -> None:
            t = pxt.get_table('test')
            t.add_computed_column(b=t.a + 1)

        (
            MultiThreadedScenario()
            # Thread 0: arm the fault in pending table ops finalization
            .then_run(thread_id=0, name='inject fault', fn=t0_inject_block)
            # Thread 0: start adding a computed column, this will block at the fault point
            .then_run_until(thread_id=0, name='add column', event=fault.reached, fn=t0_add_column)
            # Thread 1: run an insert concurrently once thread 0 is blocked at the fault point
            .then_run(thread_id=1, name='insert', fn=lambda: pxt.get_table('test').insert([{'a': 1}]))
            # Unblock thread 0
            .then_run(thread_id=1, name='unblock thread 0', fn=lambda: fault.unblock())
            .execute()
        )

        # Both operations should have completed successfully.
        result = t.select(t.a, t.b).collect()
        assert len(result) == 1
        assert result[0] == {'a': 1, 'b': 2}
