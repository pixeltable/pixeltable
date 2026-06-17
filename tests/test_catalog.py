from textwrap import dedent

import psycopg
import pytest
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Path, is_valid_identifier
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.coordinator import MultiThreadedScenario
from tests.fault_injection import BlockFault, ExceptionFault
from tests.utils import pxt_raises


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

    def test_local_catalog_uri(self) -> None:
        # A plain path lives in the local catalog (empty catalog_uri, no org/db).
        local = Path.parse('a.b')
        assert local.org is None
        assert local.db is None
        assert local.catalog_uri == ''

    def test_hosted_path_parse(self) -> None:
        """Path.parse() understands pxt:// URIs and Pixeltable web URLs."""
        hosted = Path.parse('pxt://variata:main/dir/tbl')
        assert hosted.org == 'variata'
        assert hosted.db == 'main'
        assert hosted.components == ('dir', 'tbl')
        assert hosted.catalog_uri == 'pxt://variata:main'
        assert str(hosted) == 'pxt://variata:main/dir/tbl'

        # Versioned hosted path.
        versioned = Path.parse('pxt://local:testdb/dir/tbl:7', allow_versioned_path=True)
        assert (versioned.org, versioned.db, versioned.components, versioned.version) == (
            'local',
            'testdb',
            ('dir', 'tbl'),
            7,
        )

        # Org without a db.
        no_db = Path.parse('pxt://variata/tbl')
        assert no_db.org == 'variata'
        assert no_db.db is None
        assert no_db.catalog_uri == 'pxt://variata'

        # A Pixeltable web URL normalizes to the same parse as its pxt:// form.
        assert Path.parse('https://pixeltable.com/t/variata:main/dir/tbl') == Path.parse('pxt://variata:main/dir/tbl')

        # str() round-trips for both local and hosted paths.
        assert all(
            Path.parse(str(p), allow_versioned_path=True) == p for p in (Path.parse('a/b'), hosted, versioned, no_db)
        )

    def test_hosted_path_errors(self) -> None:
        # pxt:// with no org.
        for bad in ('pxt://', 'pxt:///tbl'):
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(bad)
        # Negative version.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main/tbl:-1', allow_versioned_path=True)
        # Bad identifier component in a hosted path.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main/a..b')
        # Org slug parses out of the netloc but isn't a valid identifier.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://bad org/tbl')
        # An extra colon lands in the db slug, which then fails identifier validation.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main:extra/tbl')

    def test_path_construction_invariants(self) -> None:
        # Invariants enforced at construction, so they hold for from_components() (and direct
        # construction), not only for parse().
        # A db requires an org.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), db='main')
        # Org and db must be valid slugs.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), org='bad org')
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), org='variata', db='bad:db')
        # Version must be non-negative.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), version=-1)
        # Components must be valid identifiers (the ('',) root sentinel excepted), and non-empty.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('a', 'bad name'))
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('a', ''))
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(())
        assert Path.from_components(('a', 'b')).components == ('a', 'b')
        assert Path.from_components(('',)).is_root  # the root sentinel is accepted
        # Hyphenated org/db slugs are accepted.
        hosted = Path.parse('pxt://my-org:my-db/tbl')
        assert (hosted.org, hosted.db) == ('my-org', 'my-db')
        assert Path.from_components(('tbl',), org='my-org', db='my-db').catalog_uri == 'pxt://my-org:my-db'

    def test_hosted_path_navigation(self) -> None:
        # Navigation preserves the catalog (org/db) and drops the version.
        path = Path.parse('pxt://variata:main/a/b/c:3', allow_versioned_path=True)
        assert path.parent == Path.from_components(('a', 'b'), org='variata', db='main')
        assert path.append('d') == Path.from_components(('a', 'b', 'c', 'd'), org='variata', db='main')
        assert path.ancestors() == [
            Path.from_components(('',), org='variata', db='main'),
            Path.from_components(('a',), org='variata', db='main'),
            Path.from_components(('a', 'b'), org='variata', db='main'),
        ]
        # Same-named local and hosted paths are distinct.
        assert Path.parse('a/b') != Path.parse('pxt://variata:main/a/b')

    def test_hosted_ops_unsupported(self) -> None:
        # Hosted catalogs are stubbed: an operation routed to a hosted path must raise a user-facing
        # UNSUPPORTED_OPERATION error, not a bare NotImplementedError.
        uri = 'pxt://variata:main'
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.get_table(f'{uri}/tbl')
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.create_table(f'{uri}/tbl', {'c': pxt.String})
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.drop_table(f'{uri}/tbl')
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.create_dir(f'{uri}/d')
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.drop_dir(f'{uri}/d')
        with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.move(f'{uri}/a', f'{uri}/b')
        assert not Path.parse('a').is_ancestor(Path.parse('pxt://variata:main/a/b'))

    @pytest.mark.parametrize('path_str', ['a.b.c', 'a/b/c'])
    def test_path_ancestors(self, path_str: str) -> None:
        # Test with both dot and slash paths (both result in '/' representation)
        # multiple ancestors in path
        path = Path.parse(path_str)
        expected_ancestors = [
            Path.from_components(('',)),
            Path.from_components(('a',)),
            Path.from_components(('a', 'b')),
        ]
        assert path.ancestors() == expected_ancestors

        # single element in path
        path = Path.parse('a')
        assert path.ancestors() == [Path.from_components(('',))]

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
        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir/subdir')

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
                 Name      Kind Version              Base
            snapshot1  snapshot                   view1:2
            snapshot2  snapshot          test_dir/view2:0
               subdir       dir                          |
                  tbl     table       4                  |
                view2      view       1      test_dir/tbl
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

    def test_finalize_pending_ops_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
        exc = sql_exc.DBAPIError('', {}, orig=psycopg.errors.SerializationFailure())
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)
        t.add_column(b=pxt.Int)
        fault.assert_count(1)
        _ = t.select(t.b).collect()

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
