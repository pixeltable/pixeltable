from textwrap import dedent

import pytest

import pixeltable as pxt
from pixeltable.catalog import Path, is_valid_identifier, is_valid_path
from pixeltable.share.packager import TablePackager, TableRestorer
from tests.conftest import clean_db
from tests.utils import reload_catalog


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
        assert is_valid_path('', empty_is_valid=True)
        assert not is_valid_path('', empty_is_valid=False)

        valid_paths = ['a', 'a_.b_', 'a.b.c', 'a.b.c.d']
        invalid_paths = ['.', '..', 'a.', '.a', 'a..b']

        for valid_path in valid_paths:
            assert is_valid_path(valid_path, empty_is_valid=False), valid_path
            assert is_valid_path(valid_path, empty_is_valid=True), valid_path

        for invalid_path in invalid_paths:
            assert not is_valid_path(invalid_path, empty_is_valid=False), invalid_path
            assert not is_valid_path(invalid_path, empty_is_valid=True), invalid_path

    def test_path_ancestors(self) -> None:
        # multiple ancestors in path
        path = Path('a.b.c')
        ancestors = path.ancestors()
        assert str(next(ancestors)) == ''
        assert str(next(ancestors)) == 'a'
        assert str(next(ancestors)) == 'a.b'
        with pytest.raises(StopIteration):
            next(ancestors)

        # single element in path
        path = Path('a')
        ancestors = path.ancestors()
        assert str(next(ancestors)) == ''
        with pytest.raises(StopIteration):
            next(ancestors)

        # root
        path = Path('', empty_is_valid=True)
        ancestors = path.ancestors()
        with pytest.raises(StopIteration):
            next(ancestors)

    def test_ls(self, reset_db: None) -> None:
        t = pxt.create_table('tbl_for_replica', {'a': pxt.Int})
        snapshot = pxt.create_snapshot('snapshot_for_replica', t)
        packager = TablePackager(snapshot)
        bundle_path = packager.package()
        clean_db()
        reload_catalog()

        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir.subdir')

        restorer = TableRestorer('test_dir.replica1')
        restorer.restore(bundle_path)

        t = pxt.create_table('test_dir.tbl', {'a': pxt.Int})
        t.insert(a=3)
        v1 = pxt.create_view('view1', t)
        t.insert(a=5)
        v1.add_column(b=pxt.Int)
        _s1 = pxt.create_snapshot('test_dir.snapshot1', v1)
        t.insert(a=22)
        v2 = pxt.create_view('test_dir.view2', t)
        _s2 = pxt.create_snapshot('test_dir.snapshot2', v2, additional_columns={'c': pxt.String})
        t.insert(a=4171780)
        df = pxt.ls('test_dir')
        print(repr(df))
        assert dedent(repr(df)) == dedent(
            '''
                 Name              Kind Version                    Base
             replica1  snapshot-replica          <anonymous base table>
            snapshot1          snapshot                         view1:2
            snapshot2          snapshot                test_dir.view2:0
               subdir               dir                                |
                  tbl             table       4                        |
                view2              view       1            test_dir.tbl
            '''
        ).strip('\n').replace('|', '')  # fmt: skip
