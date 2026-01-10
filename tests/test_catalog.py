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

    def test_path_parse(self) -> None:
        """Test Path.parse() with '/' delimiter and backward compatibility with '.'."""
        # Test valid paths with SLASH
        valid_slash_paths = ['a', 'a_', 'a/b', 'a/b/c', 'a_/b_']
        for valid_path in valid_slash_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == valid_path.split('/')
            assert str(parsed) == valid_path

        # Test backward compatibility with DOT delimiter
        valid_dot_paths = ['a', 'a_', 'a.b', 'a.b.c', 'a_.b_']
        for valid_path in valid_dot_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == valid_path.split('.')
            # String representation always uses SLASH
            assert str(parsed) == valid_path.replace('.', '/')

        # Test empty path
        empty_parsed = Path.parse('', allow_empty_path=True)
        assert empty_parsed.components == ['']
        assert str(empty_parsed) == ''

        # Test versioned paths with SLASH
        versioned = Path.parse('a/b/c:5', allow_versioned_path=True)
        assert versioned.components == ['a', 'b', 'c']
        assert versioned.version == 5
        assert str(versioned) == 'a/b/c:5'

        # Test versioned paths with DOT (backward compatibility)
        versioned_dot = Path.parse('a.b.c:5', allow_versioned_path=True)
        assert versioned_dot.components == ['a', 'b', 'c']
        assert versioned_dot.version == 5
        assert str(versioned_dot) == 'a/b/c:5'

    @pytest.mark.parametrize('path_str', ['a.b.c', 'a/b/c'])
    def test_path_ancestors(self, path_str: str) -> None:
        # Test with both dot and slash paths (both result in '/' representation)
        # multiple ancestors in path
        path = Path.parse(path_str)
        expected_ancestors = [Path([''], None), Path(['a'], None), Path(['a', 'b'], None)]
        assert path.ancestors() == expected_ancestors

        # single element in path
        path = Path.parse('a')
        assert path.ancestors() == [Path([''], None)]

        # root
        path = Path.parse('', allow_empty_path=True)
        assert path.ancestors() == []

    def test_path_delimiter_str_hash_compare(self) -> None:
        """Test that paths with different input delimiters but same components compare equal and hash the same."""
        # Parse with DOT delimiter (backward compatibility)
        dotted_path = Path.parse('a.b.c')

        # Parse with SLASH delimiter
        unix_path = Path.parse('a/b/c')

        assert dotted_path.components == unix_path.components == ['a', 'b', 'c']

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

        assert dotted_versioned.components == unix_versioned.components == ['a', 'b', 'c']
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
        assert dotted_appended.components == unix_appended.components == ['a', 'b', 'c', 'd']

    def test_ls(self, reset_db: None) -> None:
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
