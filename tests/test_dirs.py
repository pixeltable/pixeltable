from typing import Callable

import pytest

import pixeltable as pxt

from .utils import make_tbl, pxt_raises, reload_catalog


@pxt.udf
def _fail_on_neg(x: int) -> int:
    if x < 0:
        raise ValueError('negative')
    return x


class TestDirs:
    def test_create(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        dirs = ['dir1', 'dir1/sub1', 'dir1/sub1/subsub1']
        for name in dirs:
            pxt.create_dir(p(name))
        # the created dirs are verified to exist at the expected paths by the get_dir_contents checks below

        # invalid names
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Invalid path: 1dir'):
            pxt.create_dir(p('1dir'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Invalid path: _dir1'):
            pxt.create_dir(p('_dir1'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Invalid path: dir 1'):
            pxt.create_dir(p('dir 1'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match=r'Invalid path: dir1..sub2'):
            pxt.create_dir(p('dir1..sub2'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match=r'Invalid path: dir1.sub2.'):
            pxt.create_dir(p('dir1.sub2.'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match=r'Invalid path: .*dir1:sub2.'):
            pxt.create_dir(p('dir1:sub2.'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Versioned path not allowed here: .*dir1:120'):
            pxt.create_dir(p('dir1:120'))

        # existing dirs raise error by default
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='is an existing'):
            pxt.create_dir(p('dir1'))
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='is an existing'):
            pxt.create_dir(p('dir1/sub1'))
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='is an existing'):
            pxt.create_dir(p('dir1/sub1/subsub1'))

        # existing table
        make_tbl(p('dir1/t1'))
        _ = pxt.get_table(p('dir1/t1'))
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='is an existing'):
            pxt.create_dir(p('dir1/t1'))
        with pxt_raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match=r'does not exist. Create it first with:'):
            pxt.create_dir(p('dir2/sub2'))
        make_tbl(p('t2'))
        with pxt_raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match="Directory 't2' does not exist"):
            pxt.create_dir(p('t2/sub2'))

        # new client: force loading from store
        reload_catalog()

        listing = pxt.get_dir_contents(p(''), recursive=True)
        assert listing == {
            'dirs': [p('dir1'), p('dir1/sub1'), p('dir1/sub1/subsub1')],
            'tables': [p('dir1/t1'), p('t2')],
        }
        listing = pxt.get_dir_contents(p(''), recursive=False)
        assert listing == {'dirs': [p('dir1')], 'tables': [p('t2')]}
        listing = pxt.get_dir_contents(p('dir1'), recursive=True)
        assert listing == {'dirs': [p('dir1/sub1'), p('dir1/sub1/subsub1')], 'tables': [p('dir1/t1')]}
        listing = pxt.get_dir_contents(p('dir1'), recursive=False)
        assert listing == {'dirs': [p('dir1/sub1')], 'tables': [p('dir1/t1')]}
        listing = pxt.get_dir_contents(p('dir1/sub1'), recursive=True)
        assert listing == {'dirs': [p('dir1/sub1/subsub1')], 'tables': []}
        listing = pxt.get_dir_contents(p('dir1/sub1'), recursive=False)
        assert listing == {'dirs': [p('dir1/sub1/subsub1')], 'tables': []}

    def test_get_dir_tree(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        for name in ['dir1', 'dir1/sub1', 'dir1/sub1/subsub1']:
            pxt.create_dir(p(name))
        t = make_tbl(p('dir1/t1'))
        make_tbl(p('t2'))
        pxt.create_view(p('dir1/v'), t)
        pxt.create_view(p('dir1/snap'), t, is_snapshot=True)
        reload_catalog()

        subsub1 = {'name': 'subsub1', 'path': 'dir1/sub1/subsub1', 'kind': 'directory', 'entries': []}
        sub1 = {'name': 'sub1', 'path': 'dir1/sub1', 'kind': 'directory', 'entries': [subsub1]}
        t1 = {'name': 't1', 'path': 'dir1/t1', 'kind': 'table', 'version': 0, 'error_count': 0, 'base': None}
        v = {'name': 'v', 'path': 'dir1/v', 'kind': 'view', 'version': 0, 'error_count': 0, 'base': 'dir1/t1'}
        snap = {
            'name': 'snap',
            'path': 'dir1/snap',
            'kind': 'snapshot',
            'version': None,
            'error_count': 0,
            'base': 'dir1/t1:0',
        }
        dir1 = {'name': 'dir1', 'path': 'dir1', 'kind': 'directory', 'entries': [snap, sub1, t1, v]}
        t2 = {'name': 't2', 'path': 't2', 'kind': 'table', 'version': 0, 'error_count': 0, 'base': None}
        assert pxt.get_dir_tree(p('')) == [dir1, t2]

    def test_get_dir_tree_error_count(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        t = pxt.create_table(p('errs'), {'x': pxt.Int})
        t.add_computed_column(y=_fail_on_neg(t.x))
        t.insert([{'x': 1}, {'x': -1}, {'x': -2}, {'x': 3}], on_error='ignore')

        # Two failing rows; pixeltable counts each error per affected column slot, so the reported
        # count is 4 (= 2 rows x 2 slots: the computed column 'y' plus the row-level error slot).
        tree = pxt.get_dir_tree(p(''))
        assert len(tree) == 1
        node = tree[0]
        assert node['kind'] == 'table'
        assert node['error_count'] == 4

    def test_create_if_exists(self, make_catalog_path: Callable[[str], str]) -> None:
        """Test if_exists parameter of create_dir API"""
        p = make_catalog_path
        dirs = ['dir1', 'dir1/sub1', 'dir1/sub1/subsub1']
        expected = [p(d) for d in dirs]
        id_before = {}
        for name in dirs:
            dir = pxt.create_dir(p(name))
            id_before[name] = dir._id

        # invalid if_exists value is rejected
        with pxt_raises(
            pxt.ErrorCode.INVALID_ARGUMENT,
            match=r"if_exists must be one of: \['error', 'ignore', 'replace', 'replace_force'\]",
        ):
            pxt.create_dir(p('dir1'), if_exists='invalid')  # type: ignore[arg-type]

        # scenario 1: path already has a directory

        # if_exists='error' should raise error - already tested in test_create

        # if_exists='ignore' should not raise error and return existing Dir
        d1 = pxt.create_dir(p('dir1'), if_exists='ignore')
        assert d1._id == id_before['dir1']
        assert pxt.list_dirs(p(''), recursive=True) == expected
        d2 = pxt.create_dir(p('dir1/sub1'), if_exists='ignore')
        assert d2._id == id_before['dir1/sub1']
        assert pxt.list_dirs(p(''), recursive=True) == expected
        d3 = pxt.create_dir(p('dir1/sub1/subsub1'), if_exists='ignore')
        assert d3._id == id_before['dir1/sub1/subsub1']
        assert pxt.list_dirs(p(''), recursive=True) == expected

        # if_exists='replace' should replace existing Dir, but only if it's empty
        d3 = pxt.create_dir(p('dir1/sub1/subsub1'), if_exists='replace')
        assert d3._id != id_before['dir1/sub1/subsub1']
        id_before['dir1/sub1/subsub1'] = d3._id
        assert pxt.list_dirs(p(''), recursive=True) == expected
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists and is not empty'):
            pxt.create_dir(p('dir1/sub1'), if_exists='replace')
        assert pxt.list_dirs(p(''), recursive=True) == expected
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists and is not empty'):
            pxt.create_dir(p('dir1'), if_exists='replace')
        assert pxt.list_dirs(p(''), recursive=True) == expected

        # if_exists='replace_force' should replace existing Dir,
        # and all its children
        d3 = pxt.create_dir(p('dir1/sub1/subsub1'), if_exists='replace_force')
        assert d3._id != id_before['dir1/sub1/subsub1']
        assert pxt.list_dirs(p(''), recursive=True) == expected
        d2 = pxt.create_dir(p('dir1/sub1'), if_exists='replace_force')
        assert d2._id != id_before['dir1/sub1']
        assert pxt.list_dirs(p(''), recursive=True) == [p('dir1'), p('dir1/sub1')]
        d3 = pxt.create_dir(p('dir1/sub1/subsub1'))
        assert pxt.list_dirs(p(''), recursive=True) == expected
        d1 = pxt.create_dir(p('dir1'), if_exists='replace_force')
        assert d1._id != id_before['dir1']
        assert pxt.list_dirs(p(''), recursive=True) == [p('dir1')]

        # scenario 2: path already exists but is not a Dir
        make_tbl(p('dir1/t1'))
        for if_exists in ['ignore', 'replace', 'replace_force']:
            with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
                pxt.create_dir(p('dir1/t1'), if_exists=if_exists)  # type: ignore[arg-type]

    def _test_drop_if_not_exists(self, p: Callable[[str], str], dir_name: str) -> None:
        """Test if_not_exists parameter of drop_dir"""
        orig_dirs = pxt.list_dirs(p(''), recursive=True)
        # if_not_exists parameter can be used control behavior
        # of drop_dir when the directory doesn't exist.

        # if_not_exists='error' should raise error
        # default behavior is to raise error
        # why pytest.raises() instead of pxt_raises(): may fire PATH_NOT_FOUND or DIRECTORY_NOT_FOUND depending on path
        # state
        with pytest.raises(pxt.NotFoundError, match='does not exist'):
            pxt.drop_dir(p(dir_name), if_not_exists='error')
        with pytest.raises(pxt.NotFoundError, match='does not exist'):
            pxt.drop_dir(p(dir_name))

        # if_not_exists='ignore' should be successful but a no-op
        pxt.drop_dir(p(dir_name), if_not_exists='ignore')
        assert pxt.list_dirs(p(''), recursive=True) == orig_dirs
        # when force=True, if_not_exists is ignored
        pxt.drop_dir(p(dir_name), if_not_exists='error', force=True)
        assert pxt.list_dirs(p(''), recursive=True) == orig_dirs
        # invalid if_not_exists value is rejected, but only
        # when the directory doesn't exist.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            pxt.drop_dir(p(dir_name), if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

    def test_drop(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        dirs = ['dir1', 'dir1/sub1', 'dir1/sub1/subsub1']
        for name in dirs:
            pxt.create_dir(p(name))
        make_tbl(p('t1'))
        make_tbl(p('dir1/t1'))

        # bad name
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Invalid path: 1dir'):
            pxt.drop_dir(p('1dir'))
        # bad path
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match=r'Invalid path: dir1..sub1'):
            pxt.drop_dir(p('dir1..sub1'))
        with pxt_raises(pxt.ErrorCode.INVALID_PATH, match='Versioned path not allowed here: .*dir1:120'):
            pxt.drop_dir(p('dir1:120'))
        # doesn't exist
        self._test_drop_if_not_exists(p, 'dir2')
        self._test_drop_if_not_exists(p, 'not_a_parent_dir/subdir')
        # not empty
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not empty'):
            pxt.drop_dir(p('dir1'))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be one of'):
            pxt.drop_dir(p('dir1'), if_not_exists='invalid')  # type: ignore[arg-type]
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='needs to be a directory but is a table'):
            pxt.drop_dir(p('t1'))

        pxt.drop_dir(p('dir1/sub1/subsub1'))
        assert pxt.list_dirs(p('dir1/sub1')) == []

        # check after reloading
        reload_catalog()
        assert pxt.list_dirs(p('dir1/sub1')) == []

    def test_drop_force(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('dir1'))
        pxt.create_dir(p('dir2'))
        pxt.create_dir(p('dir1/subdir'))
        pxt.create_dir(p('dir1/subdir/subsub'))

        t = pxt.create_table(p('dir1/subdir/tbl'), {'col': pxt.String})
        v = pxt.create_view(p('dir1/subdir/subsub/v1'), t)
        _ = pxt.create_view(p('dir1/v2'), t)
        _ = pxt.create_view(p('dir2/v3'), t)
        v4 = pxt.create_view(p('dir1/v4'), v)
        _ = pxt.create_view(p('dir1/subdir/subsub/v5'), v4)
        _ = pxt.create_view(p('dir2/v6'), v4)
        assert len(pxt.list_tables(p(''))) == 7
        assert len(pxt.list_dirs(p(''))) == 4
        pxt.drop_dir(p('dir1'), force=True)
        assert len(pxt.list_tables(p(''))) == 0
        assert len(pxt.list_dirs(p(''))) == 1

    def test_move(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('dir1'))
        pxt.create_dir(p('dir1/sub1'))
        make_tbl(p('dir1/sub1/t1'))
        assert pxt.list_tables(p('dir1')) == [p('dir1/sub1/t1')]
        pxt.move(p('dir1/sub1/t1'), p('dir1/sub1/t2'))
        assert pxt.list_tables(p('dir1')) == [p('dir1/sub1/t2')]
        pxt.create_dir(p('dir2'))
        pxt.move(p('dir1'), p('dir2/dir1'))
        assert pxt.list_tables(p('dir2')) == [p('dir2/dir1/sub1/t2')]

        pxt.create_dir(p('dir2/sub1'))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be identical'):
            pxt.move(p('dir2/sub1'), p('dir2/sub1'))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='into its own subdirectory'):
            pxt.create_dir(p('dir2/sub1/subsub1'))
            pxt.move(p('dir2/sub1'), p('dir2/sub1/subsub1'))

        # new client: force loading from store
        reload_catalog()
        assert pxt.list_tables(p('dir2')) == [p('dir2/dir1/sub1/t2')]

    def test_create_with_parents(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        all_dirs = ['dir1', 'dir1/dir2', 'dir1/dir2/dir3']
        pxt.create_dir(p('dir1/dir2/dir3'), parents=True)
        listing = pxt.list_dirs(p(''), recursive=True)
        assert listing == [p(d) for d in all_dirs]

        # create a subdirectory where couple of intermediate parents are missing
        pxt.drop_dir(p('dir1/dir2/dir3'))
        pxt.drop_dir(p('dir1/dir2'))
        pxt.create_dir(p('dir1/dir2/dir3/dir4'), parents=True)
        listing = pxt.list_dirs(p(''), recursive=True)
        all_dirs.append('dir1/dir2/dir3/dir4')
        assert listing == [p(d) for d in all_dirs]
