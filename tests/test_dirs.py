import pytest

import pixeltable as pxt

from .utils import make_tbl, reload_catalog


class TestDirs:
    def test_create(self, reset_db: None) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            dir = pxt.create_dir(name)
            assert dir._path() == name
            assert dir._name == name.split('.')[-1]

        # invalid names
        with pytest.raises(pxt.Error, match='Invalid path: 1dir'):
            pxt.create_dir('1dir')
        with pytest.raises(pxt.Error, match='Invalid path: _dir1'):
            pxt.create_dir('_dir1')
        with pytest.raises(pxt.Error, match='Invalid path: dir 1'):
            pxt.create_dir('dir 1')
        with pytest.raises(pxt.Error, match=r'Invalid path: dir1..sub2'):
            pxt.create_dir('dir1..sub2')
        with pytest.raises(pxt.Error, match=r'Invalid path: dir1.sub2.'):
            pxt.create_dir('dir1.sub2.')
        with pytest.raises(pxt.Error, match=r'Invalid path: dir1:sub2.'):
            pxt.create_dir('dir1:sub2.')
        with pytest.raises(pxt.Error, match='Versioned path not allowed here: dir1:120'):
            pxt.create_dir('dir1:120')

        # existing dirs raise error by default
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_dir('dir1')
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_dir('dir1.sub1')
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_dir('dir1.sub1.subsub1')

        # existing table
        make_tbl('dir1.t1')
        _ = pxt.get_table('dir1.t1')
        with pytest.raises(pxt.Error, match='is an existing'):
            pxt.create_dir('dir1.t1')
        with pytest.raises(pxt.Error, match=r'does not exist. Create it first with:'):
            pxt.create_dir('dir2.sub2')
        make_tbl('t2')
        with pytest.raises(pxt.Error, match="Directory 't2' does not exist"):
            pxt.create_dir('t2.sub2')

        # new client: force loading from store
        reload_catalog()

        listing = pxt.get_dir_contents(recursive=True)
        assert listing == {'dirs': ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1'], 'tables': ['dir1.t1', 't2']}
        listing = pxt.get_dir_contents(recursive=False)
        assert listing == {'dirs': ['dir1'], 'tables': ['t2']}
        listing = pxt.get_dir_contents('dir1', recursive=True)
        assert listing == {'dirs': ['dir1.sub1', 'dir1.sub1.subsub1'], 'tables': ['dir1.t1']}
        listing = pxt.get_dir_contents('dir1', recursive=False)
        assert listing == {'dirs': ['dir1.sub1'], 'tables': ['dir1.t1']}
        listing = pxt.get_dir_contents('dir1.sub1', recursive=True)
        assert listing == {'dirs': ['dir1.sub1.subsub1'], 'tables': []}
        listing = pxt.get_dir_contents('dir1.sub1', recursive=False)
        assert listing == {'dirs': ['dir1.sub1.subsub1'], 'tables': []}

    def test_create_if_exists(self, reset_db: None) -> None:
        """Test if_exists parameter of create_dir API"""
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        id_before = {}
        for name in dirs:
            dir = pxt.create_dir(name)
            assert dir._path() == name
            id_before[name] = dir._id

        # invalid if_exists value is rejected
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_dir('dir1', if_exists='invalid')  # type: ignore[arg-type]
        assert (
            "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value).lower()
        )

        # scenrio 1: path already has a directory

        # if_exists='error' should raise error - already tested in test_create

        # if_exists='ignore' should not raise error and return existing Dir
        d1 = pxt.create_dir('dir1', if_exists='ignore')
        assert d1._id == id_before['dir1']
        assert pxt.list_dirs(recursive=True) == dirs
        d2 = pxt.create_dir('dir1.sub1', if_exists='ignore')
        assert d2._id == id_before['dir1.sub1']
        assert pxt.list_dirs(recursive=True) == dirs
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='ignore')
        assert d3._id == id_before['dir1.sub1.subsub1']
        assert pxt.list_dirs(recursive=True) == dirs

        # if_exists='replace' should replace existing Dir, but only if it's empty
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='replace')
        assert d3._id != id_before['dir1.sub1.subsub1']
        id_before['dir1.sub1.subsub1'] = d3._id
        assert pxt.list_dirs(recursive=True) == dirs
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_dir('dir1.sub1', if_exists='replace')
        err_msg = str(exc_info.value).lower()
        assert 'already exists' in err_msg and 'is not empty' in err_msg and 'replace_force' in err_msg
        assert pxt.list_dirs(recursive=True) == dirs
        with pytest.raises(pxt.Error) as exc_info:
            pxt.create_dir('dir1', if_exists='replace')
        err_msg = str(exc_info.value).lower()
        assert 'already exists' in err_msg and 'is not empty' in err_msg and 'replace_force' in err_msg
        assert pxt.list_dirs(recursive=True) == dirs

        # if_exists='replace_force' should replace existing Dir,
        # and all its children
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='replace_force')
        assert d3._id != id_before['dir1.sub1.subsub1']
        assert pxt.list_dirs(recursive=True) == dirs
        d2 = pxt.create_dir('dir1.sub1', if_exists='replace_force')
        assert d2._id != id_before['dir1.sub1']
        assert pxt.list_dirs(recursive=True) == ['dir1', 'dir1.sub1']
        d3 = pxt.create_dir('dir1.sub1.subsub1')
        assert pxt.list_dirs(recursive=True) == dirs
        d1 = pxt.create_dir('dir1', if_exists='replace_force')
        assert d1._id != id_before['dir1']
        assert pxt.list_dirs(recursive=True) == ['dir1']

        # scenrio 2: path already exists but is not a Dir
        make_tbl('dir1.t1')
        for if_exists in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(pxt.Error) as exc_info:
                pxt.create_dir('dir1.t1', if_exists=if_exists)  # type: ignore[arg-type]
            err_msg = str(exc_info.value).lower()
            assert 'already exists' in err_msg and 'not a dir' in err_msg, f' for if_exists={if_exists!r}'

    def _test_drop_if_not_exists(self, dir_name: str) -> None:
        """Test if_not_exists parameter of drop_dir"""
        orig_dirs = pxt.list_dirs(recursive=True)
        # if_not_exists parameter can be used control behavior
        # of drop_dir when the directory doesn't exist.

        # if_not_exists='error' should raise error
        # default behavior is to raise error
        with pytest.raises(pxt.Error, match='does not exist'):
            pxt.drop_dir(dir_name, if_not_exists='error')
        with pytest.raises(pxt.Error, match='does not exist'):
            pxt.drop_dir(dir_name)

        # if_not_exists='ignore' should be successful but a no-op
        pxt.drop_dir(dir_name, if_not_exists='ignore')
        assert pxt.list_dirs(recursive=True) == orig_dirs
        # when force=True, if_not_exists is ignored
        pxt.drop_dir(dir_name, if_not_exists='error', force=True)
        assert pxt.list_dirs(recursive=True) == orig_dirs
        # invalid if_not_exists value is rejected, but only
        # when the directory doesn't exist.
        with pytest.raises(pxt.Error) as exc_info:
            pxt.drop_dir(dir_name, if_not_exists='invalid')  # type: ignore[arg-type]
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

    def test_drop(self, reset_db: None) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            pxt.create_dir(name)
        make_tbl('t1')
        make_tbl('dir1.t1')

        # bad name
        with pytest.raises(pxt.Error, match='Invalid path: 1dir'):
            pxt.drop_dir('1dir')
        # bad path
        with pytest.raises(pxt.Error, match=r'Invalid path: dir1..sub1'):
            pxt.drop_dir('dir1..sub1')
        with pytest.raises(pxt.Error, match='Versioned path not allowed here: dir1:120'):
            pxt.drop_dir('dir1:120')
        # doesn't exist
        self._test_drop_if_not_exists('dir2')
        self._test_drop_if_not_exists('not_a_parent_dir.subdir')
        # not empty
        with pytest.raises(pxt.Error, match='is not empty'):
            pxt.drop_dir('dir1')
        with pytest.raises(pxt.Error, match='must be one of'):
            pxt.drop_dir('dir1', if_not_exists='invalid')  # type: ignore[arg-type]
        with pytest.raises(pxt.Error, match='needs to be a directory but is a table'):
            pxt.drop_dir('t1')

        pxt.drop_dir('dir1.sub1.subsub1')
        assert pxt.list_dirs('dir1.sub1') == []

        # check after reloading
        reload_catalog()
        assert pxt.list_dirs('dir1.sub1') == []

    def test_drop_force(self, reset_db: None) -> None:
        pxt.create_dir('dir1')
        pxt.create_dir('dir2')
        pxt.create_dir('dir1.subdir')
        pxt.create_dir('dir1.subdir.subsub')

        t = pxt.create_table('dir1.subdir.tbl', {'col': pxt.String})
        v = pxt.create_view('dir1.subdir.subsub.v1', t)
        _ = pxt.create_view('dir1.v2', t)
        _ = pxt.create_view('dir2.v3', t)
        v4 = pxt.create_view('dir1.v4', v)
        _ = pxt.create_view('dir1.subdir.subsub.v5', v4)
        _ = pxt.create_view('dir2.v6', v4)
        assert len(pxt.list_tables()) == 7
        assert len(pxt.list_dirs()) == 4
        pxt.drop_dir('dir1', force=True)
        assert len(pxt.list_tables()) == 0
        assert len(pxt.list_dirs()) == 1

    def test_move(self, reset_db: None) -> None:
        pxt.create_dir('dir1')
        pxt.create_dir('dir1.sub1')
        make_tbl('dir1.sub1.t1')
        assert pxt.list_tables('dir1') == ['dir1.sub1.t1']
        pxt.move('dir1.sub1.t1', 'dir1.sub1.t2')
        assert pxt.list_tables('dir1') == ['dir1.sub1.t2']
        pxt.create_dir('dir2')
        pxt.move('dir1', 'dir2.dir1')
        assert pxt.list_tables('dir2') == ['dir2.dir1.sub1.t2']

        pxt.create_dir('dir2.sub1')
        with pytest.raises(pxt.Error, match='cannot be identical'):
            pxt.move('dir2.sub1', 'dir2.sub1')
        with pytest.raises(pxt.Error, match='into its own subdirectory'):
            pxt.create_dir('dir2.sub1.subsub1')
            pxt.move('dir2.sub1', 'dir2.sub1.subsub1')

        # new client: force loading from store
        reload_catalog()
        assert pxt.list_tables('dir2') == ['dir2.dir1.sub1.t2']

    def test_create_with_parents(self, reset_db: None) -> None:
        all_dirs = ['dir1', 'dir1.dir2', 'dir1.dir2.dir3']
        dir3 = pxt.create_dir('dir1.dir2.dir3', parents=True)
        assert dir3._path() == 'dir1.dir2.dir3'
        assert dir3._name == 'dir3'
        listing = pxt.list_dirs(recursive=True)
        assert listing == all_dirs

        # create a subdirectory where couple of intermediate parents are missing
        pxt.drop_dir('dir1.dir2.dir3')
        pxt.drop_dir('dir1.dir2')
        dir4 = pxt.create_dir('dir1.dir2.dir3.dir4', parents=True)
        assert dir4._path() == 'dir1.dir2.dir3.dir4'
        assert dir4._name == 'dir4'
        listing = pxt.list_dirs(recursive=True)
        all_dirs.append('dir1.dir2.dir3.dir4')
        assert listing == all_dirs
