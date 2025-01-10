import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from .utils import make_tbl, reload_catalog


class TestDirs:
    def test_create(self, reset_db) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            dir = pxt.create_dir(name)
            assert dir._path == name
            assert dir._name == name.split('.')[-1]
            assert dir._parent._path == '.'.join(name.split('.')[:-1])

        # invalid names
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('1dir')
        assert 'Invalid path format' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('_dir1')
        assert 'Invalid path format' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir 1')
        assert 'Invalid path format' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1..sub2')
        assert 'Invalid path format' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1.sub2.')
        assert 'Invalid path format' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1:sub2.')
        assert 'Invalid path format' in str(exc_info.value)

        # existing dirs raise error by default
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1')
        assert 'already exists' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1.sub1')
        assert 'already exists' in str(exc_info.value)
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1.sub1.subsub1')
        assert 'already exists' in str(exc_info.value)

        # existing table
        make_tbl('dir1.t1')
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1.t1')
        assert 'already exists' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir2.sub2')
        assert 'No such path' in str(exc_info.value)
        make_tbl('t2')
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('t2.sub2')
        assert 'Not a directory' in str(exc_info.value)

        # new client: force loading from store
        reload_catalog()

        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        listing = pxt.list_dirs(recursive=False)
        assert listing == ['dir1']
        listing = pxt.list_dirs('dir1', recursive=True)
        assert listing == ['dir1.sub1', 'dir1.sub1.subsub1']
        listing = pxt.list_dirs('dir1', recursive=False)
        assert listing == ['dir1.sub1']
        listing = pxt.list_dirs('dir1.sub1', recursive=True)
        assert listing == ['dir1.sub1.subsub1']
        listing = pxt.list_dirs('dir1.sub1', recursive=False)
        assert listing == ['dir1.sub1.subsub1']

    def test_create_if_exists(self, reset_db) -> None:
        """Test if_exists parameter of create_dir API"""
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        id_before = {}
        for name in dirs:
            dir = pxt.create_dir(name)
            assert dir._path == name
            id_before[name] = dir._id

        # invalid if_exists value is rejected
        with pytest.raises(excs.Error) as exc_info:
            pxt.create_dir('dir1', if_exists='invalid')
        assert "if_exists must be one of: ['error', 'ignore', 'replace', 'replace_force']" in str(exc_info.value)

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
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_dir('dir1.sub1', if_exists='replace')
        assert ('already exists' in str(exc_info.value)
            and 'has dependents' in str(exc_info.value)
            and 'replace_force' in str(exc_info.value))
        assert pxt.list_dirs(recursive=True) == dirs
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_dir('dir1', if_exists='replace')
        assert ('already exists' in str(exc_info.value)
            and 'has dependents' in str(exc_info.value)
            and 'replace_force' in str(exc_info.value))
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
        for _ie in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(excs.Error) as exc_info:
                pxt.create_dir('dir1.t1', if_exists=_ie)
            assert ('already exists' in str(exc_info.value)
                and 'not a Dir' in str(exc_info.value)), f" for if_exists='{_ie}'"

    def _test_drop_if_not_exists(self, dir_name: str) -> None:
        """ Test if_not_exists parameter of drop_dir """
        orig_dirs = pxt.list_dirs(recursive=True)
        # if_not_exists parameter can be used control behavior
        # of drop_dir when the directory doesn't exist.

        # if_not_exists='error' should raise error
        # default behavior is to raise error
        with pytest.raises(excs.Error, match=r'does not exist'):
            pxt.drop_dir(dir_name, if_not_exists='error')
        with pytest.raises(excs.Error, match=r'does not exist'):
            pxt.drop_dir(dir_name)

        # if_not_exists='ignore' should be successful but a no-op
        pxt.drop_dir(dir_name, if_not_exists='ignore')
        assert pxt.list_dirs(recursive=True) == orig_dirs
        # when force=True, if_not_exists is ignored
        pxt.drop_dir(dir_name, if_not_exists='error', force=True)
        assert pxt.list_dirs(recursive=True) == orig_dirs
        # invalid if_not_exists value is rejected, but only
        # when the directory doesn't exist.
        with pytest.raises(excs.Error) as exc_info:
            pxt.drop_dir(dir_name, if_not_exists='invalid')
        assert "if_not_exists must be one of: ['error', 'ignore']" in str(exc_info.value).lower()

    def test_drop(self, reset_db) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            pxt.create_dir(name)
        make_tbl('t1')
        make_tbl('dir1.t1')

        # bad name
        with pytest.raises(excs.Error, match=r'Invalid path format'):
            pxt.drop_dir('1dir')
        # bad path
        with pytest.raises(excs.Error, match=r'Invalid path format'):
            pxt.drop_dir('dir1..sub1')
        # doesn't exist
        self._test_drop_if_not_exists('dir2')
        # not empty
        with pytest.raises(excs.Error, match=r'is not empty'):
            pxt.drop_dir('dir1')
        with pytest.raises(excs.Error, match=r'is not empty'):
            pxt.drop_dir('dir1', if_not_exists='invalid')
        with pytest.raises(excs.Error, match=r"needs to be a directory but is a table"):
            pxt.drop_dir('t1')

        pxt.drop_dir('dir1.sub1.subsub1')
        assert pxt.list_dirs('dir1.sub1') == []

        # check after reloading
        reload_catalog()
        assert pxt.list_dirs('dir1.sub1') == []

    def test_drop_force(self, reset_db) -> None:
        pxt.create_dir('dir1')
        pxt.create_dir('dir2')
        pxt.create_dir('dir1.subdir')
        pxt.create_dir('dir1.subdir.subsub')
        # Create lots of views all over the place
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

    def test_move(self, reset_db) -> None:
        pxt.create_dir('dir1')
        pxt.create_dir('dir1.sub1')
        make_tbl('dir1.sub1.t1')
        assert pxt.list_tables('dir1') == ['dir1.sub1.t1']
        pxt.move('dir1.sub1.t1', 'dir1.sub1.t2')
        assert pxt.list_tables('dir1') == ['dir1.sub1.t2']
        pxt.create_dir('dir2')
        pxt.move('dir1', 'dir2.dir1')
        assert pxt.list_tables('dir2') == ['dir2.dir1.sub1.t2']

        # new client: force loading from store
        reload_catalog()
        assert pxt.list_tables('dir2') == ['dir2.dir1.sub1.t2']
