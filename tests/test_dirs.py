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
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('1dir')
        assert 'Invalid path format' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('_dir1')
        assert 'Invalid path format' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir 1')
        assert 'Invalid path format' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1..sub2')
        assert 'Invalid path format' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1.sub2.')
        assert 'Invalid path format' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1:sub2.')
        assert 'Invalid path format' in str(exc.value)

        # existing dirs raise error by default
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1')
        assert 'already exists' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1.sub1')
        assert 'already exists' in str(exc.value)
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1.sub1.subsub1')
        assert 'already exists' in str(exc.value)

        # existing table
        make_tbl('dir1.t1')
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir1.t1')
        assert 'already exists' in str(exc.value)

        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('dir2.sub2')
        assert 'No such path' in str(exc.value)
        make_tbl('t2')
        with pytest.raises(excs.Error) as exc:
            pxt.create_dir('t2.sub2')
        assert 'not a directory' in str(exc.value)

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
        """Test if_exists parameter of create_dir"""
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        id_before = {}
        for name in dirs:
            dir = pxt.create_dir(name)
            assert dir._path == name
            id_before[name] = dir._id

        # scenrio 1: path already has a directory

        # if_exists='error' should raise error - already tested in test_create

        # if_exists='ignore' should not raise error and return existing Dir
        d1 = pxt.create_dir('dir1', if_exists='ignore')
        assert d1._id == id_before['dir1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        d2 = pxt.create_dir('dir1.sub1', if_exists='ignore')
        assert d2._id == id_before['dir1.sub1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='ignore')
        assert d3._id == id_before['dir1.sub1.subsub1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs

        # if_exists='replace' should replace existing Dir, but only if it's empty
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='replace')
        assert d3._id != id_before['dir1.sub1.subsub1']
        id_before['dir1.sub1.subsub1'] = d3._id
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        with pytest.raises(excs.Error) as exc:
            _ = pxt.create_dir('dir1.sub1', if_exists='replace')
        assert ('already exists' in str(exc.value)
            and 'not empty' in str(exc.value))
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        with pytest.raises(excs.Error) as exc:
            _ = pxt.create_dir('dir1', if_exists='replace')
        assert ('already exists' in str(exc.value)
            and 'not empty' in str(exc.value))
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs

        # if_exists='replace_force' should replace existing Dir,
        # and all its children
        d3 = pxt.create_dir('dir1.sub1.subsub1', if_exists='replace_force')
        assert d3._id != id_before['dir1.sub1.subsub1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        d2 = pxt.create_dir('dir1.sub1', if_exists='replace_force')
        assert d2._id != id_before['dir1.sub1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == ['dir1', 'dir1.sub1']
        d3 = pxt.create_dir('dir1.sub1.subsub1')
        listing = pxt.list_dirs(recursive=True)
        assert listing == dirs
        d1 = pxt.create_dir('dir1', if_exists='replace_force')
        assert d1._id != id_before['dir1']
        listing = pxt.list_dirs(recursive=True)
        assert listing == ['dir1']

        # scenrio 3: path already exists but is not a Dir
        make_tbl('dir1.t1')
        for _ie in ['ignore', 'replace', 'replace_force']:
            with pytest.raises(excs.Error) as exc:
                pxt.create_dir('dir1.t1', if_exists=_ie)
            assert ('already exists' in str(exc.value)
                and 'not a Dir' in str(exc.value)), f" for if_exists='{_ie}'"

    def test_rm(self, reset_db) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            pxt.create_dir(name)
        make_tbl('t1')
        make_tbl('dir1.t1')

        # bad name
        with pytest.raises(excs.Error):
            pxt.drop_dir('1dir')
        # bad path
        with pytest.raises(excs.Error):
            pxt.drop_dir('dir1..sub1')
        # doesn't exist
        with pytest.raises(excs.Error):
            pxt.drop_dir('dir2')
        # not empty
        with pytest.raises(excs.Error):
            pxt.drop_dir('dir1')

        pxt.drop_dir('dir1.sub1.subsub1')
        assert pxt.list_dirs('dir1.sub1') == []

        # check after reloading
        reload_catalog()
        assert pxt.list_dirs('dir1.sub1') == []

    def test_rm_force(self, reset_db) -> None:
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
