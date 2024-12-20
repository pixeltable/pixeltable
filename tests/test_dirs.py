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
        with pytest.raises(excs.Error):
            pxt.create_dir('1dir')
        with pytest.raises(excs.Error):
            pxt.create_dir('_dir1')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir 1')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1..sub2')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1.sub2.')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1:sub2.')

        # existing dirs
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1')
        pxt.create_dir('dir1', ignore_errors=True)
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1.sub1')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1.sub1.subsub1')

        # existing table
        make_tbl('dir1.t1')
        with pytest.raises(excs.Error):
            pxt.create_dir('dir1.t1')

        with pytest.raises(excs.Error):
            pxt.create_dir('dir2.sub2')
        make_tbl('t2')
        with pytest.raises(excs.Error):
            pxt.create_dir('t2.sub2')

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
