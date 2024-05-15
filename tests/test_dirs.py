import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from .utils import make_tbl, reload_db


class TestDirs:
    def test_create(self, reset_db) -> None:
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            pxt.create_dir(name)

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
        reload_db()

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
            pxt.rm_dir('1dir')
        # bad path
        with pytest.raises(excs.Error):
            pxt.rm_dir('dir1..sub1')
        # doesn't exist
        with pytest.raises(excs.Error):
            pxt.rm_dir('dir2')
        # not empty
        with pytest.raises(excs.Error):
            pxt.rm_dir('dir1')

        pxt.rm_dir('dir1.sub1.subsub1')
        assert pxt.list_dirs('dir1.sub1') == []

        # check after reloading
        reload_db()
        assert pxt.list_dirs('dir1.sub1') == []

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
        reload_db()
        assert pxt.list_tables('dir2') == ['dir2.dir1.sub1.t2']
