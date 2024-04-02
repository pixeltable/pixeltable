import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.tests.utils import make_tbl


class TestDirs:
    def test_create(self, test_client: pxt.Client) -> None:
        cl = test_client
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            cl.create_dir(name)

        # invalid names
        with pytest.raises(excs.Error):
            cl.create_dir('1dir')
        with pytest.raises(excs.Error):
            cl.create_dir('_dir1')
        with pytest.raises(excs.Error):
            cl.create_dir('dir 1')
        with pytest.raises(excs.Error):
            cl.create_dir('dir1..sub2')
        with pytest.raises(excs.Error):
            cl.create_dir('dir1.sub2.')
        with pytest.raises(excs.Error):
            cl.create_dir('dir1:sub2.')

        # existing dirs
        with pytest.raises(excs.Error):
            cl.create_dir('dir1')
        cl.create_dir('dir1', ignore_errors=True)
        with pytest.raises(excs.Error):
            cl.create_dir('dir1.sub1')
        with pytest.raises(excs.Error):
            cl.create_dir('dir1.sub1.subsub1')

        # existing table
        make_tbl(cl, 'dir1.t1')
        with pytest.raises(excs.Error):
            cl.create_dir('dir1.t1')

        with pytest.raises(excs.Error):
            cl.create_dir('dir2.sub2')
        make_tbl(cl, 't2')
        with pytest.raises(excs.Error):
            cl.create_dir('t2.sub2')

        # new client: force loading from store
        cl2 = pxt.Client(reload=True)

        listing = cl2.list_dirs(recursive=True)
        assert listing == dirs
        listing = cl2.list_dirs(recursive=False)
        assert listing == ['dir1']
        listing = cl2.list_dirs('dir1', recursive=True)
        assert listing == ['dir1.sub1', 'dir1.sub1.subsub1']
        listing = cl2.list_dirs('dir1', recursive=False)
        assert listing == ['dir1.sub1']
        listing = cl2.list_dirs('dir1.sub1', recursive=True)
        assert listing == ['dir1.sub1.subsub1']
        listing = cl2.list_dirs('dir1.sub1', recursive=False)
        assert listing == ['dir1.sub1.subsub1']

    def test_rm(self, test_client: pxt.Client) -> None:
        cl = test_client
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            cl.create_dir(name)
        make_tbl(cl, 't1')
        make_tbl(cl, 'dir1.t1')

        # bad name
        with pytest.raises(excs.Error):
            cl.rm_dir('1dir')
        # bad path
        with pytest.raises(excs.Error):
            cl.rm_dir('dir1..sub1')
        # doesn't exist
        with pytest.raises(excs.Error):
            cl.rm_dir('dir2')
        # not empty
        with pytest.raises(excs.Error):
            cl.rm_dir('dir1')

        cl.rm_dir('dir1.sub1.subsub1')
        assert cl.list_dirs('dir1.sub1') == []

        # check after reloading
        cl = pxt.Client(reload=True)
        assert cl.list_dirs('dir1.sub1') == []

    def test_move(self, test_client: pxt.Client) -> None:
        cl = test_client
        cl.create_dir('dir1')
        cl.create_dir('dir1.sub1')
        make_tbl(cl, 'dir1.sub1.t1')
        assert cl.list_tables('dir1') == ['dir1.sub1.t1']
        cl.move('dir1.sub1.t1', 'dir1.sub1.t2')
        assert cl.list_tables('dir1') == ['dir1.sub1.t2']
        cl.create_dir('dir2')
        cl.move('dir1', 'dir2.dir1')
        assert cl.list_tables('dir2') == ['dir2.dir1.sub1.t2']

        # new client: force loading from store
        cl2 = pxt.Client(reload=True)
        assert cl2.list_tables('dir2') == ['dir2.dir1.sub1.t2']
