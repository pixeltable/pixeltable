import pytest

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable.tests.utils import make_tbl
from pixeltable import catalog


class TestDirs:
    def test_create(self, test_db: catalog.Db) -> None:
        db = test_db
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            db.create_dir(name)

        with pytest.raises(exc.BadFormatError):
            db.create_dir('1dir')
        with pytest.raises(exc.BadFormatError):
            db.create_dir('_dir1')
        with pytest.raises(exc.BadFormatError):
            db.create_dir('dir 1')
        with pytest.raises(exc.BadFormatError):
            db.create_dir('dir1..sub2')
        with pytest.raises(exc.BadFormatError):
            db.create_dir('dir1.sub2.')
        with pytest.raises(exc.BadFormatError):
            db.create_dir('dir1:sub2.')

        # existing dirs
        with pytest.raises(exc.DuplicateNameError):
            db.create_dir('dir1')
        with pytest.raises(exc.DuplicateNameError):
            db.create_dir('dir1.sub1')
        with pytest.raises(exc.DuplicateNameError):
            db.create_dir('dir1.sub1.subsub1')

        # existing table
        make_tbl(db, 'dir1.t1')
        with pytest.raises(exc.DuplicateNameError):
            db.create_dir('dir1.t1')

        with pytest.raises(exc.UnknownEntityError):
            db.create_dir('dir2.sub2')
        make_tbl(db, 't2')
        with pytest.raises(exc.UnknownEntityError):
            db.create_dir('t2.sub2')

        # new client: force loading from store
        cl2 = pt.Client()
        db = cl2.get_db('test')

        listing = db.list_dirs(recursive=True)
        assert listing == dirs
        listing = db.list_dirs(recursive=False)
        assert listing == ['dir1']
        listing = db.list_dirs('dir1', recursive=True)
        assert listing == ['dir1.sub1', 'dir1.sub1.subsub1']
        listing = db.list_dirs('dir1', recursive=False)
        assert listing == ['dir1.sub1']
        listing = db.list_dirs('dir1.sub1', recursive=True)
        assert listing == ['dir1.sub1.subsub1']
        listing = db.list_dirs('dir1.sub1', recursive=False)
        assert listing == ['dir1.sub1.subsub1']

    def test_rm(self, test_db: catalog.Db) -> None:
        db = test_db
        dirs = ['dir1', 'dir1.sub1', 'dir1.sub1.subsub1']
        for name in dirs:
            db.create_dir(name)
        make_tbl(db, 't1')
        make_tbl(db, 'dir1.t1')

        with pytest.raises(exc.BadFormatError):
            db.rm_dir('1dir')
        with pytest.raises(exc.BadFormatError):
            db.rm_dir('dir1..sub1')
        with pytest.raises(exc.UnknownEntityError):
            db.rm_dir('dir2')
        with pytest.raises(exc.UnknownEntityError):
            db.rm_dir('t1')

        with pytest.raises(exc.DirectoryNotEmptyError):
            db.rm_dir('dir1')

    def test_rename_tbl(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_dir('dir1')
        make_tbl(db, 'dir1.t1')
        assert db.list_tables('dir1') == ['dir1.t1']
        db.rename_table('dir1.t1', 't2')
        assert db.list_tables('dir1') == ['dir1.t2']
