import pytest

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.runtime import get_runtime


class TestUser:
    def test_user_namespace(self, uses_db: None) -> None:
        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir/subdir')
        t = pxt.create_table('test_dir/test_tbl', {'col': pxt.Int})
        _ = pxt.create_table('test_dir/subdir/test_tbl', {'col': pxt.Int})
        t.insert(col=5)

        get_runtime().catalog.create_user('marcel')
        Env.get().user = 'marcel'
        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir/subdir')
        marcel_t = pxt.create_table('test_dir/test_tbl', {'col': pxt.Int})
        _ = pxt.create_table('test_dir/subdir/test_tbl', {'col': pxt.Int})
        marcel_t.insert(col=22)

        get_runtime().catalog.create_user('asiegel')
        Env.get().user = 'asiegel'
        pxt.create_dir('test_dir')
        pxt.create_dir('test_dir/subdir')
        asiegel_t = pxt.create_table('test_dir/test_tbl', {'col': pxt.Int})
        _ = pxt.create_table('test_dir/subdir/test_tbl', {'col': pxt.Int})
        asiegel_t.insert(col=4171780)

        assert t.select().collect()['col'] == [5]
        assert marcel_t.select().collect()['col'] == [22]
        assert asiegel_t.select().collect()['col'] == [4171780]

        # Table is dropped from correct userspace
        pxt.drop_table('test_dir/test_tbl')
        assert t.select().collect()['col'] == [5]
        assert marcel_t.select().collect()['col'] == [22]
        with pytest.raises(pxt.Error, match='Table was dropped'):
            asiegel_t.select().collect()

        # get_table operates over correct userspace
        Env.get().user = None
        assert pxt.get_table('test_dir/test_tbl').select().collect()['col'] == [5]

        Env.get().user = 'marcel'
        assert pxt.get_table('test_dir/test_tbl').select().collect()['col'] == [22]

        Env.get().user = 'asiegel'
        with pytest.raises(pxt.Error, match=r"Path 'test_dir/test_tbl' does not exist."):
            pxt.get_table('test_dir/test_tbl').select().collect()

        # Directory is dropped from correct userspace
        pxt.drop_dir('test_dir', force=True)
        assert t.select().collect()['col'] == [5]
        assert marcel_t.select().collect()['col'] == [22]

        # Unknown user
        Env.get().user = 'pbrunelle'
        with pytest.raises(pxt.Error, match='Unknown user: pbrunelle'):
            pxt.create_dir('test_dir')
