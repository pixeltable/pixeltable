import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation

from .coordinator import MultiThreadedScenario
from .fault_injection import BlockFault
from .utils import pxt_raises


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
        with pxt_raises(pxt.ErrorCode.TABLE_NOT_FOUND, match='Table was dropped'):
            asiegel_t.select().collect()

        # get_table operates over correct userspace
        Env.get().user = None
        assert pxt.get_table('test_dir/test_tbl').select().collect()['col'] == [5]

        Env.get().user = 'marcel'
        assert pxt.get_table('test_dir/test_tbl').select().collect()['col'] == [22]

        Env.get().user = 'asiegel'
        with pxt_raises(pxt.ErrorCode.PATH_NOT_FOUND, match=r"Path 'test_dir/test_tbl' does not exist."):
            pxt.get_table('test_dir/test_tbl').select().collect()

        # Directory is dropped from correct userspace
        pxt.drop_dir('test_dir', force=True)
        assert t.select().collect()['col'] == [5]
        assert marcel_t.select().collect()['col'] == [22]

        # Unknown user
        Env.get().user = 'pbrunelle'
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Unknown user: pbrunelle'):
            pxt.create_dir('test_dir')

    def test_create_user_concurrent(self, uses_db: None, fault_injection: None) -> None:
        """
        Repro for PXT-1183: two processes/threads creating the root dir for the same user concurrently must not
        produce duplicate root dirs.
        """
        fault = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_CREATE_USER_AFTER_EXISTS_CHECK, fault=fault)
            # Thread 0: create_user() blocks after checking that the root dir doesn't exist
            .then_run_until(
                thread_id=0,
                name='create_user (blocks)',
                event=fault.reached,
                fn=lambda: get_runtime().catalog.create_user('user1'),
            )
            # Thread 1: create_user() for the same user
            .then_run(thread_id=1, name='create_user (wins)', fn=lambda: get_runtime().catalog.create_user('user1'))
            # Thread 1: unblock thread 0, which now hits a serialization failure and retries
            .then_run(thread_id=1, name='unblock thread 0', fn=lambda: fault.unblock())
            .execute()
        )

        # verify user1 has a functioning catalog
        Env.get().user = 'user1'
        assert pxt.list_dirs() == []
