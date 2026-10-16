import gc
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

import pixeltable as pxt
from pixeltable.service import proxy_daemon

from .utils import DatabaseRoot, reload_catalog


class TestRuntime:
    @pytest.mark.db_roots('proxy', reason="checks the connections to the local proxy daemon's port")
    def test_proxy_connections_closed_with_runtime(self, db_root: DatabaseRoot, proxy_daemon_db: str) -> None:
        """A reset or a thread exit closes the discarded runtime's connections to the daemon."""
        port_lock = proxy_daemon.read_port_lock(proxy_daemon_db)
        assert port_lock is not None
        daemon_addr = f"raddr=('127.0.0.1', {port_lock['port']})"
        tbl_path = db_root.make_catalog_path('tbl')
        t = pxt.create_table(tbl_path, {'a': pxt.Int})
        t.insert(a=1)

        gc.collect()  # so that only sockets left open below are collected
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ResourceWarning)
            reload_catalog()
            t.insert(a=2)  # dispatched by a handle from before the reset
            with ThreadPoolExecutor(max_workers=1) as pool:  # a thread that ends without closing its runtime
                pool.submit(lambda: pxt.get_table(tbl_path).insert(a=3)).result()
            assert t.count() == 3
            del t
            reload_catalog()
            gc.collect()
        assert [str(w.message) for w in caught if daemon_addr in str(w.message)] == []
