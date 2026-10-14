import gc
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.catalog.catalog_base import CatalogBase
from pixeltable.catalog.path import Path as CatalogPath
from pixeltable.runtime import Runtime, get_runtime
from pixeltable.service import proxy_daemon
from pixeltable.service.proxy_client import ProxyClient

from .utils import DatabaseRoot, reload_catalog


class TestRuntime:
    @pytest.mark.db_roots('proxy', reason="checks the connections to the local proxy daemon's port")
    def test_proxy_connections_closed_with_runtime(
        self, db_root: DatabaseRoot, proxy_daemon_db: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every connection to the daemon belongs to a live runtime, which closes it. A table handle or a cursor from
        before a reset uses the current runtime's connections, and a thread's runtime closes its own when the thread
        ends."""
        port_lock = proxy_daemon.read_port_lock(proxy_daemon_db)
        assert port_lock is not None
        daemon_addr = f"raddr=('127.0.0.1', {port_lock['port']})"
        img = tmp_path / 'img.png'
        PIL.Image.new('RGB', (8, 6)).save(img)
        tbl_path = db_root.make_catalog_path('tbl')
        t = pxt.create_table(tbl_path, {'a': pxt.Int, 'img': pxt.Image})
        t.insert(a=1, img=str(img))
        cursor = t.select(t.img).cursor()

        # a local daemon's media is fetched outside the client's pool, so the cursor's client is checked directly
        fetched_by: list[ProxyClient] = []
        fetch_media = ProxyClient.fetch_media

        def recording_fetch_media(client: ProxyClient, urls: list[str]) -> dict[str, str]:
            fetched_by.append(client)
            return fetch_media(client, urls)

        monkeypatch.setattr(ProxyClient, 'fetch_media', recording_fetch_media)

        gc.collect()  # so that only sockets left open below are collected
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ResourceWarning)
            reload_catalog()
            t.insert(a=2, img=str(img))  # dispatched by a handle from before the reset
            with ThreadPoolExecutor(max_workers=1) as pool:  # a thread that ends without closing its runtime
                pool.submit(t.insert, a=3, img=str(img)).result()
            [row] = cursor  # opened after the reset
            assert isinstance(row['img'], PIL.Image.Image)
            assert fetched_by == [get_runtime().get_proxy_client(CatalogPath.parse(tbl_path))]
            assert t.count() == 3
            del t, cursor, row
            reload_catalog()
            gc.collect()
        assert [str(w.message) for w in caught if daemon_addr in str(w.message)] == []

    def test_close_catalogs_closes_only_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A runtime that copied another's catalogs closes only the catalogs it created, and leaves the copied ones to
        the runtime they came from."""
        catalogs = {'db1': Mock(spec=CatalogBase), 'db2': Mock(spec=CatalogBase)}
        monkeypatch.setattr(Runtime, '_make_proxy_catalog', lambda self, catalog_uri: catalogs[catalog_uri.db])
        parent, worker = Runtime(), Runtime()
        parent.get_catalog(CatalogPath.parse('pxt://local:db1/t'))
        worker.copy_db_context(parent)
        worker.get_catalog(CatalogPath.parse('pxt://local:db1/t'))
        worker.get_catalog(CatalogPath.parse('pxt://local:db2/t'))

        worker.close_catalogs()
        catalogs['db1'].close.assert_not_called()
        catalogs['db2'].close.assert_called_once()
        parent.close_catalogs()
        catalogs['db1'].close.assert_called_once()

    def test_close_catalogs_continues_past_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A catalog whose close() raises does not keep the runtime from closing its other catalogs."""
        catalogs = {'db1': Mock(spec=CatalogBase), 'db2': Mock(spec=CatalogBase)}
        catalogs['db1'].close.side_effect = OSError('connection reset')
        monkeypatch.setattr(Runtime, '_make_proxy_catalog', lambda self, catalog_uri: catalogs[catalog_uri.db])
        runtime = Runtime()
        runtime.get_catalog(CatalogPath.parse('pxt://local:db1/t'))
        runtime.get_catalog(CatalogPath.parse('pxt://local:db2/t'))

        runtime.close_catalogs()
        catalogs['db2'].close.assert_called_once()
