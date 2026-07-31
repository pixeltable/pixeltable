import os
import pathlib
import sys
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.service import proxy_daemon, proxy_dispatch, proxy_protocol
from pixeltable.utils.local_store import TempStore

from .utils import pxt_raises


class _RemoteMediaSink(proxy_protocol.PartSink):
    """PartSink that stores media parts in a dict of R2-style object keys, mirroring R2PartSink's contract."""

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}

    def add_media_bytes(self, data: bytes, extension: str) -> str:
        key = f'uploads/req/{len(self.objects)}{extension}'
        self.objects[key] = data
        return key

    def add_media_file(self, path: str) -> str:
        with open(path, 'rb') as f:
            return self.add_media_bytes(f.read(), pathlib.Path(path).suffix)


class TestProxyDaemon:
    def test_pid_alive_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Exercise the POSIX os.kill() path explicitly; on Windows _pid_alive() dispatches elsewhere (see
        # test_pid_alive_dispatches_to_win32), so pin the platform to keep these os.kill cases meaningful everywhere.
        monkeypatch.setattr(proxy_daemon.sys, 'platform', 'linux')

        # ProcessLookupError (no such pid) -> gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
        assert proxy_daemon._pid_alive(99999) is False

        # PermissionError (pid exists, owned by another user) -> alive
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(PermissionError()))
        assert proxy_daemon._pid_alive(1) is True

        # A non-lookup OSError is treated as gone
        monkeypatch.setattr(
            os, 'kill', lambda pid, sig: (_ for _ in ()).throw(OSError(22, 'The parameter is incorrect'))
        )
        assert proxy_daemon._pid_alive(0) is False

        # SystemError is treated as gone
        monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(SystemError()))
        assert proxy_daemon._pid_alive(0) is False

    def test_pid_alive_dispatches_to_win32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # On Windows, _pid_alive() must use the Win32 probe and never call os.kill(), which there maps to
        # TerminateProcess and would kill the very process being probed.
        monkeypatch.setattr(proxy_daemon.sys, 'platform', 'win32')
        monkeypatch.setattr(os, 'kill', lambda pid, sig: pytest.fail('os.kill() must not be called on Windows'))
        monkeypatch.setattr(proxy_daemon, '_win_pid_alive', lambda pid: pid == 4242)
        assert proxy_daemon._pid_alive(4242) is True
        assert proxy_daemon._pid_alive(1) is False

    @pytest.mark.skipif(sys.platform != 'win32', reason='Win32 process probe')
    def test_win_pid_alive_probe(self) -> None:
        # The current process is live; a pid that cannot be opened reads as gone.
        assert proxy_daemon._win_pid_alive(os.getpid()) is True
        assert proxy_daemon._win_pid_alive(0xFFFFFFFE) is False

    @staticmethod
    def _media_args(tmp_path: pathlib.Path) -> dict[str, Any]:
        """A rows payload with one of each binary-bearing value type."""
        src = tmp_path / 'cat.png'
        PIL.Image.new('RGB', (8, 6), color=(1, 2, 3)).save(src, format='PNG')
        mem_img = PIL.Image.new('RGB', (4, 4), color=(9, 8, 7))
        return {
            'rows': [
                {'img_file': proxy_protocol.LocalFile(str(src)), 'img': mem_img, 'data': b'abc', 'arr': np.arange(3)}
            ]
        }

    def test_media_sink_round_trip(self, tmp_path: pathlib.Path) -> None:
        args = self._media_args(tmp_path)
        sink = _RemoteMediaSink()
        wire = proxy_protocol.serialize_args(args, sink)
        row = wire['rows'][0]

        # media parts go out of band as object keys (names/formats preserved); scalar binary parts stay inline
        assert row['img_file'] == {'$pxt': 'file', 'name': 'cat.png', 'v': 'uploads/req/0.png'}
        assert row['img'] == {'$pxt': 'image', 'format': 'PNG', 'v': 'uploads/req/1.png'}
        assert row['data'] == {'$pxt': 'bytes', 'v': 0}
        assert row['arr'] == {'$pxt': 'ndarray', 'v': 1}
        assert proxy_protocol.collect_remote_keys(wire) == ['uploads/req/0.png', 'uploads/req/1.png']

        # deserializing resolves each key through a remote_parts map of pre-downloaded local paths
        remote_parts: dict[str, str] = {}
        for key, data in sink.objects.items():
            local = tmp_path / key.replace('/', '_')
            local.write_bytes(data)
            remote_parts[key] = str(local)
        uploaded_names: dict[str, str] = {}
        result = proxy_protocol._deserialize(wire, sink.binary_parts, uploaded_names, remote_parts)
        out_row = result['rows'][0]
        assert out_row['img_file'] == remote_parts['uploads/req/0.png']
        assert uploaded_names[out_row['img_file']] == 'cat.png'
        assert isinstance(out_row['img'], PIL.Image.Image)
        assert out_row['img'].size == (4, 4)
        assert out_row['data'] == b'abc'
        assert np.array_equal(out_row['arr'], np.arange(3))

        # a remote key without a remote_parts map cannot be localized
        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION):
            proxy_protocol._deserialize(wire, sink.binary_parts, None, None)

    def test_inline_sink_wire_format(self, tmp_path: pathlib.Path) -> None:
        # the default PartSink inlines every binary value as an int-indexed part (the local daemon's wire shape)
        args = self._media_args(tmp_path)
        sink = proxy_protocol.PartSink()
        wire = proxy_protocol.serialize_args(args, sink)
        row = wire['rows'][0]
        assert row['img_file'] == {'$pxt': 'file', 'name': 'cat.png', 'v': 0}
        assert row['img'] == {'$pxt': 'image', 'format': 'PNG', 'v': 1}
        assert row['data'] == {'$pxt': 'bytes', 'v': 2}
        assert row['arr'] == {'$pxt': 'ndarray', 'v': 3}
        assert len(sink.binary_parts) == 4
        assert sink.binary_parts[0] == (tmp_path / 'cat.png').read_bytes()
        assert sink.binary_parts[2] == b'abc'
        assert proxy_protocol.collect_remote_keys(wire) == []

    def test_collect_remote_keys(self) -> None:
        file_tag = {'$pxt': 'file', 'name': 'a.png', 'v': 'uploads/r/0.png'}
        args = {
            # a dir tree: one {relpath, file} entry per file
            'source': [
                {'relpath': 'd/a.png', 'file': file_tag},
                {'relpath': 'd/b.png', 'file': {'$pxt': 'file', 'name': 'b.png', 'v': 'uploads/r/1.png'}},
            ],
            # duplicate references to one key collapse to a single download
            'rows': [{'img': {'$pxt': 'image', 'format': 'PNG', 'v': 'uploads/r/2.png'}, 'dup': dict(file_tag)}],
            # keys inside nested containers are found
            'nested': {'$pxt': 'tuple', 'v': [{'$pxt': 'file', 'name': 'c', 'v': 'uploads/r/3'}]},
            # int-indexed (inline) media and non-media str tags are not remote keys
            'inline': {'$pxt': 'file', 'name': 'd', 'v': 0},
            'not_media': {'$pxt': 'mediapath', 'v': 'uploads/r/9.png'},
        }
        expected = ['uploads/r/0.png', 'uploads/r/1.png', 'uploads/r/2.png', 'uploads/r/3']
        assert proxy_protocol.collect_remote_keys(args) == expected

    def test_prepare_once_on_stale_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.catalog.path import Path as CatalogPath
        from pixeltable.service.proxy_client import ProxyClient

        client = ProxyClient('http://127.0.0.1:1', CatalogPath.parse('pxt://local:somedb', allow_empty_path=True))
        prepare_calls = 0
        orig_prepare = ProxyClient._prepare

        def counting_prepare(self: ProxyClient, args: dict[str, Any]) -> Any:
            nonlocal prepare_calls
            prepare_calls += 1
            return orig_prepare(self, args)

        # a stale-md response makes dispatch_table_method retry the POST without re-serializing (and thus
        # without re-reading/re-uploading media)
        responses = [proxy_protocol.ProxyResponse(is_stale_md=True), proxy_protocol.ProxyResponse(result='ok')]
        monkeypatch.setattr(ProxyClient, '_prepare', counting_prepare)
        monkeypatch.setattr(ProxyClient, '_post', lambda self, *args, **kwargs: responses.pop(0))
        result = client.dispatch_table_method(
            'insert', {'rows': []}, path_key=None, get_snapshot_key=lambda: None, refresh=lambda md: None
        )
        assert result == 'ok'
        assert prepare_calls == 1

    @staticmethod
    def _install_fake_upload_store(
        monkeypatch: pytest.MonkeyPatch, objects: dict[str, bytes], store_uris: list[str]
    ) -> None:
        """Route ObjectOps.get_store to a fake store serving objects (keyed store-relative, i.e. without the
        'uploads/' prefix) and configure the daemon's org/db identity."""
        from pixeltable.utils.object_stores import ObjectOps

        class FakeStore:
            def copy_object_to_local_file(self, src_path: str, dest_path: pathlib.Path) -> None:
                if src_path not in objects:
                    # what a real store raises for a 404 (message blames the bucket)
                    raise excs.NotFoundError(excs.ErrorCode.STORAGE_NOT_FOUND, "Bucket 'b' not found")
                dest_path.write_bytes(objects[src_path])

        def fake_get_store(dest: Any, allow_obj_name: bool, col_name: Any = None) -> Any:
            store_uris.append(dest)
            return FakeStore()

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(fake_get_store))
        monkeypatch.setenv('PIXELTABLE_DAEMON_ORG', 'org1')
        monkeypatch.setenv('PIXELTABLE_DAEMON_DB', 'db1')

    @staticmethod
    def _remote_file_request(*keys: str) -> proxy_protocol.ProxyRequest:
        return proxy_protocol.ProxyRequest(
            class_name='CatalogBase',
            method='echo_test',
            args={'rows': [{'f': {'$pxt': 'file', 'name': f'x{i}', 'v': k}} for i, k in enumerate(keys)]},
        )

    def test_prefetch_remote_parts(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        objects = {'req/0.png': b'png-bytes', 'req/1.jpg': b'jpg-bytes'}
        store_uris: list[str] = []
        self._install_fake_upload_store(monkeypatch, objects, store_uris)

        # happy path: keys download into TempStore, preserving each key's extension
        request = self._remote_file_request('uploads/req/0.png', 'uploads/req/1.jpg')
        proxy_dispatch._prefetch_remote_parts(request)
        assert store_uris == ['pxtfs://org1:db1/home/uploads/']
        assert set(request._remote_parts) == {'uploads/req/0.png', 'uploads/req/1.jpg'}
        for key, path_str in request._remote_parts.items():
            path = pathlib.Path(path_str)
            assert TempStore.contains_path(path)
            assert path.suffix == pathlib.Path(key).suffix
            assert path.read_bytes() == objects[key.removeprefix('uploads/')]
            path.unlink()

        # a request without remote keys makes no store (and thus no control-plane) call
        store_uris.clear()
        proxy_dispatch._prefetch_remote_parts(
            proxy_protocol.ProxyRequest(class_name='CatalogBase', method='echo_test', args={'rows': []})
        )
        assert store_uris == []

        # keys outside uploads/ (e.g. persisted store objects) are rejected before any download
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('pixeltable/data/foo.png'))

        # a missing object is reported as an expired/incomplete upload, naming the key
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match=r'uploads/req/9\.png.*expired'):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('uploads/req/9.png'))

        # without a configured daemon identity, remote keys cannot be localized
        monkeypatch.delenv('PIXELTABLE_DAEMON_ORG')
        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('uploads/req/0.png'))

    def test_handle_cleans_remote_parts(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        objects = {'req/0.png': b'png-bytes'}
        self._install_fake_upload_store(monkeypatch, objects, [])
        localized: list[str] = []

        def echo_handler(request: proxy_protocol.ProxyRequest) -> None:
            args = proxy_protocol.deserialize_request(request)
            localized.append(args['rows'][0]['f'])
            assert pathlib.Path(localized[-1]).exists()

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), echo_handler)

        # success: the handler saw the localized file; handle() unlinked it afterwards
        request = self._remote_file_request('uploads/req/0.png')
        response_json, _ = proxy_dispatch.handle(request.model_dump_json(), [])
        response = proxy_protocol.ProxyResponse.model_validate_json(response_json)
        assert response.error is None
        assert len(localized) == 1
        assert not pathlib.Path(localized[0]).exists()

        # failure: cleanup also runs when the handler raises
        def failing_handler(request: proxy_protocol.ProxyRequest) -> None:
            localized.extend(request._remote_parts.values())
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'boom')

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), failing_handler)
        request = self._remote_file_request('uploads/req/0.png')
        response_json, _ = proxy_dispatch.handle(request.model_dump_json(), [])
        response = proxy_protocol.ProxyResponse.model_validate_json(response_json)
        assert response.error is not None
        assert 'boom' in response.error['message']
        assert len(localized) == 2
        assert not pathlib.Path(localized[1]).exists()
