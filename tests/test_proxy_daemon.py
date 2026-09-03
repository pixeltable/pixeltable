import json
import pathlib
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.service import proxy_dispatch, proxy_protocol
from pixeltable.service.proxy_client import HttpTransport, ProxyClient, PxtStorePartSink, TunnelTransport
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import FileDestination, ObjectOps

from .utils import pxt_raises


class _RemoteMediaSink(proxy_protocol.PartSink[str]):
    """PartSink that stores media parts in a dict of object store-style object keys,
    mirroring PxtStorePartSink's contract."""

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
        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION, match='has no access to uploaded objects'):
            proxy_protocol._deserialize(wire, sink.binary_parts, None, None)

    def test_inline_sink_wire_format(self, tmp_path: pathlib.Path) -> None:
        # InlinePartSink inlines every binary value as an int-indexed part (the local daemon's wire shape)
        args = self._media_args(tmp_path)
        sink = proxy_protocol.InlinePartSink()
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
        client = ProxyClient.local('http://127.0.0.1:1')
        prepare_calls = 0
        orig_prepare = ProxyClient._prepare

        def counting_prepare(self: ProxyClient, args: dict[str, Any]) -> Any:
            nonlocal prepare_calls
            prepare_calls += 1
            return orig_prepare(self, args)

        # a stale-md response makes dispatch_table_method retry the POST without re-serializing (and thus
        # without re-reading/re-uploading media)
        # _post() hands back the response head with the body's binary parts
        responses: list[tuple[proxy_protocol.ProxyResponse, list[bytes]]] = [
            (proxy_protocol.ProxyResponse(is_stale_md=True), []),
            (proxy_protocol.ProxyResponse(result='ok'), []),
        ]
        monkeypatch.setattr(ProxyClient, '_prepare', counting_prepare)
        monkeypatch.setattr(ProxyClient, '_post', lambda self, *args, **kwargs: responses.pop(0))
        result = client.dispatch_table_method(
            'insert', {'rows': []}, path_key=None, get_snapshot_key=lambda: None, refresh=lambda md: None
        )
        assert result == 'ok'
        assert prepare_calls == 1

    def test_transport_part_sinks(self) -> None:
        # media parts travel inline to a local daemon, but out of band (via R2) to a hosted db
        local_sink = HttpTransport('http://127.0.0.1:1').new_part_sink()
        assert type(local_sink) is proxy_protocol.InlinePartSink

        tunnel = TunnelTransport('org1', 'db1', 'key', host='h', port=443)
        remote_sink = tunnel.new_part_sink()
        next_sink = tunnel.new_part_sink()
        assert isinstance(remote_sink, PxtStorePartSink)
        assert isinstance(next_sink, PxtStorePartSink)
        # each request gets its own uploads/ prefix
        assert next_sink._key_prefix != remote_sink._key_prefix

    def test_pxt_store_sink_defers_uploads(
        self, init_env: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PxtStorePartSink mints keys while serializing and performs every upload in flush()."""
        uploaded: dict[str, tuple[pathlib.Path, bytes]] = {}
        store_uris: list[str] = []

        class FakeStore:
            def copy_local_file(self, src_path: pathlib.Path, dest: FileDestination) -> str:
                assert dest.remote_key is not None
                uploaded[dest.remote_key] = (src_path, src_path.read_bytes())
                return dest.url

        def fake_get_store(dest: Any, allow_obj_name: bool, col_name: Any = None) -> Any:
            store_uris.append(dest)
            return FakeStore()

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(fake_get_store))

        src = tmp_path / 'cat.png'
        PIL.Image.new('RGB', (8, 6), color=(1, 2, 3)).save(src, format='PNG')
        sink = PxtStorePartSink('org1', 'db1')
        # the same path twice, plus an in-memory value (which stages a temp file)
        keys = [sink.add_media_file(str(src)), sink.add_media_file(str(src)), sink.add_media_bytes(b'raw', '.jpg')]

        # nothing has been uploaded yet, and no credentials have been fetched
        assert uploaded == {}
        assert store_uris == []
        # repeated references to one path get distinct keys (the daemon consumes each localized file)
        assert len(set(keys)) == 3
        assert all(k.startswith(sink._key_prefix) for k in keys)

        sink.flush()
        # one store (one credential fetch) for the whole request, scoped to its own prefix
        assert store_uris == [f'pxtfs://org1:db1/home/{sink._key_prefix}']
        assert set(uploaded) == set(keys)
        assert uploaded[keys[0]][1] == uploaded[keys[1]][1] == src.read_bytes()
        assert uploaded[keys[2]][1] == b'raw'

        # the file staged for the in-memory value was removed after its upload; the caller's own file was not
        staged = uploaded[keys[2]][0]
        assert TempStore.contains_path(staged)
        assert not staged.exists()
        assert uploaded[keys[0]][0] == src
        assert src.exists()

        # flush() drained the queue, so a second call uploads nothing
        store_uris.clear()
        sink.flush()
        assert store_uris == []

    @staticmethod
    def _install_fake_upload_store(
        monkeypatch: pytest.MonkeyPatch, objects: dict[str, bytes], store_uris: list[str]
    ) -> None:
        """Route ObjectOps.get_store to a fake store serving objects (keyed store-relative, i.e. without the
        'uploads/' prefix) and put the daemon's org/db identity in the environment."""
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
        monkeypatch.setenv('PXTCLOUD_ORG', 'org1')
        monkeypatch.setenv('PXTCLOUD_DB', 'db1')

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
        with pxt_raises(
            pxt.ErrorCode.INVALID_ARGUMENT, match=r"Invalid uploaded media object key: 'pixeltable/data/foo\.png'"
        ):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('pixeltable/data/foo.png'))

        # a missing object is reported as an expired/incomplete upload, naming the key
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match=r'uploads/req/9\.png.*expired'):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('uploads/req/9.png'))

        # without the container's org/db in the environment, remote keys cannot be localized
        monkeypatch.delenv('PXTCLOUD_ORG')
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match=r'Internal error: PXTCLOUD_ORG and PXTCLOUD_DB are not present in the container.',
        ):
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
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        assert json.loads(head).get('error') is None
        assert len(localized) == 1
        assert not pathlib.Path(localized[0]).exists()

        # failure: cleanup also runs when the handler raises
        def failing_handler(request: proxy_protocol.ProxyRequest) -> None:
            localized.extend(request._remote_parts.values())
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'boom')

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), failing_handler)
        request = self._remote_file_request('uploads/req/0.png')
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        error = json.loads(head)['error']
        assert 'boom' in error['message']
        assert len(localized) == 2
        assert not pathlib.Path(localized[1]).exists()
