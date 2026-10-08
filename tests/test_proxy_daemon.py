import http
import io
import json
import math
import pathlib
import socket
import socketserver
import ssl
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import numpy as np
import PIL.Image
import pytest
import requests

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.catalog import TablePathKey, TableVersionKey
from pixeltable.catalog.path import Path as CatalogPath
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.service import proxy_client, proxy_daemon, proxy_dispatch, proxy_protocol
from pixeltable.service.management_client import Credential
from pixeltable.service.proxy_client import HttpTransport, ProxyClient, TunnelTransport
from pixeltable.service.proxy_protocol import ArchiveMember, PxtArchivePartSink, PxtStorePartSink
from pixeltable.utils import cloud_utils
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import FileDestination, ObjectOps, StorageTarget

from .utils import pxt_raises, reload_env

_KEY = Credential('api_key', 'key', 'the PIXELTABLE_API_KEY environment variable')


@pytest.fixture
def hosted_identity(init_env: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give the Env the daemon's org/db identity for the test, and take it back afterwards."""
    monkeypatch.setenv('PXTCLOUD_ORG', 'org1')
    monkeypatch.setenv('PXTCLOUD_DB', 'db1')
    reload_env()
    yield
    monkeypatch.delenv('PXTCLOUD_ORG', raising=False)
    monkeypatch.delenv('PXTCLOUD_DB', raising=False)
    reload_env()


class _RemotePartSink(proxy_protocol.PartSink[int | str]):
    """PartSink that stores out-of-band parts in a dict of object store-style object keys,
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

    def add_scalar_bytes(self, data: bytes, extension: str) -> int | str:
        if len(data) < proxy_protocol.PxtStorePartSink._MIN_OUT_OF_BAND_SIZE:
            return self.add_inline(data)
        return self.add_media_bytes(data, extension)


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _sparse_tar_bytes(name: str, size: int) -> bytes:
    """A tar whose one member stores a single byte but extracts to size bytes (GNU sparse format 0.1)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        info = tarfile.TarInfo(name)
        info.size = 1
        info.pax_headers = {'GNU.sparse.map': '0,1', 'GNU.sparse.size': str(size)}
        tf.addfile(info, io.BytesIO(b'x'))
    return buf.getvalue()


def _tar_members(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        members: dict[str, bytes] = {}
        for info in tf.getmembers():
            f = tf.extractfile(info)
            assert f is not None
            members[info.name] = f.read()
        return members


class _RecordingHttpTransport(httpx.BaseTransport):
    """Records the headers of each request and the pieces its body stream yields, and answers 'ok'."""

    def __init__(self) -> None:
        self.requests: list[tuple[httpx.Headers, list[bytes]]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.SyncByteStream)
        self.requests.append((request.headers, [bytes(piece) for piece in request.stream]))
        return httpx.Response(200, content=b'ok')


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
        sink = _RemotePartSink()
        wire = proxy_protocol.serialize_args(args, sink)
        row = wire['rows'][0]

        # media parts go out of band as object keys (names/formats preserved); small scalars stay inline
        assert row['img_file'] == {'$pxt': 'file', 'name': 'cat.png', 'v': 'uploads/req/0.png'}
        assert row['img'] == {'$pxt': 'image', 'format': 'PNG', 'v': 'uploads/req/1.png'}
        assert row['data'] == {'$pxt': 'bytes', 'v': 0}
        assert row['arr'] == {'$pxt': 'ndarray', 'v': 1}
        assert proxy_protocol.collect_remote_keys(wire) == [('uploads/req/0.png', None), ('uploads/req/1.png', None)]

        # deserializing resolves each key through a remote_parts map of pre-downloaded local paths
        remote_parts: dict[tuple[str, str | None], str] = {}
        for key, data in sink.objects.items():
            local = tmp_path / key.replace('/', '_')
            local.write_bytes(data)
            remote_parts[key, None] = str(local)
        uploaded_names: dict[str, str] = {}
        result = proxy_protocol._deserialize(wire, sink.binary_parts, uploaded_names, remote_parts)
        out_row = result['rows'][0]
        assert out_row['img_file'] == remote_parts['uploads/req/0.png', None]
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

    def test_scalars_reach_a_handler_from_the_object_store(
        self, hosted_identity: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end on the daemon side: prefetch localizes an uploaded scalar, dispatch decodes it."""
        arr = np.arange(64, dtype=np.float32)
        npy = io.BytesIO()
        np.save(npy, arr, allow_pickle=False)
        objects = {'req/0.bin': b'z' * 1024, 'req/1.npy': npy.getvalue()}
        self._install_fake_upload_store(monkeypatch, objects, [])
        seen: list[Any] = []

        def echo_handler(request: proxy_protocol.ProxyRequest) -> None:
            seen.append(proxy_protocol.deserialize_request(request))

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), echo_handler)
        request = proxy_protocol.ProxyRequest(
            class_name='CatalogBase',
            method='echo_test',
            args={
                'blob': {'$pxt': 'bytes', 'v': 'uploads/req/0.bin'},
                'arr': {'$pxt': 'ndarray', 'v': 'uploads/req/1.npy'},
            },
        )
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        assert json.loads(head).get('error') is None
        assert seen[0]['blob'] == objects['req/0.bin']
        assert np.array_equal(seen[0]['arr'], arr)
        # handle() removed the files it localized for the request
        assert not any(pathlib.Path(p).exists() for p in request._remote_parts.values())

    def test_response_round_trip(self) -> None:
        """The generic response path preserves every value it is given."""
        tbl_id = uuid.uuid4()
        result = {
            'nan': math.nan,
            'inf': math.inf,
            'pair': (1, 2),
            'int_keys': {1: 'x'},
            'reserved': {'$pxt': 'UUID', 'v': 'not-a-uuid'},
            'id': tbl_id,
            'data': b'abc',
        }
        body = proxy_protocol.encode_response({'result': result})
        head, parts = proxy_protocol.decode_body(body)
        decoded = proxy_protocol.deserialize_value(json.loads(head)['result'], parts)

        assert math.isnan(decoded['nan'])
        assert decoded['inf'] == math.inf
        assert decoded['pair'] == (1, 2)
        assert decoded['int_keys'] == {1: 'x'}
        assert decoded['reserved'] == result['reserved']
        assert decoded['id'] == tbl_id
        assert decoded['data'] == b'abc'

    def test_iter_body_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A body is sliced, without copying, into pieces of at most _BODY_CHUNK_SIZE bytes."""
        monkeypatch.setattr(proxy_protocol, '_BODY_CHUNK_SIZE', 4)
        for body, sizes in ((b'', []), (b'abcdefgh', [4, 4]), (b'abcdefghij', [4, 4, 2])):
            chunks = list(proxy_protocol.iter_body_chunks(body))
            assert [len(chunk) for chunk in chunks] == sizes
            assert all(chunk.obj is body for chunk in chunks)
            assert b''.join(chunks) == body

    def test_http_transport_posts_in_slices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A request body goes to the socket one slice at a time, framed by its full Content-Length."""
        monkeypatch.setattr(proxy_protocol, '_BODY_CHUNK_SIZE', 4)
        recorder = _RecordingHttpTransport()
        transport = HttpTransport('http://daemon')
        transport._http = httpx.Client(base_url='http://daemon', transport=recorder)

        assert transport.post(b'abcdefghij') == b'ok'
        [(headers, pieces)] = recorder.requests
        assert headers['Content-Length'] == '10'
        assert 'Transfer-Encoding' not in headers
        assert pieces == [b'abcd', b'efgh', b'ij']

    def test_rpc_response_goes_out_in_slices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The daemon writes an /rpc response one slice at a time, framed by its full Content-Length."""
        pytest.importorskip('fastapi')
        from starlette.testclient import TestClient
        from starlette.types import Message, Receive, Scope, Send

        monkeypatch.setattr(proxy_protocol, '_BODY_CHUNK_SIZE', 4)
        monkeypatch.setattr(proxy_dispatch, 'handle', lambda *args, **kwargs: b'abcdefghij')
        app = proxy_daemon._build_app()
        pieces: list[bytes] = []

        async def recording_app(scope: Scope, receive: Receive, send: Send) -> None:
            async def record(message: Message) -> None:
                # the server makes one socket write per body message
                if message['type'] == 'http.response.body':
                    pieces.append(bytes(message['body']))
                await send(message)

            await app(scope, receive, record)

        response = TestClient(recording_app).post('/rpc', content=proxy_protocol.encode_body(b'{}', []))
        assert response.headers['Content-Length'] == '10'
        assert 'Transfer-Encoding' not in response.headers
        assert response.content == b'abcdefghij'
        assert pieces == [b'abcd', b'efgh', b'ij', b'']

    def test_main_address_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The address a cloud pod names on the command line is what the daemon serves on."""
        served: list[tuple[bool, str | None, int | None]] = []
        monkeypatch.setattr(proxy_daemon.Config, 'init', classmethod(lambda cls, **kwargs: None))
        monkeypatch.setattr(
            proxy_daemon, '_serve', lambda test_mode=False, host=None, port=None: served.append((test_mode, host, port))
        )

        proxy_daemon.main(['--test', '--host', '0.0.0.0', '--port', '8000'])
        assert served == [(True, '0.0.0.0', 8000)]

        # without them the daemon picks its own port and publishes it in the lock file
        served.clear()
        proxy_daemon.main([])
        assert served == [(False, None, None)]

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
            # scalars that outgrew the inline threshold carry keys of their own
            'blob': {'$pxt': 'bytes', 'v': 'uploads/r/4.bin'},
            'arr': {'$pxt': 'ndarray', 'v': 'uploads/r/5.npy'},
            # archived parts are identified by their archive's key and their member names
            'archived': [
                {'$pxt': 'image', 'format': 'PNG', 'v': 'uploads/r/tar0.tar', 'member': '6.png'},
                {'$pxt': 'bytes', 'v': 'uploads/r/tar0.tar', 'member': '7.bin'},
            ],
            # int-indexed (inline) parts and str tags that are not parts at all are not remote keys
            'inline': {'$pxt': 'file', 'name': 'd', 'v': 0},
            'inline_blob': {'$pxt': 'bytes', 'v': 1},
            'not_a_part': {'$pxt': 'mediapath', 'v': 'uploads/r/9.png'},
        }
        expected = [
            ('uploads/r/0.png', None),
            ('uploads/r/1.png', None),
            ('uploads/r/2.png', None),
            ('uploads/r/3', None),
            ('uploads/r/4.bin', None),
            ('uploads/r/5.npy', None),
            ('uploads/r/tar0.tar', '6.png'),
            ('uploads/r/tar0.tar', '7.bin'),
        ]
        assert proxy_protocol.collect_remote_keys(args) == expected

        # a member name that is not a string is the client's error
        bad_member = {'$pxt': 'file', 'name': 'a', 'v': 'uploads/r/tar0.tar', 'member': ['0.png']}
        with pxt_raises(
            pxt.ErrorCode.INVALID_ARGUMENT, match='Invalid archive member name: expected a string, got list'
        ):
            proxy_protocol.collect_remote_keys({'rows': [bad_member]})

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

        tunnel = TunnelTransport('org1', 'db1', lambda: _KEY, host='h', port=443)
        remote_sink = tunnel.new_part_sink()
        next_sink = tunnel.new_part_sink()
        assert type(remote_sink) is PxtArchivePartSink
        assert type(next_sink) is PxtArchivePartSink
        # each request gets its own uploads/ prefix
        assert next_sink._key_prefix != remote_sink._key_prefix

    def test_pxt_store_sink_defers_uploads(
        self, init_env: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PxtStorePartSink mints keys while serializing and performs every upload in flush().

        _ResponseMedia uses this per-object sink, since it presigns a url for each key."""
        uploaded: dict[str, tuple[pathlib.Path, bytes]] = {}
        stores: list[tuple[str, bool]] = []

        class FakeStore:
            def copy_local_file(self, src_path: pathlib.Path, dest: FileDestination) -> str:
                assert dest.remote_key is not None
                uploaded[dest.remote_key] = (src_path, src_path.read_bytes())
                return dest.url

        def fake_get_store(
            dest: Any, allow_obj_name: bool, col_name: Any = None, scope_credentials: bool = False
        ) -> Any:
            stores.append((dest, scope_credentials))
            return FakeStore()

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(fake_get_store))

        src = tmp_path / 'cat.png'
        PIL.Image.new('RGB', (8, 6), color=(1, 2, 3)).save(src, format='PNG')
        sink = PxtStorePartSink('org1', 'db1')
        # the same path twice, plus an in-memory value (which stages a temp file)
        keys: list[str] = []
        for key in (sink.add_media_file(str(src)), sink.add_media_file(str(src)), sink.add_media_bytes(b'raw', '.jpg')):
            assert isinstance(key, str)
            keys.append(key)

        # nothing has been uploaded yet, and no credentials have been fetched
        assert uploaded == {}
        assert stores == []
        # repeated references to one path get distinct keys (the daemon consumes each localized file)
        assert len(set(keys)) == 3
        assert all(k.startswith(sink._key_prefix) for k in keys)

        sink.flush()
        # one store for the whole request, with credentials for its own prefix
        assert stores == [(f'pxtfs://org1:db1/home/{sink._key_prefix}', True)]
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
        stores.clear()
        sink.flush()
        assert stores == []

    @staticmethod
    def _install_fake_upload_store(
        monkeypatch: pytest.MonkeyPatch,
        objects: dict[str, bytes],
        stores: list[tuple[str, bool]],
        downloads: list[str] | None = None,
    ) -> None:
        """Route ObjectOps.get_store to a fake store serving objects (keyed store-relative, i.e. without the
        'uploads/' prefix). Uploads are stored in objects, and each download's key is appended to downloads."""
        from pixeltable.utils.object_stores import ObjectOps

        class FakeStore:
            def copy_local_file(self, src_path: pathlib.Path, dest: FileDestination) -> str:
                assert dest.remote_key is not None and dest.remote_key.startswith('uploads/')
                objects[dest.remote_key.removeprefix('uploads/')] = src_path.read_bytes()
                return dest.url

            def copy_object_to_local_file(self, src_path: str, dest_path: pathlib.Path) -> None:
                if downloads is not None:
                    downloads.append(src_path)
                if src_path not in objects:
                    # what a real store raises for a 404 (message blames the bucket)
                    raise excs.NotFoundError(excs.ErrorCode.STORAGE_NOT_FOUND, "Bucket 'b' not found")
                dest_path.write_bytes(objects[src_path])

        def fake_get_store(
            dest: Any, allow_obj_name: bool, col_name: Any = None, scope_credentials: bool = False
        ) -> Any:
            stores.append((dest, scope_credentials))
            return FakeStore()

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(fake_get_store))

    @staticmethod
    def _remote_file_request(*parts: str | tuple[str, str]) -> proxy_protocol.ProxyRequest:
        """A request with a 'file' tag for each part: an object key, or an (archive key, member name) pair."""
        rows: list[dict[str, Any]] = []
        for i, part in enumerate(parts):
            tag: dict[str, Any] = {'$pxt': 'file', 'name': f'x{i}'}
            if isinstance(part, str):
                tag['v'] = part
            else:
                tag['v'], tag['member'] = part
            rows.append({'f': tag})
        return proxy_protocol.ProxyRequest(class_name='CatalogBase', method='echo_test', args={'rows': rows})

    def test_prefetch_remote_parts(self, hosted_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
        objects = {'req/0.png': b'png-bytes', 'req/1.jpg': b'jpg-bytes'}
        stores: list[tuple[str, bool]] = []
        self._install_fake_upload_store(monkeypatch, objects, stores)

        # happy path: keys download into TempStore, preserving each key's extension
        request = self._remote_file_request('uploads/req/0.png', 'uploads/req/1.jpg')
        proxy_dispatch._prefetch_remote_parts(request)
        assert stores == [('pxtfs://org1:db1/home/uploads/', False)]
        assert set(request._remote_parts) == {('uploads/req/0.png', None), ('uploads/req/1.jpg', None)}
        for (key, _), path_str in request._remote_parts.items():
            path = pathlib.Path(path_str)
            assert TempStore.contains_path(path)
            assert path.suffix == pathlib.Path(key).suffix
            assert path.read_bytes() == objects[key.removeprefix('uploads/')]
            path.unlink()

        # a request without remote keys makes no store (and thus no control-plane) call
        stores.clear()
        proxy_dispatch._prefetch_remote_parts(
            proxy_protocol.ProxyRequest(class_name='CatalogBase', method='echo_test', args={'rows': []})
        )
        assert stores == []

        # keys outside uploads/ (e.g. persisted store objects) are rejected before any download
        with pxt_raises(
            pxt.ErrorCode.INVALID_ARGUMENT, match=r"Invalid uploaded object key: 'pixeltable/data/foo\.png'"
        ):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('pixeltable/data/foo.png'))

        # a missing object is reported as an expired/incomplete upload, naming the key
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match=r'uploads/req/9\.png.*expired'):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('uploads/req/9.png'))

        # without the container's org/db in the environment, remote keys cannot be localized
        monkeypatch.delenv('PXTCLOUD_ORG')
        reload_env()
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match=r'Internal error: PXTCLOUD_ORG and PXTCLOUD_DB are not present in the container.',
        ):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request('uploads/req/0.png'))

    @pytest.mark.parametrize(
        'part', ['uploads/req/0.png', ('uploads/req/tar0.tar', '0.png')], ids=['object', 'archive_member']
    )
    def test_handle_cleans_remote_parts(
        self, hosted_identity: None, monkeypatch: pytest.MonkeyPatch, part: str | tuple[str, str]
    ) -> None:
        objects = {'req/0.png': b'png-bytes', 'req/tar0.tar': _tar_bytes({'0.png': b'png-bytes'})}
        self._install_fake_upload_store(monkeypatch, objects, [])
        localized: list[str] = []

        def echo_handler(request: proxy_protocol.ProxyRequest) -> None:
            args = proxy_protocol.deserialize_request(request)
            localized.append(args['rows'][0]['f'])
            assert pathlib.Path(localized[-1]).read_bytes() == b'png-bytes'

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), echo_handler)

        # success: the handler saw the localized file; handle() unlinked it afterwards
        request = self._remote_file_request(part)
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        assert json.loads(head).get('error') is None
        assert len(localized) == 1
        assert not pathlib.Path(localized[0]).exists()

        # failure: cleanup also runs when the handler raises
        def failing_handler(request: proxy_protocol.ProxyRequest) -> None:
            localized.extend(request._remote_parts.values())
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'boom')

        monkeypatch.setitem(proxy_dispatch._HANDLERS, ('CatalogBase', 'echo_test'), failing_handler)
        request = self._remote_file_request(part)
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        error = json.loads(head)['error']
        assert 'boom' in error['message']
        assert len(localized) == 2
        assert not pathlib.Path(localized[1]).exists()

    def test_archive_sink_packs_parts(
        self, init_env: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PxtArchivePartSink packs parts into archives that roll over at the target size; a large part is
        uploaded as an object of its own."""
        objects: dict[str, bytes] = {}
        stores: list[tuple[str, bool]] = []
        self._install_fake_upload_store(monkeypatch, objects, stores)
        monkeypatch.setattr(PxtArchivePartSink, '_ARCHIVE_TARGET_SIZE', 4096)
        monkeypatch.setattr(PxtArchivePartSink, '_MAX_ARCHIVE_MEMBER_SIZE', 2048)
        small = tmp_path / 'small.png'
        small.write_bytes(b's' * 1000)
        large = tmp_path / 'large.mp4'
        large.write_bytes(b'L' * 3000)
        tmp_count = TempStore.count()

        sink = PxtArchivePartSink('org1', 'db1')
        prefix = sink._key_prefix
        refs: list[int | str | ArchiveMember] = [sink.add_media_file(str(small)), sink.add_media_file(str(small))]
        # the open archive is below its target size, so nothing is uploaded and no credentials are fetched
        assert objects == {}
        assert stores == []
        # this member takes the archive past 4096 bytes, which closes it
        refs.append(sink.add_media_bytes(b'b' * 1500, '.jpg'))
        refs.append(sink.add_media_file(str(large)))
        refs.append(sink.add_scalar_bytes(b'x' * 600, '.bin'))
        refs.append(sink.add_scalar_bytes(b'tiny', '.bin'))
        sink.flush()

        assert refs == [
            ArchiveMember(f'{prefix}tar0.tar', '0.png'),
            # repeated references to one path get members of their own (the daemon consumes each localized file)
            ArchiveMember(f'{prefix}tar0.tar', '1.png'),
            ArchiveMember(f'{prefix}tar0.tar', '2.jpg'),
            f'{prefix}3.mp4',
            ArchiveMember(f'{prefix}tar1.tar', '4.bin'),
            0,
        ]
        assert stores == [(f'pxtfs://org1:db1/home/{prefix}', True)]
        rel_prefix = prefix.removeprefix('uploads/')
        assert set(objects) == {f'{rel_prefix}tar0.tar', f'{rel_prefix}tar1.tar', f'{rel_prefix}3.mp4'}
        assert _tar_members(objects[f'{rel_prefix}tar0.tar']) == {
            '0.png': small.read_bytes(),
            '1.png': small.read_bytes(),
            '2.jpg': b'b' * 1500,
        }
        assert _tar_members(objects[f'{rel_prefix}tar1.tar']) == {'4.bin': b'x' * 600}
        assert objects[f'{rel_prefix}3.mp4'] == large.read_bytes()
        assert sink.binary_parts == [b'tiny']

        # the caller's files are untouched, and every archive was removed from TempStore
        assert small.exists() and large.exists()
        assert TempStore.count() == tmp_count

    def test_archive_sink_abort(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A serialization failure before any archive fills up uploads nothing and leaves nothing in TempStore."""
        objects: dict[str, bytes] = {}
        self._install_fake_upload_store(monkeypatch, objects, [])
        monkeypatch.setattr(PxtArchivePartSink, '_MAX_ARCHIVE_MEMBER_SIZE', 2048)
        tmp_count = TempStore.count()

        sink = PxtArchivePartSink('org1', 'db1')
        # a large value staged for its own upload, a small one in the open archive, then a value that cannot be
        # serialized
        args = {'large': b'x' * 3000, 'small': b'y' * 1000, 'bad': object()}
        with pytest.raises(AssertionError, match='cannot serialize object'):
            proxy_protocol.serialize_args(args, sink)
        assert objects == {}
        assert TempStore.count() == tmp_count

    @staticmethod
    def _wait_for_temp_count(expected: int) -> None:
        """Wait for background uploads to remove their local files."""
        deadline = time.monotonic() + 10
        while TempStore.count() != expected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert TempStore.count() == expected

    def test_failures_do_not_wait_for_running_uploads(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure reaches the caller while an upload is still running, so it does not hold up a Ctrl+C; the
        running upload removes its own local file when it finishes."""
        started, release = threading.Event(), threading.Event()

        class BlockingStore:
            def copy_local_file(self, src_path: pathlib.Path, dest: FileDestination) -> str:
                assert dest.remote_key is not None
                if dest.remote_key.endswith('.fail'):
                    started.wait(5)
                    raise RuntimeError('upload failed')
                started.set()
                release.wait(30)
                return dest.url

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(lambda *args, **kwargs: BlockingStore()))
        tmp_count = TempStore.count()

        # the archive sink: every member closes its archive, so an upload is running when serialization fails
        monkeypatch.setattr(PxtArchivePartSink, '_ARCHIVE_TARGET_SIZE', 1)
        # releases the upload after 5 s; if abort() waited for it, serialize_args() would return only after that
        watchdog = threading.Timer(5, release.set)
        watchdog.start()
        with pytest.raises(AssertionError, match='cannot serialize object'):
            proxy_protocol.serialize_args({'small': b'y' * 1000, 'bad': object()}, PxtArchivePartSink('org1', 'db1'))
        assert started.is_set() and not release.is_set()
        watchdog.cancel()
        release.set()
        self._wait_for_temp_count(tmp_count)

        # PxtStorePartSink: one upload fails while another is running
        started.clear()
        release.clear()
        sink = PxtStorePartSink('org1', 'db1')
        sink.add_media_bytes(b'x' * 1000, '.slow')
        sink.add_media_bytes(b'x' * 1000, '.fail')
        watchdog = threading.Timer(5, release.set)
        watchdog.start()
        with pytest.raises(RuntimeError, match='upload failed'):
            sink.flush()
        assert not release.is_set()
        watchdog.cancel()
        release.set()
        self._wait_for_temp_count(tmp_count)

    def test_sinks_clean_up_after_failed_store_setup(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A credential fetch that fails in flush() leaves nothing in TempStore."""

        def failing_get_store(
            dest: Any, allow_obj_name: bool, col_name: Any = None, scope_credentials: bool = False
        ) -> Any:
            raise RuntimeError('credential fetch failed')

        monkeypatch.setattr(ObjectOps, 'get_store', staticmethod(failing_get_store))
        monkeypatch.setattr(PxtArchivePartSink, '_MAX_ARCHIVE_MEMBER_SIZE', 2048)
        tmp_count = TempStore.count()

        # the archive is closed but never queued for upload; flush() removes it
        archive_sink = PxtArchivePartSink('org1', 'db1')
        archive_sink.add_media_bytes(b'y' * 1000, '.bin')
        with pytest.raises(RuntimeError, match='credential fetch failed'):
            archive_sink.flush()
        assert TempStore.count() == tmp_count

        # the staged file stays queued, so abort() can remove it
        store_sink = PxtStorePartSink('org1', 'db1')
        store_sink.add_media_bytes(b'x' * 3000, '.bin')
        with pytest.raises(RuntimeError, match='credential fetch failed'):
            store_sink.flush()
        store_sink.abort()
        assert TempStore.count() == tmp_count

    def test_prefetch_archive_parts(self, hosted_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
        objects = {
            'req/tar0.tar': _tar_bytes({'0.png': b'a', '1.png': b'b'}),
            'req/tar1.tar': _tar_bytes({'2.jpg': b'c'}),
            'req/3.bin': b'd',
            'req/bad.tar': b'not a tar file',
            'req/sparse.tar': _sparse_tar_bytes('0.bin', 50 * 1024 * 1024),
        }
        stores: list[tuple[str, bool]] = []
        downloads: list[str] = []
        self._install_fake_upload_store(monkeypatch, objects, stores, downloads)
        tmp_count = TempStore.count()

        request = self._remote_file_request(
            ('uploads/req/tar0.tar', '0.png'),
            ('uploads/req/tar0.tar', '1.png'),
            ('uploads/req/tar1.tar', '2.jpg'),
            'uploads/req/3.bin',
            ('uploads/req/tar0.tar', '0.png'),
        )
        proxy_dispatch._prefetch_remote_parts(request)
        # one download per archive, however many of its members are referenced
        assert sorted(downloads) == ['req/3.bin', 'req/tar0.tar', 'req/tar1.tar']
        assert stores == [('pxtfs://org1:db1/home/uploads/', False)]
        expected = {
            ('uploads/req/tar0.tar', '0.png'): b'a',
            ('uploads/req/tar0.tar', '1.png'): b'b',
            ('uploads/req/tar1.tar', '2.jpg'): b'c',
            ('uploads/req/3.bin', None): b'd',
        }
        assert set(request._remote_parts) == set(expected)
        for (key, member), path_str in request._remote_parts.items():
            path = pathlib.Path(path_str)
            assert TempStore.contains_path(path)
            assert path.suffix == pathlib.PurePosixPath(member or key).suffix
            assert path.read_bytes() == expected[key, member]
            path.unlink()
        # the downloaded archives were removed after extraction
        assert TempStore.count() == tmp_count

        # an archive key outside uploads/ is rejected before any download
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"Invalid uploaded object key: 'pixeltable/data/x\.tar'"):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request(('pixeltable/data/x.tar', '0.png')))

        # a missing archive is reported as an expired/incomplete upload
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match=r'uploads/req/tar9\.tar.*expired'):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request(('uploads/req/tar9.tar', '0.png')))

        # a member absent from its archive
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match=r"'uploads/req/tar0\.tar' is missing member '7\.png'"):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request(('uploads/req/tar0.tar', '7.png')))

        # an object that is not a tar file
        with pxt_raises(pxt.ErrorCode.INVALID_DATA_FORMAT, match=r'bad\.tar.*not a valid tar file'):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request(('uploads/req/bad.tar', '0.png')))

        # a sparse member, which would extract to 50 MB from a 10 KB archive
        with pxt_raises(pxt.ErrorCode.INVALID_DATA_FORMAT, match=r"member '0\.bin' is not a regular, non-sparse file"):
            proxy_dispatch._prefetch_remote_parts(self._remote_file_request(('uploads/req/sparse.tar', '0.bin')))

        # the failed requests left no archive and no extracted member behind
        assert TempStore.count() == tmp_count

    def test_archive_round_trip_through_prefetch(
        self, hosted_identity: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Serialize with PxtArchivePartSink, localize on the daemon side, and decode."""
        objects: dict[str, bytes] = {}
        self._install_fake_upload_store(monkeypatch, objects, [])
        # every member closes its archive, so the request spans several archives
        monkeypatch.setattr(PxtArchivePartSink, '_ARCHIVE_TARGET_SIZE', 1)
        args = self._media_args(tmp_path)
        big_arr = np.arange(256, dtype=np.float64)
        args['rows'][0] |= {'blob': b'z' * 1024, 'big_arr': big_arr}

        sink = PxtArchivePartSink('org1', 'db1')
        wire = proxy_protocol.serialize_args(args, sink)
        parts = proxy_protocol.collect_remote_keys(wire)
        assert len(parts) == 4
        assert all(member is not None for _, member in parts)
        assert len(objects) == 4

        request = proxy_protocol.ProxyRequest(class_name='CatalogBase', method='echo_test', args=wire)
        request._binary_parts = sink.binary_parts
        proxy_dispatch._prefetch_remote_parts(request)
        row = proxy_protocol.deserialize_request(request)['rows'][0]
        assert pathlib.Path(row['img_file']).read_bytes() == (tmp_path / 'cat.png').read_bytes()
        assert request._uploaded_names[row['img_file']] == 'cat.png'
        assert isinstance(row['img'], PIL.Image.Image)
        assert row['img'].size == (4, 4)
        assert row['data'] == b'abc'
        assert np.array_equal(row['arr'], np.arange(3))
        assert row['blob'] == b'z' * 1024
        assert np.array_equal(row['big_arr'], big_arr)
        for path_str in request._remote_parts.values():
            pathlib.Path(path_str).unlink()

    def test_exclamation_mark_in_file_names(
        self, hosted_identity: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file name whose suffix contains '!' reaches the daemon intact, archived or uploaded on its own."""
        objects: dict[str, bytes] = {}
        self._install_fake_upload_store(monkeypatch, objects, [])
        monkeypatch.setattr(PxtArchivePartSink, '_MAX_ARCHIVE_MEMBER_SIZE', 2048)
        large = tmp_path / 'movie.mp4!cut'
        large.write_bytes(b'L' * 3000)
        small = tmp_path / 'clip.tar!x'
        small.write_bytes(b's' * 1000)

        sink = PxtArchivePartSink('org1', 'db1')
        args = {
            'rows': [{'large': proxy_protocol.LocalFile(str(large)), 'small': proxy_protocol.LocalFile(str(small))}]
        }
        wire = proxy_protocol.serialize_args(args, sink)
        assert wire['rows'][0]['large'] == {
            '$pxt': 'file',
            'name': 'movie.mp4!cut',
            'v': f'{sink._key_prefix}0.mp4!cut',
        }
        assert wire['rows'][0]['small']['member'] == '1.tar!x'

        request = proxy_protocol.ProxyRequest(class_name='CatalogBase', method='echo_test', args=wire)
        proxy_dispatch._prefetch_remote_parts(request)
        row = proxy_protocol.deserialize_request(request)['rows'][0]
        assert pathlib.Path(row['large']).read_bytes() == large.read_bytes()
        assert pathlib.Path(row['small']).read_bytes() == small.read_bytes()
        for path_str in request._remote_parts.values():
            pathlib.Path(path_str).unlink()

    def test_protocol_mismatch_tells_a_newer_client_to_rebuild(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = self._protocol_error(proxy_protocol.PROTOCOL_VERSION + 1, 'org1', monkeypatch)
        assert 'pxt db build-image pxt://org1:db1' in message
        assert 'upgrade a lockfile pin first' in message
        assert 'pxt db restart does not change the image' in message
        assert 'pip install --upgrade pixeltable' not in message
        assert 'pxt localproxy' not in message

    def test_protocol_mismatch_tells_a_newer_client_to_restart_local_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = self._protocol_error(proxy_protocol.PROTOCOL_VERSION + 1, 'local', monkeypatch)
        assert 'pxt localproxy stop db1, then pxt localproxy start db1' in message
        assert 'pxt db build-image' not in message
        assert 'pip install --upgrade pixeltable' not in message

    @pytest.mark.parametrize('org', ['org1', 'local'])
    def test_protocol_mismatch_tells_an_older_client_to_upgrade(
        self, org: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = self._protocol_error(proxy_protocol.PROTOCOL_VERSION - 1, org, monkeypatch)
        assert message.endswith('pip install --upgrade pixeltable')
        assert 'pxt db build-image' not in message
        assert 'pxt localproxy' not in message

    @pytest.mark.parametrize('org', ['org1', 'local'])
    @pytest.mark.parametrize('table_method', [False, True])
    @pytest.mark.parametrize(
        'server_version', [proxy_protocol.PROTOCOL_VERSION - 1, proxy_protocol.PROTOCOL_VERSION + 1]
    )
    def test_client_expands_a_legacy_protocol_mismatch(
        self, org: str, table_method: bool, server_version: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client_version = proxy_protocol.PROTOCOL_VERSION
        legacy = f'Unsupported proxy protocol version: {client_version} (server expects {server_version})'
        client = (
            ProxyClient.local('http://127.0.0.1:1', db='db1')
            if org == 'local'
            else ProxyClient.remote(org, 'db1', lambda: _KEY, host='h', port=443)
        )
        response = proxy_protocol.encode_response(
            {
                'error': {
                    'error_code': 'UNSUPPORTED_OPERATION',
                    'message': legacy,
                    'retryable': False,
                    'retry_after': 3.0,
                    'detail': 'test-only diagnostic',
                }
            }
        )
        monkeypatch.setattr(client._transport, 'post', lambda body: response)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Unsupported proxy protocol version') as err:
                self._client_call(client, table_method)
            message = err.value.message
            assert err.value.retry_after == 3.0
            assert err.value.detail == 'test-only diagnostic'
            if server_version > client_version:
                assert message.endswith('pip install --upgrade pixeltable')
            elif org == 'local':
                assert 'pxt localproxy stop db1, then pxt localproxy start db1' in message
                assert 'pxt db build-image' not in message
            else:
                assert 'pxt db build-image pxt://org1:db1' in message
                assert 'pxt localproxy' not in message
        finally:
            client.close()

    @pytest.mark.parametrize('table_method', [False, True])
    @pytest.mark.parametrize(
        'suffix', ['. Follow the newer server recovery instructions.', '\nNew recovery instructions.']
    )
    def test_client_preserves_enriched_protocol_mismatch(
        self, table_method: bool, suffix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = f'Unsupported proxy protocol version: 6 (server expects 7){suffix}'
        client = ProxyClient.local('http://127.0.0.1:1', db='db1')
        response = proxy_protocol.encode_response(
            {
                'error': {
                    'error_code': 'UNSUPPORTED_OPERATION',
                    'message': message,
                    'retryable': False,
                    'client_protocol_version': 6,
                    'server_protocol_version': 4,
                }
            }
        )
        monkeypatch.setattr(client._transport, 'post', lambda body: response)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Unsupported proxy protocol version') as err:
                self._client_call(client, table_method)
            assert err.value.message == message
        finally:
            client.close()

    @pytest.mark.parametrize('table_method', [False, True])
    def test_client_trusts_protocol_version_fields(self, table_method: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sentence says the database is newer. The fields say this client is, and they win."""
        sentence = 'Unsupported proxy protocol version: 1 (server expects 2)'
        client = ProxyClient.remote('org1', 'db1', lambda: _KEY, host='h', port=443)
        response = proxy_protocol.encode_response(
            {
                'error': {
                    'error_code': 'UNSUPPORTED_OPERATION',
                    'message': sentence,
                    'retryable': False,
                    'client_protocol_version': proxy_protocol.PROTOCOL_VERSION,
                    'server_protocol_version': proxy_protocol.PROTOCOL_VERSION - 1,
                }
            }
        )
        monkeypatch.setattr(client._transport, 'post', lambda body: response)
        try:
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Unsupported proxy protocol version') as err:
                self._client_call(client, table_method)
            assert 'pxt db build-image pxt://org1:db1' in err.value.message
            assert 'pip install --upgrade pixeltable' not in err.value.message
        finally:
            client.close()

    @staticmethod
    def _client_call(client: ProxyClient, table_method: bool) -> None:
        if table_method:
            key = TablePathKey((TableVersionKey(uuid.uuid4(), None),))
            client.dispatch_table_method(
                'insert', {'rows': []}, path_key=key, get_snapshot_key=lambda: key, refresh=lambda md: None
            )
        else:
            client.send_request('Catalog', 'list_dirs', {})

    def _protocol_error(self, client_version: int, org: str, monkeypatch: pytest.MonkeyPatch) -> str:
        values = {
            ('pxtcloud', 'org'): org if org != 'local' else None,
            ('pxtcloud', 'db'): 'db1',
            ('pixeltable', 'db'): 'db1',
        }
        monkeypatch.setattr(
            Config.get(), 'get_string_value', lambda key, section='pixeltable': values.get((section, key))
        )
        request = proxy_protocol.ProxyRequest(
            class_name='Catalog', method='list_dirs', args={}, protocol_version=client_version
        )
        head, _parts = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), []))
        error = json.loads(head)['error']
        assert error['error_code'] == 'UNSUPPORTED_OPERATION'
        assert error['client_protocol_version'] == client_version
        assert error['server_protocol_version'] == proxy_protocol.PROTOCOL_VERSION
        return error['message']


class _ScriptedResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body


class _ScriptedConn:
    """A tunnel connection whose write and read phases do what the script says.

    Its socket is a real one, so the pool's own health check runs against it: an open socketpair reads as
    a live connection, and closing the peer makes it read as one the server has closed.
    """

    responded_at: float | None = None

    def __init__(self, on_write: BaseException | None = None, on_read: object = (200, b'ok')) -> None:
        self.sock, self._peer = socket.socketpair()
        self._on_write = on_write
        self._on_read = on_read
        self.writes = 0
        self.closed = False

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> None:
        self.writes += 1
        if self._on_write is not None:
            raise self._on_write

    def getresponse(self) -> _ScriptedResponse:
        if isinstance(self._on_read, BaseException):
            raise self._on_read
        assert isinstance(self._on_read, tuple)
        return _ScriptedResponse(*self._on_read)

    def close(self) -> None:
        self.closed = True
        self.sock.close()
        self._peer.close()

    def close_peer(self) -> None:
        """Make the connection read as closed by the server."""
        self._peer.close()


class _PlainTLS:
    """An SSL context that leaves the socket as it is, for a sidecar that speaks no TLS."""

    def wrap_socket(self, sock: socket.socket, server_hostname: str | None = None) -> socket.socket:
        return sock


def _header_fields(stream: io.BufferedIOBase) -> dict[str, str]:
    """The header fields of the next message on the stream, after its first line."""
    stream.readline()
    fields: dict[str, str] = {}
    while (line := stream.readline().decode().strip()) != '':
        name, _, value = line.partition(':')
        fields[name] = value.strip()
    return fields


class _PlainSidecar(socketserver.TCPServer):
    """A sidecar on loopback TCP that records the token in each tunnel handshake.

    It answers one request per tunnel and then closes it, so the client's next request needs a new
    handshake. Handshakes get `handshakes` in turn, then 'PXT/1.0 200 OK'.
    """

    def __init__(self, handshakes: tuple[bytes, ...] = ()) -> None:
        self.tokens: list[str] = []
        self.handshakes = list(handshakes)
        self._create_connection = socket.create_connection
        super().__init__(('127.0.0.1', 0), _SidecarHandler)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def connect(self, _address: Any, timeout: float | None = None) -> socket.socket:
        return self._create_connection(('127.0.0.1', self.server_address[1]), timeout=timeout)

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()


class _SidecarHandler(socketserver.StreamRequestHandler):
    server: _PlainSidecar

    def handle(self) -> None:
        self.server.tokens.append(_header_fields(self.rfile)['Authorization'].removeprefix('Bearer '))
        reply = self.server.handshakes.pop(0) if self.server.handshakes else b'PXT/1.0 200 OK'
        self.wfile.write(reply + b'\r\n\r\n')
        if not reply.startswith(b'PXT/1.0 200'):
            return
        self.rfile.read(int(_header_fields(self.rfile).get('Content-Length', '0')))
        self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok')


class TestTunnelRetries:
    """What the client reissues, and what it refuses to reissue."""

    @staticmethod
    def _transport(conns: list[_ScriptedConn]) -> tuple[TunnelTransport, list[_ScriptedConn]]:
        """A transport that hands out conns in order, and the list of the ones it actually opened."""
        transport = TunnelTransport('org1', 'db1', lambda: _KEY, host='h', port=443)
        opened: list[_ScriptedConn] = []
        queue = list(conns)

        def connect() -> object:
            conn = queue.pop(0)
            opened.append(conn)
            return conn

        transport._pool = proxy_client._TunnelPool(connect)  # type: ignore[arg-type]
        return transport, opened

    def test_a_daemon_that_dies_holding_the_request_is_not_retried(self) -> None:
        """The failure that motivated this: the daemon is OOM-killed mid-request, so reissuing it just
        kills the daemon again."""
        transport, opened = self._transport(
            [_ScriptedConn(on_read=ConnectionResetError('Connection reset by peer')) for _ in range(3)]
        )
        with pxt_raises(
            pxt.ErrorCode.INTERNAL_ERROR, match=r'became unresponsive while handling this request: pxt://org1:db1'
        ):
            transport.post(b'body')
        assert len(opened) == 1  # no reissue

    def test_a_request_that_never_landed_is_retried(self) -> None:
        """A write that fails leaves the daemon with nothing to act on, so the request can go again."""
        transport, opened = self._transport(
            [_ScriptedConn(on_write=ConnectionResetError('broken pipe')), _ScriptedConn(on_read=(200, b'second'))]
        )
        assert transport.post(b'body') == b'second'
        assert len(opened) == 2
        assert opened[0].closed

    def test_a_server_error_is_retried(self) -> None:
        """A 5xx comes from a daemon that is alive and answering; a rollout can produce one."""
        transport, opened = self._transport(
            [_ScriptedConn(on_read=(503, b'unavailable')), _ScriptedConn(on_read=(200, b'second'))]
        )
        assert transport.post(b'body') == b'second'
        assert len(opened) == 2

    def test_a_connection_the_os_refuses_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A firewall or sandbox that denies the socket denies it again, so the error comes at once."""
        attempts = 0

        def deny(*_args: Any, **_kwargs: Any) -> socket.socket:
            nonlocal attempts
            attempts += 1
            raise PermissionError(1, 'Operation not permitted')

        monkeypatch.setattr(socket, 'create_connection', deny)
        transport = TunnelTransport('org1', 'db1', lambda: _KEY, host='h', port=443)

        with pytest.raises(PermissionError):
            transport.post(b'body')
        assert attempts == 1

    def test_a_refused_credential_opens_no_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Renewing a session is a round trip of its own, so the credential is resolved before connecting."""

        def refuse() -> Credential:
            raise excs.AuthorizationError(excs.ErrorCode.MISSING_CREDENTIALS, 'no credential in this test')

        def connect(*_args: Any, **_kwargs: Any) -> socket.socket:
            raise AssertionError('connected before resolving the credential')

        monkeypatch.setattr(socket, 'create_connection', connect)
        transport = TunnelTransport('org1', 'db1', refuse, host='h', port=443)

        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='no credential in this test'):
            transport.post(b'body')

    def test_each_new_tunnel_sends_the_current_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A session renewed between two tunnels reaches the second: the credential is resolved per handshake."""
        sidecar = _PlainSidecar()
        monkeypatch.setattr(socket, 'create_connection', sidecar.connect)
        monkeypatch.setattr(ssl, 'create_default_context', _PlainTLS)
        credentials = iter([Credential('session', token, 'a test') for token in ('first-token', 'renewed-token')])
        transport = TunnelTransport('org1', 'db1', lambda: next(credentials), host='h', port=443)
        try:
            assert transport.post(b'one') == b'ok'
            assert transport.post(b'two') == b'ok'
        finally:
            transport.close()
            sidecar.close()

        assert sidecar.tokens == ['first-token', 'renewed-token']

    @staticmethod
    def _post_through(sidecar: _PlainSidecar, monkeypatch: pytest.MonkeyPatch) -> bytes:
        monkeypatch.setattr(socket, 'create_connection', sidecar.connect)
        monkeypatch.setattr(ssl, 'create_default_context', _PlainTLS)
        transport = TunnelTransport('org1', 'db1', lambda: _KEY, host='h', port=443)
        try:
            return transport.post(b'body')
        finally:
            transport.close()
            sidecar.close()

    def test_a_rejected_key_names_its_source_and_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar = _PlainSidecar((b'PXT/1.0 401 Unauthorized',))
        with pxt_raises(
            excs.ErrorCode.PROVIDER_AUTH_ERROR,
            match='The API key from the PIXELTABLE_API_KEY environment variable was rejected: Unauthorized',
        ):
            self._post_through(sidecar, monkeypatch)
        assert sidecar.tokens == ['key']

    def test_a_key_without_access_to_the_database_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar = _PlainSidecar((b'PXT/1.0 403 Forbidden',))
        with pxt_raises(
            excs.ErrorCode.INSUFFICIENT_PRIVILEGES, match='is valid but is not permitted to connect to pxt://org1:db1'
        ):
            self._post_through(sidecar, monkeypatch)
        assert sidecar.tokens == ['key']

    def test_a_handshake_the_sidecar_could_not_complete_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 5xx says nothing about the credential: the sidecar could not reach the daemon yet."""
        sidecar = _PlainSidecar((b'PXT/1.0 503 Service Unavailable',))
        assert self._post_through(sidecar, monkeypatch) == b'ok'
        assert sidecar.tokens == ['key', 'key']

    def test_a_client_error_is_not_retried(self) -> None:
        transport, opened = self._transport([_ScriptedConn(on_read=(404, b'nope')) for _ in range(2)])
        with pytest.raises(RuntimeError, match='error 404'):
            transport.post(b'body')
        assert len(opened) == 1

    def test_a_pooled_connection_the_server_closed_is_not_handed_out(self) -> None:
        """Without this, an idle connection the server closed would read as a daemon that died on the
        request, and the request would fail instead of going onto a fresh connection."""
        dead, live = _ScriptedConn(), _ScriptedConn()
        dead.close_peer()
        assert proxy_client._is_server_closed(dead)  # type: ignore[arg-type]
        assert not proxy_client._is_server_closed(live)  # type: ignore[arg-type]

        opened: list[_ScriptedConn] = []

        def connect() -> object:
            opened.append(live)
            return live

        pool = proxy_client._TunnelPool(connect)  # type: ignore[arg-type]
        pool._idle.append((dead, time.monotonic()))  # type: ignore[arg-type]
        with pool.borrow() as conn:
            assert conn is live
        assert dead.closed
        assert opened == [live]

    def test_a_connection_idle_nearly_as_long_as_the_daemon_keeps_it_is_not_handed_out(self) -> None:
        """The daemon closes a connection idle 5 s, and a request sent while that close is on its way is dropped."""
        stale, fresh = _ScriptedConn(), _ScriptedConn()
        opened: list[_ScriptedConn] = []

        def connect() -> object:
            opened.append(fresh)
            return fresh

        pool = proxy_client._TunnelPool(connect)  # type: ignore[arg-type]
        # still open, so only its age can tell the pool not to send on it
        pool._idle.append((stale, time.monotonic() - proxy_client._MAX_IDLE_S - 0.1))  # type: ignore[arg-type]
        with pool.borrow() as conn:
            assert conn is fresh
        assert stale.closed

        # one that went idle just now is handed out again
        with pool.borrow() as conn:
            assert conn is fresh
        assert opened == [fresh]

    def test_a_connection_idle_since_its_response_began_is_not_handed_out(self) -> None:
        """The daemon's idle timer starts once it has sent a response, which can be long before the client has
        read all of it: the age runs from when the response's headers arrived, not from when the read ended."""
        slow, fresh = _ScriptedConn(), _ScriptedConn()
        conns = iter([slow, fresh])
        pool = proxy_client._TunnelPool(lambda: next(conns))  # type: ignore[arg-type,return-value]
        with pool.borrow() as conn:
            assert conn is slow
            # its response began arriving long ago and was read only now
            slow.responded_at = time.monotonic() - proxy_client._MAX_IDLE_S - 0.1

        with pool.borrow() as conn:
            assert conn is fresh
        assert slow.closed


class _SigningPlane:
    """The control plane's signing operations: get_presigned_urls answers batch_status (400 is a control plane from
    before it), and get_presigned_url signs one key. Any other operation, such as a request for bucket credentials,
    fails the test."""

    def __init__(self) -> None:
        self.batch_status = 200
        self.batch_body: Callable[[dict[str, Any]], str] = json.dumps  # writes the body of a 200 to get_presigned_urls
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def signed(db: str, key: str) -> str:
        return f'https://r2.example.com/{db}/{key}?X-Amz-Signature=sig'

    def post(self, url: str, data: str, headers: dict[str, str], timeout: float) -> requests.Response:
        request = json.loads(data)
        self.requests.append(request)
        op, db = request['operation_type'], request['db']
        response = requests.Response()
        response.encoding = 'utf-8'
        response.status_code = 200
        if op == 'get_presigned_urls' and self.batch_status == 200:
            body = {'urls': {key: self.signed(db, key) for key in request['keys']}, 'expires_in': request['expires_in']}
            response._content = self.batch_body(body).encode()
            return response
        elif op == 'get_presigned_urls':
            response.status_code = self.batch_status
            # the control plane's error body: '<status phrase> : <reason>'
            response._content = f'{http.HTTPStatus(self.batch_status).phrase} : refused in this test'.encode()
            return response
        elif op == 'get_presigned_url':
            key = request['key']
            body = {'url': self.signed(db, key), 'key': key, 'expiration': request['expiration']}
        else:
            raise AssertionError(f'unexpected control-plane request: {op}')
        response._content = json.dumps(body).encode()
        return response

    def operations(self) -> list[str]:
        return [request['operation_type'] for request in self.requests]


class _SignedObjects:
    """A store that serves each signed URL its path as content, or answers status; records the URLs opened."""

    def __init__(self) -> None:
        self.status = 200
        self.expired = 0  # opens refused with 403 before status applies, as R2 refuses an expired URL
        self.opened: list[str] = []
        self._lock = threading.Lock()

    def urlopen(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        url = request.full_url
        with self._lock:
            self.opened.append(url)
            expired = self.expired > 0
            self.expired -= int(expired)
        if expired or self.status != 200:
            raise urllib.error.HTTPError(url, 403 if expired else self.status, 'Forbidden', None, None)
        parsed = urllib.parse.urlsplit(url)
        assert parsed.query == 'X-Amz-Signature=sig', url  # the signature reached the store
        return io.BytesIO(parsed.path.encode())


def _assert_no_signed_url(e: BaseException) -> None:
    """Neither e nor any exception a traceback prints with it holds a signed URL in its message, args or attributes."""
    exc: BaseException | None = e
    while exc is not None:
        for text in (str(exc), repr(exc.args), repr(vars(exc))):
            assert 'X-Amz-Signature' not in text, f'{type(exc).__name__} holds a signed URL: {text}'
        # a traceback prints the cause, or else the context unless it is suppressed
        exc = exc.__cause__ if exc.__cause__ is not None or exc.__suppress_context__ else exc.__context__


class TestHostedMediaReads:
    """A client reads home-bucket media through URLs the control plane signs, never with bucket credentials."""

    @pytest.fixture
    def signing(
        self, init_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[tuple[_SigningPlane, _SignedObjects]]:
        plane, objects = _SigningPlane(), _SignedObjects()
        monkeypatch.setattr(cloud_utils, 'resolve', lambda purpose: _KEY)
        monkeypatch.setattr(cloud_utils, 'SESSION', plane)
        monkeypatch.setattr(urllib.request, 'urlopen', objects.urlopen)
        yield plane, objects
        FileCache.get().clear(proxy_client._PROXY_MEDIA_TBL_ID)

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch) -> tuple[ProxyClient, list[str], list[str]]:
        """A client on a tunnel, with the paths it GETs through the tunnel and the URLs it passes to fetch_url()."""
        transport = TunnelTransport('org1', 'db1', lambda: _KEY, host='h', port=443)
        tunnel_gets: list[str] = []
        fetched: list[str] = []

        def tunnel_request(method: str, path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
            tunnel_gets.append(path)
            return b'daemon'

        def fetch_url(url: str) -> pathlib.Path:
            fetched.append(url)
            path = TempStore.create_path(extension='.png')
            path.write_bytes(b'remote')
            return path

        monkeypatch.setattr(transport, '_request', tunnel_request)
        monkeypatch.setattr(proxy_client, 'fetch_url', fetch_url)
        return ProxyClient(transport, CatalogPath(org='org1', db='db1')), tunnel_gets, fetched

    def test_signed_in_batches(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """250 home-bucket URLs, in both spellings, take three signing calls of at most 100 keys, and no request for
        bucket credentials."""
        plane, objects = signing
        client, tunnel_gets, fetched = self._client(monkeypatch)
        db = f'db_{uuid.uuid4().hex}'
        keys = [f'media/{i:03d}.jpg' for i in range(250)]
        urls = [
            f'pxtfs://org1:{db}/home/{key}' if i % 2 == 0 else f'pxt://org1:{db}/buckets/home/{key}'
            for i, key in enumerate(keys)
        ]

        local = client.fetch_media(urls)

        assert plane.operations() == ['get_presigned_urls'] * 3
        assert [request['keys'] for request in plane.requests] == [keys[:100], keys[100:200], keys[200:]]
        assert {(r['org'], r['db'], r['bucket_name'], r['expires_in']) for r in plane.requests} == {
            ('org1', db, 'home', 900)
        }
        assert sorted(objects.opened) == sorted(plane.signed(db, key) for key in keys)
        for url, key in zip(urls, keys):
            assert local[url].endswith('.jpg')
            assert pathlib.Path(local[url]).read_bytes() == f'/{db}/{key}'.encode()
        assert tunnel_gets == [] and fetched == []
        # no store was built for the bucket, so nothing holds its credentials
        assert not any(db in key for key in Env.get().object_store_clients(StorageTarget.PIXELTABLE_STORE).clients)

        # the cache keeps the URLs as stored: reading them again signs and downloads nothing
        assert client.fetch_media(urls) == local
        assert len(plane.requests) == 3
        assert len(objects.opened) == 250

    def test_mixed_media(self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch) -> None:
        """Daemon media still comes through the tunnel and other remote media through fetch_url(). Both spellings of
        an address share a key, and each database is signed on its own."""
        plane, _ = signing
        client, tunnel_gets, fetched = self._client(monkeypatch)
        db1, db2 = f'db_{uuid.uuid4().hex}', f'db_{uuid.uuid4().hex}'
        daemon_url = 'https://h:443/media/a/b.png'
        remote_url = 'https://example.com/c.png'
        pxtfs_url = f'pxtfs://org1:{db1}/home/k/1.png'
        pxt_url = f'pxt://org1:{db1}/buckets/home/k/1.png'
        other_db_url = f'pxtfs://org1:{db2}/home/k/2.png'

        local = client.fetch_media([daemon_url, remote_url, pxtfs_url, pxt_url, other_db_url])

        assert tunnel_gets == ['/media/a/b.png']
        assert fetched == [remote_url]
        assert sorted((request['db'], request['keys']) for request in plane.requests) == sorted(
            [(db1, ['k/1.png']), (db2, ['k/2.png'])]
        )
        assert pathlib.Path(local[daemon_url]).read_bytes() == b'daemon'
        assert pathlib.Path(local[remote_url]).read_bytes() == b'remote'
        assert pathlib.Path(local[pxtfs_url]).read_bytes() == f'/{db1}/k/1.png'.encode()
        assert pathlib.Path(local[pxt_url]).read_bytes() == f'/{db1}/k/1.png'.encode()
        assert pathlib.Path(local[other_db_url]).read_bytes() == f'/{db2}/k/2.png'.encode()

    def test_batches_count_keys_not_spellings(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """150 keys, each stored in both spellings, take two signing calls of 100 and 50 keys."""
        plane, objects = signing
        client, _, _ = self._client(monkeypatch)
        db = f'db_{uuid.uuid4().hex}'
        keys = [f'media/{i:03d}.jpg' for i in range(150)]
        urls = [url for key in keys for url in (f'pxtfs://org1:{db}/home/{key}', f'pxt://org1:{db}/buckets/home/{key}')]

        local = client.fetch_media(urls)

        assert [request['keys'] for request in plane.requests] == [keys[:100], keys[100:]]
        assert len(objects.opened) == 300
        for i, url in enumerate(urls):
            assert pathlib.Path(local[url]).read_bytes() == f'/{db}/{keys[i // 2]}'.encode()

    def test_signed_per_file_by_an_older_control_plane(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A control plane from before get_presigned_urls answers it 400: each key of that batch is then signed with
        get_presigned_url."""
        plane, objects = signing
        plane.batch_status = 400
        client, _, _ = self._client(monkeypatch)
        db = f'db_{uuid.uuid4().hex}'
        keys = [f'k/{i:03d}.jpg' for i in range(150)]

        local = client.fetch_media([f'pxtfs://org1:{db}/home/{key}' for key in keys])

        assert plane.operations() == (
            ['get_presigned_urls'] + ['get_presigned_url'] * 100 + ['get_presigned_urls'] + ['get_presigned_url'] * 50
        )
        per_file = [request for request in plane.requests if request['operation_type'] == 'get_presigned_url']
        assert [request['key'] for request in per_file] == keys
        assert all((request['method'], request['expiration']) == ('get', 900) for request in per_file)
        assert len(objects.opened) == 150
        assert len(local) == 150

    @pytest.mark.parametrize(
        ('status', 'code'),
        [
            (403, excs.ErrorCode.INSUFFICIENT_PRIVILEGES),
            (404, excs.ErrorCode.PROVIDER_BAD_REQUEST),
            (503, excs.ErrorCode.PROVIDER_ERROR),
        ],
    )
    def test_other_signing_errors_are_raised(
        self,
        signing: tuple[_SigningPlane, _SignedObjects],
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        code: excs.ErrorCode,
    ) -> None:
        """Any other refusal or failure to sign is raised: no key is signed per file and nothing is downloaded."""
        plane, objects = signing
        plane.batch_status = status
        client, _, _ = self._client(monkeypatch)

        with pxt_raises(code):
            client.fetch_media([f'pxtfs://org1:db_{uuid.uuid4().hex}/home/k/1.jpg'])
        assert plane.operations() == ['get_presigned_urls']
        assert objects.opened == []

    @pytest.mark.parametrize(
        'batch_body',
        [
            pytest.param(lambda body: json.dumps({'urls': body['urls']}), id='no-expires_in'),
            pytest.param(lambda body: json.dumps(body)[:-1], id='not-json'),
        ],
    )
    def test_a_malformed_signing_answer_quotes_no_url(
        self,
        signing: tuple[_SigningPlane, _SignedObjects],
        monkeypatch: pytest.MonkeyPatch,
        batch_body: Callable[[dict[str, Any]], str],
    ) -> None:
        """A 200 that is not JSON, or not the answer's shape, is raised without its body, whose URLs are credentials."""
        plane, objects = signing
        plane.batch_body = batch_body
        client, _, _ = self._client(monkeypatch)

        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR) as info:
            client.fetch_media([f'pxtfs://org1:db_{uuid.uuid4().hex}/home/k/1.jpg'])
        assert str(info.value) == 'Pixeltable Cloud returned a malformed answer to get_presigned_urls'
        _assert_no_signed_url(info.value)
        assert plane.operations() == ['get_presigned_urls']
        assert objects.opened == []

    def test_a_refused_download_names_the_stored_url(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error names the URL as stored, not the signed URL, whose query string is a credential."""
        _, objects = signing
        objects.status = 403
        client, _, _ = self._client(monkeypatch)
        url = f'pxt://org1:db_{uuid.uuid4().hex}/buckets/home/k/1.jpg'

        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR, match='Failed to download pxt://') as info:
            client.fetch_media([url])
        assert str(info.value) == f'Failed to download {url}: HTTP 403'
        _assert_no_signed_url(info.value)
        assert len(objects.opened) == 2  # signed again once, as an expired URL would be, then raised

    def test_a_failed_download_names_the_stored_url(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any other download error is named by its type alone: its message can quote the signed URL, as a ValueError
        for a malformed URL does."""
        plane, _ = signing
        client, _, _ = self._client(monkeypatch)
        url = f'pxt://org1:db_{uuid.uuid4().hex}/buckets/home/k/1.jpg'

        def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
            raise ValueError(request.full_url)

        monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
        with pxt_raises(excs.ErrorCode.PROVIDER_ERROR) as info:
            client.fetch_media([url])
        assert str(info.value) == f'Failed to download {url}: ValueError'
        _assert_no_signed_url(info.value)
        assert plane.operations() == ['get_presigned_urls']  # only a 403 is signed again

    def test_an_expired_url_is_signed_again_once(
        self, signing: tuple[_SigningPlane, _SignedObjects], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A URL signed before a long result's first download can expire before its own starts."""
        plane, objects = signing
        objects.expired = 1
        client, _, _ = self._client(monkeypatch)
        db = f'db_{uuid.uuid4().hex}'
        url = f'pxtfs://org1:{db}/home/k/1.jpg'

        local = client.fetch_media([url])

        assert pathlib.Path(local[url]).read_bytes() == f'/{db}/k/1.jpg'.encode()
        assert plane.operations() == ['get_presigned_urls', 'get_presigned_urls']
        assert [request['keys'] for request in plane.requests] == [['k/1.jpg'], ['k/1.jpg']]
        assert len(objects.opened) == 2
