import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.functions.image import resize, rotate
from pixeltable.functions.string import format
from pixeltable.functions.timestamp import make_timestamp
from pixeltable.functions.video import clip, get_duration, segment_video
from pixeltable.functions.vision import bboxes_resize

from .utils import free_port, get_image_files, get_video_files, pxt_raises, reload_catalog, skip_test_if_not_installed

# the paths of the UDFs with a server in the udf_servers fixture
_SERVED_UDFS = (
    'pixeltable.functions.image.rotate',
    'pixeltable.functions.string.format',
    'pixeltable.functions.timestamp.make_timestamp',
    'pixeltable.functions.vision.bboxes_resize',
    'pixeltable.functions.video.editing.clip',
    'pixeltable.functions.video.editing.segment_video',
    'pixeltable.functions.huggingface.sentence_transformer',
)

# serves the UDF at argv[1] on port argv[2] until the process is terminated
_SERVER_SCRIPT = """
import sys, time
from pixeltable.serving.udf_server import UdfServer
UdfServer().start(sys.argv[1], int(sys.argv[2]), None)
time.sleep(1e9)
"""


@pytest.fixture(scope='module')
def udf_servers(init_env: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Serves each of _SERVED_UDFS from a process of its own and points this process's Modal calls at them."""
    urls: dict[str, str] = {}
    servers: list[subprocess.Popen] = []
    try:
        for udf_path in _SERVED_UDFS:
            port = free_port()
            # each server gets its own home, because UdfServer.run() clears the TempStore after each call, and a time
            # zone that differs from this process's
            env = {
                **os.environ,
                'PIXELTABLE_HOME': str(tmp_path_factory.mktemp('udf-server') / '.pixeltable'),
                'PIXELTABLE_TIME_ZONE': 'Pacific/Kiritimati',
            }
            servers.append(subprocess.Popen([sys.executable, '-c', _SERVER_SCRIPT, udf_path, str(port)], env=env))
            urls[udf_path] = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 120
        for server, url in zip(servers, urls.values()):
            while True:
                assert server.poll() is None, 'a UDF server exited'
                assert time.monotonic() < deadline, 'a UDF server did not start'
                try:
                    if httpx.get(url).status_code == 404:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('PXT_MODAL_SERVERS', json.dumps(urls))
            mp.setenv('MODAL_PROXY_KEY', 'wk-test')
            mp.setenv('MODAL_PROXY_SECRET', 'ws-test')
            yield
    finally:
        for server in servers:
            server.terminate()
        for server in servers:
            server.wait()


@pytest.fixture
def udf_setting(udf_servers: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Puts _SERVED_UDFS into the pixeltable.udf setting."""
    monkeypatch.setenv('PIXELTABLE_UDF', json.dumps({udf_path: {} for udf_path in _SERVED_UDFS}))


@pytest.mark.db_roots('local', reason='the pixeltable.udf setting of this process routes the calls')
class TestUdfServer:
    def test_scalar_values(self, uses_db: None, udf_setting: None) -> None:
        """Calls in a computed column, a select() and a query function return the UDF's results."""
        t = pxt.create_table('udf_server_tbl', {'s': pxt.String | None, 'boxes': pxt.Json, 'rel_boxes': pxt.Json})
        # variadic arguments; a null goes to the UDF
        t.add_computed_column(f=format('{0} {name} {1}', t.s, 1, name=t.s))
        # a polymorphic UDF: an Int width selects the overload for absolute boxes, a Float width the one for relative
        # boxes
        t.add_computed_column(by_int=bboxes_resize(t.boxes, 'xyxy', width=4))
        t.add_computed_column(by_float=bboxes_resize(t.rel_boxes, 'xyxy', width=0.5))
        rows = [
            {'s': 'a', 'boxes': [[0, 0, 2, 2]], 'rel_boxes': [[0.25, 0.25, 0.5, 0.5]]},
            {'s': 'b', 'boxes': [[1, 1, 3, 5]], 'rel_boxes': [[0.5, 0.25, 0.75, 0.75]]},
            {'s': None, 'boxes': [], 'rel_boxes': []},
        ]
        status = t.insert(rows)
        assert status.num_excs == 0

        @pxt.query
        def formatted(prefix: str) -> pxt.Query:
            return t.select(s=format('{0}-{1}', prefix, t.s)).order_by(t.s)

        def check() -> None:
            res = t.order_by(t.s).collect()
            assert res['f'] == ['a a 1', 'b b 1', 'None None 1']
            assert res['by_int'] == [[[-1, -1, 3, 3]], [[0, -1, 4, 7]], []]
            assert res['by_float'] == [[[0.125, 0.125, 0.625, 0.625]], [[0.375, 0.0, 0.875, 1.0]], []]
            assert t.select(out=format('<{0}>', t.s)).order_by(t.s).collect()['out'] == ['<a>', '<b>', '<None>']

        check()
        prefixes = pxt.create_table('prefix_tbl', {'prefix': pxt.String})
        prefixes.insert([{'prefix': 'x'}])
        res = prefixes.select(out=formatted(prefixes.prefix)).collect()
        assert [row['s'] for row in res['out'][0]] == ['x-a', 'x-b', 'x-None']

        reload_catalog()
        t = pxt.get_table('udf_server_tbl')
        check()

        # the UDF runs with this process's default time zone, not the server's
        expected = datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.datetime.now().astimezone().tzinfo)
        ts = t.select(ts=make_timestamp(2026, 1, 2, 3, 4)).limit(1).collect()['ts'][0]
        assert ts == expected and ts.utcoffset() == expected.utcoffset()

    def test_media(self, uses_db: None, udf_setting: None) -> None:
        """Images and videos travel to the UDF and back, also nested in a result."""
        img_t = pxt.create_table('img_tbl', {'img': pxt.Image | None})
        img_t.add_computed_column(rotated=rotate(img_t.img, 90))
        # an in-memory image that another UDF produced
        img_t.add_computed_column(small=rotate(resize(img_t.img, (32, 16)), 90))
        img_paths = get_image_files()[:3]
        status = img_t.insert([{'img': path} for path in img_paths] + [{'img': None}])
        assert status.num_excs == 0
        res = img_t.where(img_t.img != None).collect()
        assert len(res) == len(img_paths)
        for img, rotated, small in zip(res['img'], res['rotated'], res['small']):
            assert rotated.size == img.size
            assert small.size == (32, 16)
        # a null image doesn't reach the UDF
        assert img_t.where(img_t.img == None).collect()['rotated'] == [None]

        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        video_t.add_computed_column(excerpt=clip(video_t.video, start_time=0.0, duration=1.0))
        video_t.add_computed_column(segments=segment_video(video_t.video, duration=1.0))
        status = video_t.insert([{'video': get_video_files(include_bad_video=False, extension='.mp4')[0]}])
        assert status.num_excs == 0
        res = video_t.select(
            video_t.segments, excerpt=video_t.excerpt.localpath, duration=get_duration(video_t.excerpt)
        ).collect()
        assert Path(res['excerpt'][0]).stat().st_size > 0
        assert 0.5 < res['duration'][0] < 1.5
        assert len(res['segments'][0]) > 1

    def test_errors(self, uses_db: None, udf_setting: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exception of the UDF fails its row with the UDF's error, and a UDF without a Server fails the query."""
        t = pxt.create_table('err_tbl', {'boxes': pxt.Json, 'width': pxt.Int})
        t.add_computed_column(resized=bboxes_resize(t.boxes, 'xyxy', width=t.width))
        status = t.insert(
            [{'boxes': [[0, 0, 2, 2]], 'width': 4}, {'boxes': [[0, 0, 2, 2]], 'width': -1}], on_error='ignore'
        )
        assert status.num_rows == 2 and status.num_excs == 1
        res = t.select(t.width, t.resized, errortype=t.resized.errortype, errormsg=t.resized.errormsg).collect()
        failed = next(row for row in res if row['width'] == -1)
        assert failed['resized'] is None
        assert failed['errortype'] == 'RequestError'
        assert 'width must be positive, got -1' in failed['errormsg']
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match='width must be positive, got -1'):
            t.insert([{'boxes': [[0, 0, 2, 2]], 'width': -1}])
        with pxt_raises(excs.ErrorCode.INVALID_ARGUMENT, match='width must be positive, got -1'):
            t.select(bboxes_resize(t.boxes, 'xyxy', width=-1)).collect()

        img_t = pxt.create_table('img_tbl', {'img': pxt.Image})
        img_t.insert([{'img': get_image_files()[0]}])
        monkeypatch.setenv('PIXELTABLE_UDF', json.dumps({'pixeltable.functions.image.resize': {}}))
        with pxt_raises(
            excs.ErrorCode.INVALID_CONFIGURATION,
            match=r"'pixeltable\.functions\.image\.resize' is in the pixeltable\.udf setting, but PXT_MODAL_SERVERS "
            r'has no Modal Server for it',
        ):
            img_t.select(resize(img_t.img, (32, 16))).collect()

    @pytest.mark.xdist_group('large_model')
    def test_batched_udf(self, uses_db: None, udf_setting: None) -> None:
        """The batches of a batched UDF run on the server, one call per batch."""
        skip_test_if_not_installed('sentence_transformers')
        from pixeltable.functions.huggingface import sentence_transformer

        t = pxt.create_table('batch_tbl', {'s': pxt.String})
        t.add_computed_column(embed=sentence_transformer(t.s, model_id='sentence-transformers/all-MiniLM-L6-v2'))
        status = t.insert([{'s': f'sentence {i}'} for i in range(40)])
        assert status.num_excs == 0
        embeds = t.select(t.embed).collect()['embed']
        assert len(embeds) == 40 and all(embed.shape == (384,) for embed in embeds)
