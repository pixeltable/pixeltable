import json
import os
import time
from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.functions.json as pxt_json
from pixeltable.env import Env
from tests.utils import get_audio_files, get_image_files, get_video_files, skip_test_if_not_installed, sleep


@pxt.udf
def add_one(x: int) -> int:
    return x + 1


class TestFastAPI:
    def test_add_insert_route_scalars(self, uses_db: None) -> None:
        """Test insert routes with all scalar types and various input/output combinations."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table(
            'test_serve.scalars',
            {
                'id': pxt.Int,
                'str_col': pxt.String,
                'int_col': pxt.Int,
                'float_col': pxt.Float,
                'bool_col': pxt.Bool,
                'json_col': pxt.Json,
            },
        )
        t.add_computed_column(str_upper=t.str_col.upper())
        t.add_computed_column(int_plus1=t.int_col + 1)
        t.add_computed_column(float_abs=t.float_col.abs())
        t.add_computed_column(json_str=pxt_json.dumps(t.json_col))

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        # default inputs and outputs
        router.add_insert_route(t, path='/all')
        # subset of inputs, all outputs
        router.add_insert_route(t, path='/partial-in', inputs=['id', 'str_col', 'int_col'])
        # all inputs, subset of outputs
        router.add_insert_route(t, path='/partial-out', outputs=['id', 'str_upper', 'int_plus1'])
        # minimal inputs and outputs
        router.add_insert_route(t, path='/minimal', inputs=['id', 'int_col'], outputs=['int_plus1'])
        app.include_router(router)
        client = TestClient(app)

        all_input = {
            'id': 1,
            'str_col': 'hello',
            'int_col': -5,
            'float_col': -3.14,
            'bool_col': True,
            'json_col': {'key': 'value'},
        }
        resp = client.post('/all', json=all_input)
        assert resp.status_code == 200, resp.text
        expected = {
            'id': 1,
            'str_col': 'hello',
            'int_col': -5,
            'float_col': -3.14,
            'bool_col': True,
            'json_col': {'key': 'value'},
            'str_upper': 'HELLO',
            'int_plus1': -4,
            'float_abs': 3.14,
            'json_str': json.dumps({'key': 'value'}),
        }
        assert resp.json() == expected
        row = t.where(t.id == 1).collect()[0]
        assert row == expected

        resp = client.post('/partial-in', json={'id': 2, 'str_col': 'world', 'int_col': 10})
        assert resp.status_code == 200, resp.text
        expected = {
            'id': 2,
            'str_col': 'world',
            'int_col': 10,
            'float_col': None,
            'bool_col': None,
            'json_col': None,
            'str_upper': 'WORLD',
            'int_plus1': 11,
            'float_abs': None,
            'json_str': None,
        }
        assert resp.json() == expected
        print(resp.json())
        row = t.where(t.id == 2).collect()[0]
        assert row == expected

        resp = client.post('/partial-out', json={**all_input, 'id': 3})
        assert resp.status_code == 200, resp.text
        expected = {'id': 3, 'str_upper': 'HELLO', 'int_plus1': -4}
        assert resp.json() == expected
        row = t.where(t.id == 3).select(t.id, t.str_upper, t.int_plus1).collect()[0]
        assert row == expected

        resp = client.post('/minimal', json={'id': 4, 'int_col': 99})
        assert resp.status_code == 200, resp.text
        expected = {'int_plus1': 100}
        assert resp.json() == expected
        row = t.where(t.id == 4).select(t.int_plus1).collect()[0]
        assert row == expected

    @pytest.mark.parametrize('use_uploadfile', [True, False])
    def test_add_insert_route_video(self, uses_db: None, use_uploadfile: bool) -> None:
        """Test insert routes with video data, including FileResponse."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        video_path = get_video_files()[0]
        pxt.create_dir('test_serve')
        t = pxt.create_table(
            'test_serve.videos', {'id': pxt.Int, 'video': pxt.Video, 'width': pxt.Int, 'height': pxt.Int}
        )
        t.add_computed_column(resized=t.video.resize(width=t.width, height=t.height))
        t.add_computed_column(thumbnail=t.video.extract_frame(timestamp=0.0))

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        # When uploading, 'video' moves from inputs into uploadfile_inputs. Other inputs
        # (defaulted for /all, explicit for /resize and /thumbnail) become Form fields
        # automatically once any upload is present.
        uploadfile_inputs = ['video'] if use_uploadfile else None
        # /all: defaults for inputs/outputs (uploadfile_inputs still moves `video` into File)
        router.add_insert_route(t, path='/all', uploadfile_inputs=uploadfile_inputs)
        # /resize: id + video + width, only resized video output
        router.add_insert_route(
            t,
            path='/resize',
            inputs=['id', 'width'] if use_uploadfile else ['id', 'video', 'width'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['resized'],
            return_fileresponse=True,
        )
        # /thumbnail: id + video + height, only thumbnail output
        router.add_insert_route(
            t,
            path='/thumbnail',
            inputs=['id', 'height'] if use_uploadfile else ['id', 'video', 'height'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['thumbnail'],
            return_fileresponse=True,
        )
        app.include_router(router)
        client = TestClient(app)
        media_dir = str(Env.get().media_dir)

        def post(path: str, row_id: int, **fields: Any) -> Any:
            """Send a POST to `path` with the fixed video plus scalar fields, honoring use_uploadfile."""
            if use_uploadfile:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                return client.post(
                    path,
                    files={'video': (os.path.basename(video_path), video_bytes, 'video/webm')},
                    data={'id': str(row_id), **{k: str(v) for k, v in fields.items()}},
                )
            else:
                return client.post(path, json={'id': row_id, 'video': video_path, **fields})

        # route /all: JSON response with media URLs
        resp = post('/all', 1, width=320, height=240)
        assert resp.status_code == 200, resp.text
        result = resp.json()
        print(result)
        assert result['id'] == 1 and result['width'] == 320 and result['height'] == 240

        video_local = t.where(t.id == 1).select(p=t.video.localpath).collect()[0]['p']
        if use_uploadfile:
            # this needs to be a file in our media dir
            assert '/media/' in result['video'], result['video']
            assert video_local.startswith(media_dir + os.sep), f'video not under media_dir: {video_local}'
            # we got the fetchable url in the response
            media_resp = client.get(result['video'])
            assert media_resp.status_code == 200, media_resp.text
            with open(video_local, 'rb') as f:
                assert media_resp.content == f.read()
        else:
            # this stores the local file reference we passed in
            assert result['video'].startswith('file:')
            assert not video_local.startswith(media_dir + os.sep), f'external video moved into media_dir: {video_local}'

        paths = t.where(t.id == 1).select(resized=t.resized.localpath, thumbnail=t.thumbnail.localpath).collect()[0]
        # validate that 'resized' and 'thumbnail' outputs are fetchable media
        for col in ('resized', 'thumbnail'):
            assert '/media/' in result[col]
            media_resp = client.get(result[col])
            assert media_resp.status_code == 200, f'{col}: {media_resp.text}'
            with open(paths[col], 'rb') as f:
                assert media_resp.content == f.read(), f'{col}: downloaded bytes differ from stored file'
        # verify persisted row
        row = t.where(t.id == 1).select(t.id, t.width, t.height).collect()[0]
        assert row == {'id': 1, 'width': 320, 'height': 240}

        # route /resize: FileResponse with resized video; response bytes must match the stored file
        resp = post('/resize', 2, width=160)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('video/')
        resized_path = t.where(t.id == 2).select(p=t.resized.localpath).collect()[0]['p']
        with open(resized_path, 'rb') as f:
            assert resp.content == f.read()

        # route /thumbnail: FileResponse with thumbnail image; response bytes must match the stored file
        resp = post('/thumbnail', 3, height=120)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('image/')
        thumbnail_path = t.where(t.id == 3).select(p=t.thumbnail.localpath).collect()[0]['p']
        with open(thumbnail_path, 'rb') as f:
            assert resp.content == f.read()

    @pytest.mark.parametrize('use_uploadfile', [True, False])
    def test_add_insert_route_image(self, uses_db: None, use_uploadfile: bool) -> None:
        """Image counterpart of test_add_insert_route_video. Structurally parallel so the two
        tests can later be generalized over a media-kind fixture."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        image_path = get_image_files()[0]
        pxt.create_dir('test_serve')
        # Unlike video.resize (which tolerates None width/height and preserves aspect ratio),
        # image.resize requires concrete ints - so width and height must be Required, and every
        # insert (even the /rotate one, which doesn't care about them) has to supply them.
        t = pxt.create_table(
            'test_serve.images',
            {'id': pxt.Int, 'image': pxt.Image, 'width': pxt.Required[pxt.Int], 'height': pxt.Required[pxt.Int]},
        )
        # resized: uses both scalar inputs (mirrors video.resize(width=..., height=...))
        t.add_computed_column(resized=t.image.resize(size=(t.width, t.height)))
        # rotated: uses neither scalar input (mirrors video.extract_frame(timestamp=0.0))
        t.add_computed_column(rotated=t.image.rotate(angle=90))

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        # When uploading, 'image' moves from inputs into uploadfile_inputs. Other inputs
        # (defaulted for /all, explicit for /resize and /rotate) become Form fields
        # automatically once any upload is present.
        uploadfile_inputs = ['image'] if use_uploadfile else None
        # /all: defaults for inputs/outputs (uploadfile_inputs still moves `image` into File)
        router.add_insert_route(t, path='/all', uploadfile_inputs=uploadfile_inputs)
        # /resize: id + image + width + height, only resized image output
        router.add_insert_route(
            t,
            path='/resize',
            inputs=['id', 'width', 'height'] if use_uploadfile else ['id', 'image', 'width', 'height'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['resized'],
            return_fileresponse=True,
        )
        # /rotate: id + image + width + height, only rotated image output.
        # width/height are unused by the rotate computation but still required by the schema,
        # since every insert evaluates the `resized` computed column.
        router.add_insert_route(
            t,
            path='/rotate',
            inputs=['id', 'width', 'height'] if use_uploadfile else ['id', 'image', 'width', 'height'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['rotated'],
            return_fileresponse=True,
        )
        app.include_router(router)
        client = TestClient(app)
        media_dir = str(Env.get().media_dir)

        def post(path: str, row_id: int, **fields: Any) -> Any:
            """Send a POST to `path` with the fixed image plus scalar fields, honoring use_uploadfile."""
            if use_uploadfile:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                return client.post(
                    path,
                    files={'image': (os.path.basename(image_path), image_bytes, 'image/jpeg')},
                    data={'id': str(row_id), **{k: str(v) for k, v in fields.items()}},
                )
            else:
                return client.post(path, json={'id': row_id, 'image': image_path, **fields})

        # route /all: JSON response with media URLs
        resp = post('/all', 1, width=128, height=96)
        assert resp.status_code == 200, resp.text
        result = resp.json()
        print(result)
        assert result['id'] == 1 and result['width'] == 128 and result['height'] == 96

        image_local = t.where(t.id == 1).select(p=t.image.localpath).collect()[0]['p']
        if use_uploadfile:
            # this needs to be a file in our media dir
            assert '/media/' in result['image'], result['image']
            assert image_local.startswith(media_dir + os.sep), f'image not under media_dir: {image_local}'
            # we got the fetchable url in the response
            media_resp = client.get(result['image'])
            assert media_resp.status_code == 200, media_resp.text
            with open(image_local, 'rb') as f:
                assert media_resp.content == f.read()
        else:
            # this stores the local file reference we passed in
            assert result['image'].startswith('file:')
            assert not image_local.startswith(media_dir + os.sep), f'external image moved into media_dir: {image_local}'

        paths = t.where(t.id == 1).select(resized=t.resized.localpath, rotated=t.rotated.localpath).collect()[0]
        # validate that 'resized' and 'rotated' outputs are fetchable media
        for col in ('resized', 'rotated'):
            assert '/media/' in result[col]
            media_resp = client.get(result[col])
            assert media_resp.status_code == 200, f'{col}: {media_resp.text}'
            with open(paths[col], 'rb') as f:
                assert media_resp.content == f.read(), f'{col}: downloaded bytes differ from stored file'
        # verify persisted row
        row = t.where(t.id == 1).select(t.id, t.width, t.height).collect()[0]
        assert row == {'id': 1, 'width': 128, 'height': 96}

        # route /resize: FileResponse with resized image; response bytes must match the stored file
        resp = post('/resize', 2, width=64, height=48)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('image/')
        resized_path = t.where(t.id == 2).select(p=t.resized.localpath).collect()[0]['p']
        with open(resized_path, 'rb') as f:
            assert resp.content == f.read()

        # route /rotate: FileResponse with rotated image; response bytes must match the stored file.
        # width/height must be supplied (schema-required) even though the rotation doesn't use them.
        resp = post('/rotate', 3, width=32, height=24)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('image/')
        rotated_path = t.where(t.id == 3).select(p=t.rotated.localpath).collect()[0]['p']
        with open(rotated_path, 'rb') as f:
            assert resp.content == f.read()

    @pytest.mark.parametrize('use_uploadfile', [True, False])
    def test_add_insert_route_audio(self, uses_db: None, use_uploadfile: bool) -> None:
        """Audio counterpart of test_add_insert_route_video/_image. Structurally parallel so the
        three tests can later be generalized over a media-kind fixture. Uses the audio UDFs
        `multiply_volume` (two scalar inputs) and `normalize` (no scalar inputs)."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        # Use sample-16-bit.wav specifically so the mime type in upload mode is deterministic.
        audio_path = next(f for f in get_audio_files() if f.endswith('sample-16-bit.wav'))
        pxt.create_dir('test_serve')
        # factor and end_time are Required because multiply_volume's `factor` param must be non-None
        # on every insert (the `scaled` computed column runs on every row).
        t = pxt.create_table(
            'test_serve.audios',
            {'id': pxt.Int, 'audio': pxt.Audio, 'factor': pxt.Required[pxt.Float], 'end_time': pxt.Required[pxt.Float]},
        )
        # scaled: uses both scalar inputs (mirrors video.resize(width=..., height=...))
        t.add_computed_column(scaled=t.audio.multiply_volume(factor=t.factor, end_time=t.end_time))
        # normalized: uses neither scalar input (mirrors video.extract_frame(timestamp=0.0))
        t.add_computed_column(normalized=t.audio.normalize())

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        # When uploading, 'audio' moves from inputs into uploadfile_inputs. Other inputs
        # (defaulted for /all, explicit for /scale and /normalize) become Form fields
        # automatically once any upload is present.
        uploadfile_inputs = ['audio'] if use_uploadfile else None
        # /all: defaults for inputs/outputs (uploadfile_inputs still moves `audio` into File)
        router.add_insert_route(t, path='/all', uploadfile_inputs=uploadfile_inputs)
        # /scale: id + audio + factor + end_time, only the multiply_volume output
        router.add_insert_route(
            t,
            path='/scale',
            inputs=['id', 'factor', 'end_time'] if use_uploadfile else ['id', 'audio', 'factor', 'end_time'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['scaled'],
            return_fileresponse=True,
        )
        # /normalize: id + audio + factor + end_time, only the normalize output.
        # factor/end_time are unused by the normalize computation but still required by the
        # schema, since every insert evaluates the `scaled` computed column.
        router.add_insert_route(
            t,
            path='/normalize',
            inputs=['id', 'factor', 'end_time'] if use_uploadfile else ['id', 'audio', 'factor', 'end_time'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['normalized'],
            return_fileresponse=True,
        )
        app.include_router(router)
        client = TestClient(app)
        media_dir = str(Env.get().media_dir)

        def post(path: str, row_id: int, **fields: Any) -> Any:
            """Send a POST to `path` with the fixed audio plus scalar fields, honoring use_uploadfile."""
            if use_uploadfile:
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                return client.post(
                    path,
                    files={'audio': (os.path.basename(audio_path), audio_bytes, 'audio/wav')},
                    data={'id': str(row_id), **{k: str(v) for k, v in fields.items()}},
                )
            else:
                return client.post(path, json={'id': row_id, 'audio': audio_path, **fields})

        # route /all: JSON response with media URLs
        resp = post('/all', 1, factor=0.5, end_time=0.5)
        assert resp.status_code == 200, resp.text
        result = resp.json()
        print(result)
        assert result['id'] == 1 and result['factor'] == 0.5 and result['end_time'] == 0.5

        audio_local = t.where(t.id == 1).select(p=t.audio.localpath).collect()[0]['p']
        if use_uploadfile:
            # this needs to be a file in our media dir
            assert '/media/' in result['audio'], result['audio']
            assert audio_local.startswith(media_dir + os.sep), f'audio not under media_dir: {audio_local}'
            # we got the fetchable url in the response
            media_resp = client.get(result['audio'])
            assert media_resp.status_code == 200, media_resp.text
            with open(audio_local, 'rb') as f:
                assert media_resp.content == f.read()
        else:
            # this stores the local file reference we passed in
            assert result['audio'].startswith('file:')
            assert not audio_local.startswith(media_dir + os.sep), f'external audio moved into media_dir: {audio_local}'

        paths = t.where(t.id == 1).select(scaled=t.scaled.localpath, normalized=t.normalized.localpath).collect()[0]
        # validate that 'scaled' and 'normalized' outputs are fetchable media
        for col in ('scaled', 'normalized'):
            assert '/media/' in result[col]
            media_resp = client.get(result[col])
            assert media_resp.status_code == 200, f'{col}: {media_resp.text}'
            with open(paths[col], 'rb') as f:
                assert media_resp.content == f.read(), f'{col}: downloaded bytes differ from stored file'
        # verify persisted row
        row = t.where(t.id == 1).select(t.id, t.factor, t.end_time).collect()[0]
        assert row == {'id': 1, 'factor': 0.5, 'end_time': 0.5}

        # route /scale: FileResponse with volume-scaled audio; response bytes must match the stored file
        resp = post('/scale', 2, factor=0.25, end_time=1.0)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('audio/')
        scaled_path = t.where(t.id == 2).select(p=t.scaled.localpath).collect()[0]['p']
        with open(scaled_path, 'rb') as f:
            assert resp.content == f.read()

        # route /normalize: FileResponse with normalized audio; response bytes must match the stored file.
        # factor/end_time must be supplied (schema-required) even though normalize doesn't use them.
        resp = post('/normalize', 3, factor=0.5, end_time=0.5)
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('audio/')
        normalized_path = t.where(t.id == 3).select(p=t.normalized.localpath).collect()[0]['p']
        with open(normalized_path, 'rb') as f:
            assert resp.content == f.read()

    @pytest.mark.parametrize('use_uploadfile', [True, False])
    def test_add_insert_route_video_bg(self, uses_db: None, use_uploadfile: bool) -> None:
        """Background variant of test_add_insert_route_video: POST returns a job id/url, the
        work runs in FastAPIRouter._executor, and the result is fetched via /jobs/{id}."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        video_path = get_video_files()[0]
        pxt.create_dir('test_serve')
        t = pxt.create_table(
            'test_serve.videos', {'id': pxt.Int, 'video': pxt.Video, 'width': pxt.Int, 'height': pxt.Int}
        )
        t.add_computed_column(resized=t.video.resize(width=t.width, height=t.height))
        t.add_computed_column(thumbnail=t.video.extract_frame(timestamp=0.0))
        # Delay every insert so the polling loop actually observes a 'pending' response
        # before 'done' - otherwise fast inserts would transition straight to 'done'
        # between the POST and the first GET and we wouldn't exercise the polling path.
        t.add_computed_column(delay=sleep(1.0))

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        uploadfile_inputs = ['video'] if use_uploadfile else None
        # /all: defaults for inputs/outputs, all columns in the response
        router.add_insert_route(t, path='/all', uploadfile_inputs=uploadfile_inputs, background=True)
        # /resize: single-output JSON response (no FileResponse - mutually exclusive with background)
        router.add_insert_route(
            t,
            path='/resize',
            inputs=['id', 'width'] if use_uploadfile else ['id', 'video', 'width'],
            uploadfile_inputs=uploadfile_inputs,
            outputs=['resized'],
            background=True,
        )
        app.include_router(router)
        client = TestClient(app)
        media_dir = str(Env.get().media_dir)

        def post(path: str, row_id: int, **fields: Any) -> Any:
            if use_uploadfile:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                return client.post(
                    path,
                    files={'video': (os.path.basename(video_path), video_bytes, 'video/webm')},
                    data={'id': str(row_id), **{k: str(v) for k, v in fields.items()}},
                )
            return client.post(path, json={'id': row_id, 'video': video_path, **fields})

        def await_job(job: dict[str, Any]) -> dict[str, Any]:
            """Validate the BackgroundJobResponse structure, poll until terminal, return the status body."""
            assert isinstance(job.get('id'), str) and len(job['id']) > 0
            assert isinstance(job.get('job_url'), str) and '/jobs/' in job['job_url'] and job['id'] in job['job_url']
            saw_pending = False
            deadline = time.time() + 30.0
            while True:
                status_resp = client.get(job['job_url'])
                assert status_resp.status_code == 200, status_resp.text
                body = status_resp.json()
                assert body['status'] in ('pending', 'done', 'error')
                if body['status'] == 'pending':
                    saw_pending = True
                    assert body.get('result') is None and body.get('error') is None
                    if time.time() > deadline:
                        raise AssertionError(f'job {job["id"]} still pending after 30s')
                    time.sleep(0.05)
                    continue
                # terminal state
                if body['status'] == 'error':
                    raise AssertionError(f'background job failed: {body.get("error")}')
                assert body['status'] == 'done'
                assert body.get('error') is None
                assert body.get('result') is not None
                # The sleep(0.3) computed column should guarantee we see 'pending' at least once.
                assert saw_pending, (
                    'polling never observed a pending response - sleep column is not delaying the insert'
                )
                return body

        # unknown job id -> 404 (exercised once; independent of the actual jobs)
        assert client.get('/jobs/__nonexistent__').status_code == 404

        # /all route (row id 1): multi-column response
        resp = post('/all', 1, width=320, height=240)
        assert resp.status_code == 200, resp.text
        job = resp.json()
        result = await_job(job)['result']
        assert result['id'] == 1 and result['width'] == 320 and result['height'] == 240

        video_local = t.where(t.id == 1).select(p=t.video.localpath).collect()[0]['p']
        if use_uploadfile:
            assert '/media/' in result['video'], result['video']
            assert video_local.startswith(media_dir + os.sep), f'video not under media_dir: {video_local}'
            media_resp = client.get(result['video'])
            assert media_resp.status_code == 200, media_resp.text
            with open(video_local, 'rb') as vf:
                assert media_resp.content == vf.read()
        else:
            assert result['video'].startswith('file:')
            assert not video_local.startswith(media_dir + os.sep)

        paths = t.where(t.id == 1).select(resized=t.resized.localpath, thumbnail=t.thumbnail.localpath).collect()[0]
        for col in ('resized', 'thumbnail'):
            assert '/media/' in result[col]
            media_resp = client.get(result[col])
            assert media_resp.status_code == 200, f'{col}: {media_resp.text}'
            with open(paths[col], 'rb') as mf:
                assert media_resp.content == mf.read()

        # /resize route (row id 2): single-output JSON response
        resp = post('/resize', 2, width=160)
        assert resp.status_code == 200, resp.text
        job = resp.json()
        result = await_job(job)['result']
        # single-output response model: only 'resized' is present
        assert set(result.keys()) == {'resized'}, result
        assert '/media/' in result['resized']
        resize_local = t.where(t.id == 2).select(p=t.resized.localpath).collect()[0]['p']
        media_resp = client.get(result['resized'])
        assert media_resp.status_code == 200, media_resp.text
        with open(resize_local, 'rb') as rf:
            assert media_resp.content == rf.read()

    def test_openapi(self, uses_db: None) -> None:
        """Verify the generated OpenAPI schema reflects column comments, column types, and route shapes."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        # non-computed columns carry comments via the dict-form ColumnSpec
        t = pxt.create_table(
            'test_serve.openapi',
            schema={
                'id': {'type': pxt.Int, 'comment': 'unique row identifier'},
                'prompt': {'type': pxt.String, 'comment': 'input text prompt'},
                'image': {'type': pxt.Image, 'comment': 'source image'},
            },
        )
        # add_computed_column() currently offers no way to set a column comment, so the
        # computed column goes in uncommented - the openapi test only checks that its
        # description is absent (i.e. not fabricated).
        t.add_computed_column(rotated=t.image.rotate(90))

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        router.add_insert_route(t, path='/json')
        router.add_insert_route(t, path='/upload', uploadfile_inputs=['image'])
        router.add_insert_route(t, path='/file', outputs=['rotated'], return_fileresponse=True)
        router.add_insert_route(t, path='/bg', background=True)
        app.include_router(router)
        client = TestClient(app)

        spec = client.get('/openapi.json').json()
        paths = spec['paths']
        schemas = spec['components']['schemas']

        # routes present
        # note: Starlette's `:path` converter is normalized away in OpenAPI: the route registered
        # as /media/{path:path} appears as /media/{path}.
        for p in ('/json', '/upload', '/file', '/bg', '/jobs/{job_id}', '/media/{path}'):
            assert p in paths, f'missing {p} from openapi paths: {list(paths)}'

        def deref(schema_or_ref: dict[str, Any]) -> dict[str, Any]:
            """
            If schema_or_ref is an OpenAPI $ref into components/schemas, return the referenced schema; otherwise return
            it unchanged.
            """
            if '$ref' in schema_or_ref:
                name = schema_or_ref['$ref'].rsplit('/', 1)[-1]
                return schemas[name]
            return schema_or_ref

        # /json: application/json request body, inputs with comments
        json_body = paths['/json']['post']['requestBody']['content']['application/json']['schema']
        json_body = deref(json_body)
        # all three non-computed columns show up, with their comments as descriptions
        assert set(json_body['properties'].keys()) == {'id', 'prompt', 'image'}
        assert json_body['properties']['id']['description'] == 'unique row identifier'
        assert json_body['properties']['prompt']['description'] == 'input text prompt'
        assert json_body['properties']['image']['description'] == 'source image'

        # /json: response schema is JsonResponse (name derived from path)
        json_resp = paths['/json']['post']['responses']['200']['content']['application/json']['schema']
        json_resp = deref(json_resp)
        assert set(json_resp['properties'].keys()) == {'id', 'prompt', 'image', 'rotated'}
        assert json_resp['properties']['id']['description'] == 'unique row identifier'
        assert json_resp['properties']['prompt']['description'] == 'input text prompt'
        assert json_resp['properties']['image']['description'] == 'source image'
        # media output columns: format: uri + contentMediaType
        for media_col in ('image', 'rotated'):
            prop = json_resp['properties'][media_col]
            assert prop.get('format') == 'uri', f'{media_col}: {prop}'
            assert prop.get('contentMediaType') == 'image/*', f'{media_col}: {prop}'

        # /upload: multipart/form-data request body (File + Form fields)
        upload_content = paths['/upload']['post']['requestBody']['content']
        assert 'multipart/form-data' in upload_content, list(upload_content)
        upload_body = deref(upload_content['multipart/form-data']['schema'])
        assert set(upload_body['properties'].keys()) == {'id', 'prompt', 'image'}
        # File field for image (rendered either as format: binary or contentMediaType: octet-stream
        # depending on pydantic/fastapi version); comment is still propagated.
        image_field = upload_body['properties']['image']
        assert image_field.get('type') == 'string'
        assert (
            image_field.get('format') == 'binary' or image_field.get('contentMediaType') == 'application/octet-stream'
        )
        assert image_field['description'] == 'source image'
        # Form fields keep their comments
        assert upload_body['properties']['id']['description'] == 'unique row identifier'
        assert upload_body['properties']['prompt']['description'] == 'input text prompt'

        # /file: FileResponse route - response_class=FileResponse, no JSON model
        file_resp = paths['/file']['post']['responses']['200']
        # FastAPI renders a FileResponse route with no application/json schema on 200
        assert 'application/json' not in file_resp.get('content', {}), file_resp

        # /bg: background route returns BackgroundJobResponse
        bg_resp = paths['/bg']['post']['responses']['200']['content']['application/json']['schema']
        assert bg_resp.get('$ref', '').endswith('/BackgroundJobResponse'), bg_resp
        assert 'BackgroundJobResponse' in schemas
        bg_model = schemas['BackgroundJobResponse']
        assert set(bg_model['properties'].keys()) == {'id', 'job_url'}

        # /jobs/{job_id}: GET returns JobStatusResponse
        jobs_op = paths['/jobs/{job_id}']['get']
        jobs_resp = jobs_op['responses']['200']['content']['application/json']['schema']
        assert jobs_resp.get('$ref', '').endswith('/JobStatusResponse'), jobs_resp
        assert 'JobStatusResponse' in schemas
        job_status = schemas['JobStatusResponse']
        assert set(job_status['properties'].keys()) == {'status', 'error', 'result'}
        # path parameter is declared
        p0 = jobs_op['parameters'][0]
        assert p0['name'] == 'job_id' and p0['in'] == 'path'

        # /media/{path}: path parameter declared
        media_op = paths['/media/{path}']['get']
        p0 = media_op['parameters'][0]
        assert p0['name'] == 'path' and p0['in'] == 'path'

    def test_add_query_route_scalars(self, uses_db: None) -> None:
        """Multi-column scalar query route, plus retrieval_udf flavor and registration errors."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table('test_serve.docs', {'id': pxt.Int, 'text': pxt.String})
        t.add_computed_column(length=t.text.len())
        rows = [{'id': i, 'text': 'x' * i} for i in range(1, 6)]
        t.insert(rows)

        @pxt.query
        def lookup(min_len: int) -> pxt.Query:
            return t.where(t.length >= min_len).select(t.id, t.text).order_by(t.id)

        @pxt.query
        def lookup2(min_len: int, max_len: int) -> pxt.Query:
            return t.where((t.length >= min_len) & (t.length <= max_len)).select(t.id, t.text).order_by(t.id)

        @pxt.query
        def lookup_with_default(min_len: int = 3) -> pxt.Query:
            return t.where(t.length >= min_len).select(t.id, t.text).order_by(t.id)

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        router.add_query_route(path='/lookup', query=lookup)
        router.add_query_route(path='/lookup-id-only', query=lookup)
        # inputs=[...] restricts which parameters the endpoint accepts
        router.add_query_route(path='/lookup-restricted', query=lookup2, inputs=['min_len'])
        # query with a parameter default
        router.add_query_route(path='/lookup-default', query=lookup_with_default)
        # retrieval_udf variant: all columns from the table are returned, one parameter
        id_lookup = pxt.retrieval_udf(t, parameters=['id'])
        router.add_query_route(path='/by-id', query=id_lookup)
        app.include_router(router)
        client = TestClient(app)

        # /lookup: all rows with length >= 3 -> rows 3,4,5
        resp = client.post('/lookup', json={'min_len': 3})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            'rows': [{'id': 3, 'text': 'xxx'}, {'id': 4, 'text': 'xxxx'}, {'id': 5, 'text': 'xxxxx'}]
        }

        # /lookup-id-only: same query, all columns in the response
        resp = client.post('/lookup-id-only', json={'min_len': 4})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'rows': [{'id': 4, 'text': 'xxxx'}, {'id': 5, 'text': 'xxxxx'}]}

        # retrieval_udf: fetch row by id -> single-row list with all columns
        resp = client.post('/by-id', json={'id': 2})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'rows': [{'id': 2, 'text': 'xx', 'length': 2}]}

        # empty result -> empty list (not an error)
        resp = client.post('/lookup', json={'min_len': 100})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'rows': []}

        # /lookup-restricted: inputs=['min_len'] restricts the endpoint to only accept min_len;
        # max_len should not appear in the OpenAPI schema
        openapi = client.get('/openapi.json').json()
        ref = openapi['paths']['/lookup-restricted']['post']['requestBody']['content']['application/json']['schema'][
            '$ref'
        ]
        schema_name = ref.split('/')[-1]
        restricted_schema = openapi['components']['schemas'][schema_name]
        assert 'min_len' in restricted_schema['properties']
        assert 'max_len' not in restricted_schema['properties']

        # /lookup-default: parameter default propagates to the endpoint
        # calling without min_len should use the default (3) -> rows 3,4,5
        resp = client.post('/lookup-default', json={})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            'rows': [{'id': 3, 'text': 'xxx'}, {'id': 4, 'text': 'xxxx'}, {'id': 5, 'text': 'xxxxx'}]
        }
        # calling with an explicit value overrides the default
        resp = client.post('/lookup-default', json={'min_len': 4})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'rows': [{'id': 4, 'text': 'xxxx'}, {'id': 5, 'text': 'xxxxx'}]}
        # OpenAPI should show the default value and min_len should not be required
        ref = openapi['paths']['/lookup-default']['post']['requestBody']['content']['application/json']['schema'][
            '$ref'
        ]
        default_schema = openapi['components']['schemas'][ref.split('/')[-1]]
        assert default_schema['properties']['min_len']['default'] == 3
        assert 'min_len' not in default_schema.get('required', [])

    def test_add_query_route_single_column(self, uses_db: None) -> None:
        """Single-column queries: return_scalar=False produces dict-per-row in a wrapper,
        return_scalar=True produces a plain list of scalar values."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table('test_serve.docs', {'id': pxt.Int, 'text': pxt.String})
        t.insert([{'id': i, 'text': f't{i}'} for i in range(3)])

        @pxt.query
        def all_texts() -> pxt.Query:
            return t.select(t.text).order_by(t.id)

        @pxt.query(return_scalar=True)
        def all_texts_scalar() -> pxt.Query:
            return t.select(t.text).order_by(t.id)

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        router.add_query_route(path='/texts', query=all_texts)
        router.add_query_route(path='/texts-scalar', query=all_texts_scalar)
        app.include_router(router)
        client = TestClient(app)

        # return_scalar=False (default): dict-per-row wrapped in {'rows': [...]}
        resp = client.post('/texts', json={})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'rows': [{'text': 't0'}, {'text': 't1'}, {'text': 't2'}]}

        # return_scalar=True: plain JSON array of scalar values
        resp = client.post('/texts-scalar', json={})
        assert resp.status_code == 200, resp.text
        assert resp.json() == ['t0', 't1', 't2']

    def test_add_query_route_image(self, uses_db: None) -> None:
        """Image query route: JSON response, return_fileresponse (happy/404/500), and background."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        image_path = get_image_files()[0]
        pxt.create_dir('test_serve')
        t = pxt.create_table('test_serve.images', {'id': pxt.Int, 'image': pxt.Image})
        # A computed resize produces derived media stored under media_dir, which the route will
        # rewrite to /media/ URLs. The raw `image` column stays at its pinned external path.
        t.add_computed_column(resized=t.image.resize(size=(32, 32)))
        # Delay every computed-column eval so the background polling loop can observe 'pending'.
        t.add_computed_column(delay=sleep(1.0))
        t.insert([{'id': 1, 'image': image_path}, {'id': 2, 'image': image_path}])

        @pxt.query
        def one_image(img_id: int) -> pxt.Query:
            return t.where(t.id == img_id).select(t.resized)

        @pxt.query
        def all_images() -> pxt.Query:
            return t.select(t.resized).order_by(t.id)

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        # JSON variant: list of {'image': <media url>}
        router.add_query_route(path='/all-json', query=all_images)
        # FileResponse variant: exactly one row
        router.add_query_route(path='/one-file', query=one_image, return_fileresponse=True)
        # FileResponse variant with >1 row -> 409
        router.add_query_route(path='/all-file', query=all_images, return_fileresponse=True)
        # Background variant
        router.add_query_route(path='/one-bg', query=one_image, background=True)
        app.include_router(router)
        client = TestClient(app)

        resized_locals = {row['id']: row['p'] for row in t.select(t.id, p=t.resized.localpath).collect()}

        # JSON variant: wrapper with rows containing objects with 'resized' fields rewritten as media URLs
        resp = client.post('/all-json', json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert 'rows' in body
        assert len(body['rows']) == 2
        for item in body['rows']:
            assert '/media/' in item['resized'], item['resized']
            media_resp = client.get(item['resized'])
            assert media_resp.status_code == 200

        # FileResponse: exactly one matching row -> image bytes
        resp = client.post('/one-file', json={'img_id': 1})
        assert resp.status_code == 200, resp.text
        assert resp.headers['content-type'].startswith('image/')
        with open(resized_locals[1], 'rb') as f:
            assert resp.content == f.read()

        # FileResponse: 0 matching rows -> 404
        resp = client.post('/one-file', json={'img_id': 999})
        assert resp.status_code == 404, resp.text

        # FileResponse: >1 row -> 409
        resp = client.post('/all-file', json={})
        assert resp.status_code == 409, resp.text
        assert 'expected exactly 1' in resp.json()['detail']

        # Background variant: poll /jobs/{id} until done
        resp = client.post('/one-bg', json={'img_id': 1})
        assert resp.status_code == 200, resp.text
        job = resp.json()
        assert isinstance(job.get('id'), str) and '/jobs/' in job['job_url']
        saw_pending = False
        deadline = time.time() + 30.0
        while True:
            status_resp = client.get(job['job_url'])
            assert status_resp.status_code == 200
            st = status_resp.json()
            if st['status'] == 'pending':
                saw_pending = True
                assert time.time() < deadline, f'job {job["id"]} still pending after 30s'
                time.sleep(0.05)
                continue
            assert st['status'] == 'done', st
            break
        assert saw_pending, 'polling never observed pending - sleep column not delaying'
        result = st['result']
        assert isinstance(result, dict) and 'rows' in result
        assert len(result['rows']) == 1
        assert '/media/' in result['rows'][0]['resized']

    def test_add_query_route_errors(self, uses_db: None) -> None:
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table('test_serve.docs', {'id': pxt.Int, 'text': pxt.String, 'image': pxt.Image})
        t.insert([{'id': 1, 'text': 'a'}])

        @pxt.query
        def lookup(min_id: int) -> pxt.Query:
            return t.where(t.id >= min_id).select(t.id, t.text)

        @pxt.query
        def with_default(min_id: int = 0) -> pxt.Query:
            return t.select(t.id)

        @pxt.query
        def by_text(needle: str) -> pxt.Query:
            return t.where(t.text == needle).select(t.image)

        @pxt.query
        def by_image(img: pxt.Image) -> pxt.Query:
            return t.where(t.image == img).select(t.id)

        router = FastAPIRouter()

        with pytest.raises(pxt.Error, match=r'must be a @pxt\.query or retrieval_udf'):
            router.add_query_route(path='/e', query=add_one)  # regular UDF, not a query
        with pytest.raises(pxt.Error, match="unknown input parameter 'doesnotexist'"):
            router.add_query_route(path='/e', query=lookup, inputs=['doesnotexist'])
        with pytest.raises(pxt.Error, match="unknown uploadfile input parameter 'doesnotexist'"):
            router.add_query_route(path='/e', query=lookup, uploadfile_inputs=['doesnotexist'])
        with pytest.raises(pxt.Error, match="uploadfile input parameter 'min_id' is not a media parameter"):
            router.add_query_route(path='/e', query=lookup, uploadfile_inputs=['min_id'])
        with pytest.raises(pxt.Error, match='return_fileresponse and background are mutually exclusive'):
            router.add_query_route(path='/e', query=by_text, return_fileresponse=True, background=True)
        with pytest.raises(pxt.Error, match='exactly one media-typed output column'):
            # by_text returns a single media column; lookup returns (id, text) which is not media-typed
            router.add_query_route(path='/e', query=lookup, return_fileresponse=True)
        with pytest.raises(pxt.Error, match='GET endpoints cannot have uploadfile_inputs'):
            router.add_query_route(path='/e', query=by_image, uploadfile_inputs=['img'], method='get')

    def test_add_insert_route_errors(self, uses_db: None) -> None:
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table(
            'test_serve.errors', {'id': pxt.Int, 'text': pxt.String, 'image': pxt.Image, 'video': pxt.Video}
        )
        # a scalar computed column and a media computed column, so we can test the
        # "computed column cannot be used as input" check on both code paths
        t.add_computed_column(text_upper=t.text.upper())
        t.add_computed_column(frame=t.video.extract_frame(timestamp=0.0))

        router = FastAPIRouter()

        with pytest.raises(pxt.Error, match='cannot insert into'):
            v = pxt.create_view('test_serve.errors_view', t)
            router.add_insert_route(v, path='/v')
        with pytest.raises(pxt.Error, match="unknown input column 'doesnotexist'"):
            router.add_insert_route(t, path='/e', inputs=['doesnotexist'])
        with pytest.raises(pxt.Error, match="unknown uploadfile input column 'doesnotexist'"):
            router.add_insert_route(t, path='/e', uploadfile_inputs=['doesnotexist'])
        with pytest.raises(pxt.Error, match="'text_upper' is a computed column"):
            router.add_insert_route(t, path='/e', inputs=['text_upper'])
        with pytest.raises(pxt.Error, match="uploadfile input column 'text' is not a media column"):
            router.add_insert_route(t, path='/e', uploadfile_inputs=['text'])
        with pytest.raises(pxt.Error, match="'frame' is a computed column"):
            router.add_insert_route(t, path='/e', uploadfile_inputs=['frame'])
        with pytest.raises(pxt.Error, match="'image' appears in both `inputs` and `uploadfile_inputs`"):
            router.add_insert_route(t, path='/e', inputs=['image'], uploadfile_inputs=['image'])
        with pytest.raises(pxt.Error, match="unknown output column 'doesnotexist'"):
            router.add_insert_route(t, path='/e', outputs=['doesnotexist'])
        with pytest.raises(pxt.Error, match='return_fileresponse and background are mutually exclusive'):
            router.add_insert_route(t, path='/e', outputs=['frame'], return_fileresponse=True, background=True)
        with pytest.raises(pxt.Error, match='exactly one media-typed output column'):
            router.add_insert_route(t, path='/e', outputs=['id', 'frame'], return_fileresponse=True)
        with pytest.raises(pxt.Error, match='exactly one media-typed output column'):
            router.add_insert_route(t, path='/e', outputs=['text_upper'], return_fileresponse=True)

    def test_add_delete_route(self, uses_db: None) -> None:
        """Delete routes: primary-key default, explicit match_columns, multi-col AND, 0-match, background."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table(
            'test_serve.items', {'id': pxt.Required[pxt.Int], 'group': pxt.String, 'value': pxt.Int}, primary_key='id'
        )
        t.insert(
            [
                {'id': 1, 'group': 'a', 'value': 10},
                {'id': 2, 'group': 'a', 'value': 20},
                {'id': 3, 'group': 'b', 'value': 10},
                {'id': 4, 'group': 'b', 'value': 20},
                {'id': 5, 'group': 'c', 'value': 30},
            ]
        )

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        router.add_delete_route(t, path='/by-pk')  # defaults to primary key
        router.add_delete_route(t, path='/by-group', match_columns=['group'])
        router.add_delete_route(t, path='/by-group-value', match_columns=['group', 'value'])
        router.add_delete_route(t, path='/by-pk-bg', background=True)
        app.include_router(router)
        client = TestClient(app)

        # /by-pk: default pk match deletes exactly one row
        resp = client.post('/by-pk', json={'id': 1})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'num_rows': 1}
        assert t.where(t.id == 1).count() == 0

        # /by-pk: 0 matches is not an error
        resp = client.post('/by-pk', json={'id': 9999})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'num_rows': 0}

        # /by-group: explicit single-col match, deletes all rows in the group
        # group 'a' started with 2 rows (ids 1, 2); id 1 already deleted -> 1 row remains
        resp = client.post('/by-group', json={'group': 'a'})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'num_rows': 1}
        assert t.where(t.group == 'a').count() == 0

        # /by-group-value: multi-col AND; only rows where BOTH match are deleted
        # group 'b' has ids 3,4 with values 10,20; match (group='b', value=10) -> 1 row
        resp = client.post('/by-group-value', json={'group': 'b', 'value': 10})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {'num_rows': 1}
        # the other 'b' row (id=4, value=20) should remain
        assert t.where(t.group == 'b').count() == 1

        # /by-pk-bg: background variant
        resp = client.post('/by-pk-bg', json={'id': 5})
        assert resp.status_code == 200, resp.text
        job = resp.json()
        assert isinstance(job.get('id'), str) and '/jobs/' in job['job_url']
        deadline = time.time() + 10.0
        while True:
            status_resp = client.get(job['job_url'])
            assert status_resp.status_code == 200
            st = status_resp.json()
            if st['status'] == 'pending':
                assert time.time() < deadline, 'job still pending'
                time.sleep(0.02)
                continue
            assert st['status'] == 'done', st
            break
        assert st['result'] == {'num_rows': 1}
        assert t.where(t.id == 5).count() == 0

    def test_add_delete_route_errors(self, uses_db: None) -> None:
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        t = pxt.create_table('test_serve.items', {'id': pxt.Required[pxt.Int], 'group': pxt.String}, primary_key='id')
        t_no_pk = pxt.create_table('test_serve.nopk', {'id': pxt.Int, 'group': pxt.String})

        router = FastAPIRouter()

        with pytest.raises(pxt.Error, match='cannot delete from'):
            v = pxt.create_view('test_serve.items_view', t)
            router.add_delete_route(v, path='/v')
        with pytest.raises(pxt.Error, match="unknown column 'doesnotexist'"):
            router.add_delete_route(t, path='/e', match_columns=['doesnotexist'])
        with pytest.raises(pxt.Error, match='`match_columns` must be non-empty'):
            router.add_delete_route(t, path='/e', match_columns=[])
        with pytest.raises(pxt.Error, match='table has no primary key'):
            router.add_delete_route(t_no_pk, path='/e')

    @pytest.mark.parametrize(
        ('op_name', 'first_body', 'retry_body'),
        [('insert', {'id': 2, 'val': 20}, {'id': 3, 'val': 30}), ('delete', {'id': 1}, {'id': 2})],
    )
    @pytest.mark.parametrize('schema_op', ['add_column', 'drop'])
    def test_schema_change(
        self, uses_db: None, op_name: str, first_body: dict[str, Any], retry_body: dict[str, Any], schema_op: str
    ) -> None:
        """Schema-version bump or drop-and-recreate after route registration causes the handler to 409."""
        skip_test_if_not_installed('fastapi')
        import fastapi
        from fastapi.testclient import TestClient

        from pixeltable.serving import FastAPIRouter

        pxt.create_dir('test_serve')
        schema = {'id': pxt.Required[pxt.Int], 'val': pxt.Int}
        t = pxt.create_table('test_serve.items', schema, primary_key='id')
        t.insert([{'id': 1, 'val': 10}])

        app = fastapi.FastAPI()
        router = FastAPIRouter()
        if op_name == 'insert':
            router.add_insert_route(t, path='/ep')
        else:
            router.add_delete_route(t, path='/ep')
        app.include_router(router)
        client = TestClient(app)

        # baseline: endpoint works before schema change
        resp = client.post('/ep', json=first_body)
        assert resp.status_code == 200, resp.text

        if schema_op == 'add_column':
            # mutate the schema in place; bumps schema_version behind the route's back
            t.add_computed_column(val_plus_1=t.val + 1)
        else:
            # drop and recreate at the same path; new table has a fresh UUID
            pxt.drop_table('test_serve.items', force=True)
            pxt.create_table('test_serve.items', schema, primary_key='id')

        # handler now detects the mismatch and rejects the request
        resp = client.post('/ep', json=retry_body)
        assert resp.status_code == 409, resp.text
        assert 'table schema changed' in resp.json()['detail']
