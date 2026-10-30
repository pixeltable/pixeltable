import asyncio
from base64 import b64decode
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.functions.audio import audio_splitter
from pixeltable.functions.video import video_splitter
from pixeltable.runtime import get_runtime, reset_runtime

from ..utils import (
    get_audio_files,
    get_image_files,
    get_video_files,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.fixture
def mock_client(init_env: None, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    skip_test_if_not_installed('twelvelabs')
    from pixeltable.functions import twelvelabs

    data = [SimpleNamespace(embedding=[0.5] * 512)]
    client = SimpleNamespace(
        embed=SimpleNamespace(
            v_2=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(data=data)),
                tasks=SimpleNamespace(
                    create=AsyncMock(return_value=SimpleNamespace(id='test-task')),
                    retrieve=AsyncMock(return_value=SimpleNamespace(status='ready', data=data, error=None)),
                ),
            )
        ),
        multipart_upload=SimpleNamespace(upload_file=AsyncMock(return_value=SimpleNamespace(asset_id='test-asset'))),
        assets=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(status='ready', error=None)), delete=AsyncMock()
        ),
    )
    monkeypatch.setattr(twelvelabs, '_twelvelabs_client', lambda: client)
    monkeypatch.setattr(twelvelabs, '_POLL_INTERVAL_SECS', 0.0)
    return client


@pytest.fixture
def mock_embeddings(mock_client: SimpleNamespace) -> AsyncMock:
    return mock_client.embed.v_2.create


def write_media(path: Path, size: int) -> str:
    with path.open('wb') as fp:
        fp.write(b'media')
        fp.truncate(size)
    return str(path)


class TestTwelveLabsClient:
    def test_client_closed_on_reset(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        monkeypatch.setenv('TWELVELABS_API_KEY', 'test-key')
        # no public accessor exposes the SDK's httpx client, so this reads the private attribute that holds it
        http_client = get_runtime().get_client('twelvelabs')._client_wrapper.httpx_client.httpx_client
        reset_runtime()
        assert http_client.is_closed


class TestTwelveLabsLocal:
    @pytest.mark.parametrize('model_name', ['marengo3.0', 'marengo3.5'])
    @pytest.mark.parametrize('input_type', ['text', 'text_image', 'image', 'audio', 'video'])
    def test_embed(
        self,
        mock_embeddings: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        model_name: str,
        input_type: str,
    ) -> None:
        from pixeltable.functions import twelvelabs

        kwargs: dict[str, Any] = {'model_name': model_name}
        image = PIL.Image.new('RGB', (16, 16))
        signature_idx = 0
        if input_type in ('text', 'text_image'):
            kwargs['text'] = 'A red car'
            if input_type == 'text_image':
                kwargs['image'] = image
        elif input_type == 'image':
            signature_idx = 1
            kwargs['image'] = image
        else:
            signature_idx = 2 if input_type == 'audio' else 3
            media = tmp_path / 'media'
            media.write_bytes(b'media')
            kwargs[input_type] = str(media)
            monkeypatch.setattr(twelvelabs.av_utils, f'get_{input_type}_duration', lambda _: 30.0)
            if model_name == 'marengo3.0':
                kwargs.update(start_sec=1.0, end_sec=10.0, embedding_option=['audio'])

        vector = asyncio.run(twelvelabs.embed.py_fns[signature_idx](**kwargs))
        assert vector.shape == (512,)
        assert vector.dtype == np.float32
        request = mock_embeddings.call_args.kwargs
        assert request['model_name'] == model_name
        if model_name == 'marengo3.5':
            assert request['input_type'] == 'multi_input'
            assert request['embedding_dimension'] == 512
            # .dict() is the body the SDK encodes. An omitted field stays off the wire; an explicit None becomes null.
            encoded = request['multi_input'].dict()
            if input_type == 'text':
                assert encoded == {'input_text': kwargs['text']}
            else:
                sources = encoded['media_sources']
                assert len(sources) == 1
                source = sources[0]
                assert source['media_type'] == ('image' if input_type == 'text_image' else input_type)
                assert source['base64_string']
                if input_type == 'text_image':
                    assert encoded['input_text'] == kwargs['text']
                    assert set(encoded) == {'input_text', 'media_sources'}
                else:
                    assert set(encoded) == {'media_sources'}
                if input_type in ('audio', 'video'):
                    assert b64decode(source['base64_string']) == b'media'
        else:
            assert request['input_type'] == input_type
            assert 'embedding_dimension' not in request
            if input_type in ('text', 'text_image'):
                assert request[input_type].input_text == kwargs['text']
            elif input_type in ('audio', 'video'):
                assert request[input_type].start_sec == pytest.approx(1.0)
                assert request[input_type].end_sec == pytest.approx(10.0)
                assert request[input_type].embedding_option == ['audio']

    @pytest.mark.parametrize('input_type', ['audio', 'video'])
    def test_unknown_duration(
        self, mock_embeddings: AsyncMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_type: str
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = write_media(tmp_path / 'media', 5)
        monkeypatch.setattr(twelvelabs.av_utils, f'get_{input_type}_duration', lambda _: None)
        with pxt_raises(pxt.ErrorCode.INVALID_DATA_FORMAT, match=f'Cannot determine {input_type} duration') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[2 if input_type == 'audio' else 3](media, model_name='marengo3.5'))
        assert str(tmp_path) not in str(exc_info.value)
        mock_embeddings.assert_not_called()

    @pytest.mark.parametrize('input_type', ['audio', 'video'])
    @pytest.mark.parametrize(
        ('size', 'duration'),
        [
            (5, 30.01),
            # a duration of None would raise if it were probed
            (32 * 10**6 + 1, None),
            (36 * 10**6 + 1, None),
        ],
    )
    def test_embed_task(
        self,
        mock_client: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        input_type: str,
        size: int,
        duration: float | None,
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = write_media(tmp_path / 'media', size)
        monkeypatch.setattr(twelvelabs.av_utils, f'get_{input_type}_duration', lambda _: duration)
        vector = asyncio.run(twelvelabs.embed.py_fns[2 if input_type == 'audio' else 3](media, model_name='marengo3.5'))
        assert vector.shape == (512,)
        mock_client.embed.v_2.create.assert_not_called()
        tasks = mock_client.embed.v_2.tasks
        request = tasks.create.call_args.kwargs
        assert request['input_type'] == input_type
        assert request['model_name'] == 'marengo3.5'
        assert request['embedding_dimension'] == 512
        encoded = request[input_type].dict()
        assert encoded['embedding_scope'] == ['asset']
        assert encoded['embedding_option'] == ['audio' if input_type == 'audio' else 'visual']
        if size > 36 * 10**6:
            assert encoded['media_source'] == {'asset_id': 'test-asset'}
            mock_client.assets.delete.assert_awaited_once_with('test-asset')
        else:
            assert b64decode(encoded['media_source']['base64_string']) == Path(media).read_bytes()
            mock_client.multipart_upload.upload_file.assert_not_called()
        tasks.retrieve.assert_awaited_once_with('test-task')

    @pytest.mark.parametrize('model_name', ['marengo3.0', 'marengo3.5'])
    def test_asset_ready(self, mock_client: SimpleNamespace, tmp_path: Path, model_name: str) -> None:
        from pixeltable.functions import twelvelabs

        v2 = mock_client.embed.v_2
        create = v2.create if model_name == 'marengo3.0' else v2.tasks.create
        statuses = ['processing', 'processing', 'ready']

        # the asset is used only once it is ready
        def retrieve(asset_id: str) -> SimpleNamespace:
            assert asset_id == 'test-asset'
            create.assert_not_called()
            return SimpleNamespace(status=statuses.pop(0), error=None)

        mock_client.assets.retrieve.side_effect = retrieve
        media = write_media(tmp_path / 'media', 36 * 10**6 + 1)
        vector = asyncio.run(twelvelabs.embed.py_fns[3](media, model_name=model_name))
        assert vector.shape == (512,)
        assert statuses == []
        assert create.call_args.kwargs['video'].media_source.asset_id == 'test-asset'
        mock_client.assets.delete.assert_awaited_once_with('test-asset')

    @pytest.mark.parametrize('stage', ['asset', 'embedding task'])
    def test_poll_failed(self, mock_client: SimpleNamespace, tmp_path: Path, stage: str) -> None:
        from pixeltable.functions import twelvelabs

        failed = SimpleNamespace(status='failed', data=None, error=SimpleNamespace(message='corrupt file'))
        if stage == 'asset':
            mock_client.assets.retrieve.return_value = failed
        else:
            mock_client.embed.v_2.tasks.retrieve.return_value = failed
        media = write_media(tmp_path / 'media', 36 * 10**6 + 1)
        test_id = 'test-asset' if stage == 'asset' else 'test-task'
        with pxt_raises(pxt.ErrorCode.PROVIDER_BAD_REQUEST, match=f'TwelveLabs {stage} {test_id} failed: corrupt file'):
            asyncio.run(twelvelabs.embed.py_fns[3](media, model_name='marengo3.5'))
        assert mock_client.embed.v_2.tasks.create.await_count == (0 if stage == 'asset' else 1)
        mock_client.assets.delete.assert_awaited_once_with('test-asset')

    def test_poll_timeout(self, mock_client: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.functions import twelvelabs

        monkeypatch.setattr(twelvelabs, '_POLL_TIMEOUT_SECS', 0.01)
        mock_client.embed.v_2.tasks.retrieve.return_value = SimpleNamespace(status='processing', data=None, error=None)
        audio = get_audio_files()[0]  # 60 seconds
        with pxt_raises(
            pxt.ErrorCode.PROVIDER_TIMEOUT, match='embedding task test-task was not ready after 0.01 seconds'
        ):
            asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name='marengo3.5'))

    def test_poll_errors(self, mock_client: SimpleNamespace) -> None:
        from twelvelabs.errors import NotFoundError, TooManyRequestsError

        from pixeltable.functions import twelvelabs

        tasks = mock_client.embed.v_2.tasks
        ready = tasks.retrieve.return_value
        audio = get_audio_files()[0]  # 60 seconds

        # a rate-limited retrieve is polled again
        tasks.retrieve.side_effect = [TooManyRequestsError(body=None), ready]
        vector = asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name='marengo3.5'))
        assert vector.shape == (512,)
        assert tasks.retrieve.await_count == 2

        # any other error is final
        tasks.retrieve.side_effect = [NotFoundError(body=None), ready]
        with pytest.raises(NotFoundError):
            asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name='marengo3.5'))

    @pytest.mark.parametrize('input_type', ['audio', 'video'])
    @pytest.mark.parametrize('options', [{'start_sec': 0.0}, {'end_sec': 5.0}, {'embedding_option': ['audio']}])
    def test_unsupported_options(self, mock_embeddings: AsyncMock, input_type: str, options: dict[str, Any]) -> None:
        from pixeltable.functions import twelvelabs

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='do not support start_sec, end_sec, or embedding_option'):
            asyncio.run(
                twelvelabs.embed.py_fns[2 if input_type == 'audio' else 3]('unused', model_name='marengo3.5', **options)
            )
        mock_embeddings.assert_not_called()

    @pytest.mark.parametrize('vectors', [[], [[0.0] * 256], [[0.0] * 512, [1.0] * 512]])
    def test_invalid_response(self, mock_embeddings: AsyncMock, vectors: list[list[float]]) -> None:
        from pixeltable.functions import twelvelabs

        mock_embeddings.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='single 512-dimensional embedding'):
            asyncio.run(twelvelabs.embed.py_fns[0](text='test', model_name='marengo3.5'))

    @pytest.mark.parametrize('model_name', ['marengo3.0', 'marengo3.5'])
    def test_long_audio(self, mock_client: SimpleNamespace, model_name: str) -> None:
        from pixeltable.functions import twelvelabs

        audio = get_audio_files()[0]  # 60 seconds
        vector = asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name=model_name))
        assert vector.shape == (512,)
        # Marengo 3.5 embeds audio over 30 seconds with an embedding task
        v2 = mock_client.embed.v_2
        assert (v2.create.await_count, v2.tasks.create.await_count) == (
            (1, 0) if model_name == 'marengo3.0' else (0, 1)
        )

    def test_sdk_version(self, mock_embeddings: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable.functions import twelvelabs

        def require_package(self: Any, package_name: str, min_version: list[int] | None = None) -> None:
            if min_version is not None:
                assert min_version == [1, 3, 6]
                raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, 'twelvelabs>=1.3.6 is required')

        monkeypatch.setattr(twelvelabs.env.Env, 'require_package', require_package)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'twelvelabs>=1\.3\.6'):
            asyncio.run(twelvelabs.embed.py_fns[0](text='test', model_name='marengo3.5'))
        mock_embeddings.assert_not_called()
        asyncio.run(twelvelabs.embed.py_fns[0](text='test', model_name='marengo3.0'))
        mock_embeddings.assert_awaited_once()

    def test_embedding_indexes(self, uses_db: None, mock_embeddings: AsyncMock) -> None:
        from pixeltable.functions import twelvelabs

        t = pxt.create_table(
            'test_tbl', {'text': pxt.String, 'image': pxt.Image, 'audio': pxt.Audio, 'video': pxt.Video}
        )
        embedding = twelvelabs.embed.using(model_name='marengo3.5')
        for column in ('text', 'image', 'audio', 'video'):
            t.add_embedding_index(column, embedding=embedding)
        validate_update_status(
            t.insert(
                text='test',
                image=get_image_files()[0],
                audio=get_audio_files(extension='wav')[0],
                video=get_video_files()[0],
            ),
            1,
        )
        for column in ('text', 'image', 'audio', 'video'):
            res = t.select(vector=t[column].embedding()).collect()
            assert res['vector'][0].shape == (512,)
        res = t.select(similarity=t.video.similarity(string='test')).collect()
        assert res['similarity'][0] == pytest.approx(1.0)


@pytest.mark.remote_api
@pytest.mark.very_expensive
@pytest.mark.parametrize('model_name', ['marengo3.0', 'marengo3.5'])
@rerun_on_network_error()
class TestTwelveLabs:
    def test_embed_text(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        t = pxt.create_table('test_tbl', {'input': pxt.String | None, 'image': pxt.Image | None})
        t.add_computed_column(embed=embed(model_name=model_name, text=t.input, image=t.image))
        images = get_image_files()
        rows = [
            {'input': 'Twelve Labs provides multimodal embedding models.', 'image': None},
            {'input': 'An optional image can be specified with text embeddings.', 'image': images[0]},
        ]
        validate_update_status(t.insert(rows), 2)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

        t.add_embedding_index(t.input, embedding=embed.using(model_name=model_name))
        res = t.select(embedding=t.input.embedding()).collect()
        assert res['embedding'][0].shape == (512,)

    def test_embed_image(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        image_filepaths = get_image_files()[:2]
        t = pxt.create_table('image_tbl', {'image': pxt.Image | None})
        t.add_computed_column(embed=embed(model_name=model_name, image=t.image))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

        t.add_embedding_index(t.image, embedding=embed.using(model_name=model_name))
        res = t.select(embedding=t.image.embedding()).collect()
        assert res['embedding'][0].shape == (512,)

    def test_embed_audio(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio | None})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths[:1]), expected_rows=1)
        v = pxt.create_view(
            'audio_segments',
            base_t,
            # Twelvelabs models require a minimum audio duration of 4 seconds
            iterator=audio_splitter(base_t.audio, duration=5.0, min_segment_duration=4.0),
        )
        v.add_embedding_index(v.audio_segment, embedding=embed.using(model_name=model_name))
        res = v.select(embedding=v.audio_segment.embedding()).collect()
        assert res['embedding'][0].shape == (512,)

    def test_embed_video(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        video_filepaths = get_video_files()[:1]  # Just send one of them for testing
        base_t = pxt.create_table('video_tbl', {'video': pxt.Video | None})
        validate_update_status(base_t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        v = pxt.create_view(
            'video_segments',
            base_t,
            iterator=video_splitter(video=base_t.video, duration=5.0, min_segment_duration=4.0),
        )
        v.add_embedding_index(v.video_segment, embedding=embed.using(model_name=model_name))
        res = v.select(embedding=v.video_segment.embedding()).collect()
        assert res['embedding'][0].shape == (512,)

    def test_embed_long_audio(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        # 60 seconds, unsplit: Marengo 3.5 embeds it with an embedding task
        t = pxt.create_table('long_audio_tbl', {'audio': pxt.Audio | None})
        validate_update_status(t.insert(audio=get_audio_files()[0]), expected_rows=1)
        t.add_embedding_index(t.audio, embedding=embed.using(model_name=model_name))
        res = t.select(embedding=t.audio.embedding()).collect()
        assert res['embedding'][0].shape == (512,)

    def test_embed_large_media(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        # Test that large media files that require multipart upload can be embedded successfully
        t = pxt.create_table('large_media_tbl', {'video': pxt.Video | None})
        videos = ['s3://pxt-test/pytest-resources/large_videos/6mb.mp4']
        if model_name == 'marengo3.5':
            # over 36 * 10**6 bytes: uploaded for an embedding task
            videos.append('s3://pxt-test/pytest-resources/large_videos/35mb.mp4')
        validate_update_status(t.insert({'video': v} for v in videos), expected_rows=len(videos))
        t.add_embedding_index(t.video, embedding=embed.using(model_name=model_name))
        res = t.select(embedding=t.video.embedding()).collect()
        assert all(e.shape == (512,) for e in res['embedding'])
