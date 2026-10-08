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
def mock_embeddings(init_env: None, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    skip_test_if_not_installed('twelvelabs')
    from pixeltable.functions import twelvelabs

    create = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)]))
    client = SimpleNamespace(
        embed=SimpleNamespace(v_2=SimpleNamespace(create=create)),
        multipart_upload=SimpleNamespace(upload_file=AsyncMock(return_value=SimpleNamespace(asset_id='test-asset'))),
        assets=SimpleNamespace(delete=AsyncMock()),
    )
    monkeypatch.setattr(twelvelabs, '_twelvelabs_client', lambda: client)
    return create


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
            multi_input = request['multi_input']
            assert multi_input.input_text == kwargs.get('text')
            if input_type == 'text':
                assert multi_input.media_sources is None
            else:
                assert len(multi_input.media_sources) == 1
                source = multi_input.media_sources[0]
                assert source.media_type == ('image' if input_type == 'text_image' else input_type)
                assert source.base_64_string
                if input_type in ('audio', 'video'):
                    assert b64decode(source.base_64_string) == b'media'
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
    @pytest.mark.parametrize('duration', [30.01, 60.0, None])
    def test_media_duration(
        self,
        mock_embeddings: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        input_type: str,
        duration: float | None,
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = tmp_path / 'media'
        media.write_bytes(b'media')
        monkeypatch.setattr(twelvelabs.av_utils, f'get_{input_type}_duration', lambda _: duration)
        if duration is None:
            code, message = pxt.ErrorCode.INVALID_DATA_FORMAT, f'Cannot determine {input_type} duration'
        else:
            code, message = pxt.ErrorCode.INVALID_ARGUMENT, 'at most 30 seconds'
        with pxt_raises(code, match=message) as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[2 if input_type == 'audio' else 3](str(media), model_name='marengo3.5'))
        assert str(tmp_path) not in str(exc_info.value)
        mock_embeddings.assert_not_called()

    @pytest.mark.parametrize('input_type', ['audio', 'video'])
    def test_media_size(self, mock_embeddings: AsyncMock, tmp_path: Path, input_type: str) -> None:
        from pixeltable.functions import twelvelabs

        media = tmp_path / 'media'
        with media.open('wb') as fp:
            fp.truncate(32 * 2**20 + 1)
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='at most 32 MB'):
            asyncio.run(twelvelabs.embed.py_fns[2 if input_type == 'audio' else 3](str(media), model_name='marengo3.5'))
        mock_embeddings.assert_not_called()

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
        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='single 512-dimensional embedding'):
            asyncio.run(twelvelabs.embed.py_fns[0](text='test', model_name='marengo3.5'))

    @pytest.mark.parametrize('model_name', ['marengo3.0', 'marengo3.5'])
    def test_long_audio(self, mock_embeddings: AsyncMock, model_name: str) -> None:
        from pixeltable.functions import twelvelabs

        audio = get_audio_files()[0]  # 60 seconds
        if model_name == 'marengo3.5':
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'at most 30 seconds; got 60\.00 seconds'):
                asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name=model_name))
            mock_embeddings.assert_not_called()
        else:
            vector = asyncio.run(twelvelabs.embed.py_fns[2](audio, model_name=model_name))
            assert vector.shape == (512,)

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

    @pytest.mark.skip(reason='feature broken: PXT-1234')
    def test_embed_large_media(self, uses_db: None, model_name: str) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        # Test that large media files that require multipart upload can be embedded successfully
        t = pxt.create_table('large_media_tbl', {'video': pxt.Video | None})
        t.insert(video='s3://pxt-test/pytest-resources/large_videos/6mb.mp4')
        t.add_embedding_index(t.video, embedding=embed.using(model_name=model_name))
        res = t.select(embedding=t.video.embedding()).collect()
        assert res['embedding'][0].shape == (512,)
