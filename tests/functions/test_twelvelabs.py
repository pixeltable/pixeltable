import asyncio
import heapq
import time
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
from pixeltable.utils.http import is_retryable_error

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


class _RetrieveRateLimitError(Exception):
    """Duck-typed TwelveLabs API error for HTTP 429, including an optional Retry-After header."""

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__('too many requests')
        self.status_code = 429
        self.headers = {'Retry-After': retry_after} if retry_after is not None else {}


class _RetrieveRejectedError(Exception):
    """Duck-typed non-429 retrieve failure."""

    def __init__(self, message: str = 'asset missing', status_code: int = 404) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RetrieveStatusError(Exception):
    """Duck-typed retrieve failure with an HTTP status that `is_retryable_error()` recognizes."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f'http {status_code}')
        self.status_code = status_code


class _VirtualClock:
    """Moves time to the next wake once every task is asleep."""

    def __init__(self, n_tasks: int) -> None:
        self.now = 0.0
        self._n_tasks = n_tasks
        self._n_sleeping = 0
        self._seq = 0
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        return self.now

    def task_finished(self) -> None:
        self._n_tasks -= 1
        self._release_next_if_idle()

    async def sleep(self, delay: float) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (self.now + max(delay, 0.0), self._seq, fut))
        self._n_sleeping += 1
        try:
            self._release_next_if_idle()
            await fut
        finally:
            self._n_sleeping -= 1

    def _release_next_if_idle(self) -> None:
        if self._n_tasks <= 0 or self._n_sleeping != self._n_tasks or not self._waiters:
            return
        wake, _seq, fut = heapq.heappop(self._waiters)
        self.now = wake
        if not fut.done():
            fut.set_result(None)


@pytest.fixture
def mock_embeddings(init_env: None, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    skip_test_if_not_installed('twelvelabs')
    from pixeltable.functions import twelvelabs

    create = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)]))
    client = SimpleNamespace(
        embed=SimpleNamespace(v_2=SimpleNamespace(create=create)),
        multipart_upload=SimpleNamespace(upload_file=AsyncMock(return_value=SimpleNamespace(asset_id='test-asset'))),
        assets=SimpleNamespace(
            delete=AsyncMock(), retrieve=AsyncMock(return_value=SimpleNamespace(status='ready', error=None))
        ),
    )
    monkeypatch.setattr(twelvelabs, '_twelvelabs_client', lambda: client)
    return create


class _LargeMediaClient:
    """Upload client and media file shared by the large-media embed tests."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, media: Path) -> None:
        self._monkeypatch = monkeypatch
        self.media = media

    def install(self, *, upload_file: Any, retrieve: Any, create: Any, delete: Any) -> SimpleNamespace:
        from pixeltable.functions import twelvelabs

        client = SimpleNamespace(
            embed=SimpleNamespace(v_2=SimpleNamespace(create=self._mock(create))),
            multipart_upload=SimpleNamespace(upload_file=self._mock(upload_file)),
            assets=SimpleNamespace(retrieve=self._mock(retrieve), delete=self._mock(delete)),
        )
        self._monkeypatch.setattr(twelvelabs, '_twelvelabs_client', lambda: client)
        return client

    @staticmethod
    def _mock(fn: Any) -> AsyncMock:
        return fn if isinstance(fn, AsyncMock) else AsyncMock(side_effect=fn)


@pytest.fixture
def large_media_client(init_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _LargeMediaClient:
    """One TwelveLabs client for embed tests that upload instead of inlining the media."""
    skip_test_if_not_installed('twelvelabs')
    from pixeltable.functions import twelvelabs

    media = tmp_path / 'media.mp4'
    media.write_bytes(b'media')
    monkeypatch.setattr(twelvelabs, 'TWELVELABS_INLINE_LIMIT_BYTES', 0)
    # A retryable create stashes the asset on this process-wide map. Give each test its own.
    monkeypatch.setattr(twelvelabs, '_ready_assets', twelvelabs._ReadyAssets())
    return _LargeMediaClient(monkeypatch, media)


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

    def test_embed_waits_for_ready_asset(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        events: list[str] = []
        statuses = ['processing', 'ready']

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            events.append('upload')
            return SimpleNamespace(asset_id='asset-1')

        def retrieve(asset_id: str) -> SimpleNamespace:
            status = statuses.pop(0)
            events.append(f'retrieve:{status}')
            return SimpleNamespace(status=status, error=None)

        def sleep(delay: float) -> None:
            events.append(f'sleep:{delay}')

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            assert kwargs['video'].media_source.asset_id == 'asset-1'
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)])

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        vector = asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert vector.shape == (512,)
        assert vector.dtype == np.float32
        assert events == [
            'upload',
            'retrieve:processing',
            f'sleep:{twelvelabs._ASSET_POLL_INITIAL_SECS}',
            'retrieve:ready',
            'create',
            'delete:asset-1',
        ]

    def test_embed_failed_asset_is_deleted(self, large_media_client: _LargeMediaClient, tmp_path: Path) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        events: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-429')

        def retrieve(asset_id: str) -> SimpleNamespace:
            events.append('retrieve')
            return SimpleNamespace(status='failed', error=SimpleNamespace(message='processing timed out'))

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)

        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='TwelveLabs asset failed') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        message = str(exc_info.value)
        assert str(tmp_path) not in message
        assert 'asset-429' not in message
        assert '429' not in message
        assert 'timed out' not in message.lower()
        assert is_retryable_error(exc_info.value)[0] is False
        cause = exc_info.value.__cause__
        assert isinstance(cause, twelvelabs._FailedAssetError)
        assert cause.asset_id == 'asset-429'
        assert cause.reason == 'processing timed out'
        assert events == ['retrieve', 'delete:asset-429']

    def test_embed_asset_ready_timeout_deletes_asset(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        events: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-429')

        # The deadline is measured with monotonic time. Jumping it during retrieve exercises the
        # post-poll check without waiting out the real budget.
        clock = {'t': 0.0}

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            events.append(f'retrieve:{asset_id}')
            clock['t'] += twelvelabs._ASSET_READY_TIMEOUT_SECS
            return SimpleNamespace(status='processing', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', 600.0)

        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing after 600s') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        message = str(exc_info.value)
        assert str(tmp_path) not in message
        assert 'asset-429' not in message
        assert '429' not in message
        assert 'timed out' not in message.lower()
        assert is_retryable_error(exc_info.value)[0] is False
        assert events == ['retrieve:asset-429', 'delete:asset-429']

    def test_asset_uploads_polls_concurrently(
        self, init_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        files: list[str] = []
        for name in ('a.mp4', 'b.mp4'):
            path = tmp_path / name
            path.write_bytes(b'media')
            files.append(str(path))

        release = asyncio.Event()
        started: list[str] = []
        deleted: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id=Path(file_path).name)

        async def retrieve(asset_id: str) -> SimpleNamespace:
            started.append(asset_id)
            if len(started) == 1:
                try:
                    await asyncio.wait_for(release.wait(), timeout=1)
                except TimeoutError:
                    raise AssertionError('assets were not polled concurrently') from None
            else:
                release.set()
            return SimpleNamespace(status='ready', error=None)

        def delete(asset_id: str) -> None:
            deleted.append(asset_id)

        client = SimpleNamespace(
            multipart_upload=SimpleNamespace(upload_file=AsyncMock(side_effect=upload_file)),
            assets=SimpleNamespace(retrieve=retrieve, delete=AsyncMock(side_effect=delete)),
        )
        monkeypatch.setattr(twelvelabs, '_twelvelabs_client', lambda: client)

        async def run() -> list[str]:
            async with twelvelabs._asset_uploads('video', files) as asset_ids:
                return list(asset_ids)

        assert asyncio.run(run()) == ['a.mp4', 'b.mp4']
        assert sorted(deleted) == ['a.mp4', 'b.mp4']

    def test_embed_retrieve_past_deadline_deletes_asset(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', 0.2)
        events: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-1')

        async def retrieve(asset_id: str) -> SimpleNamespace:
            events.append(f'retrieve:{asset_id}')
            await asyncio.sleep(5)
            events.append('retrieve-finished')
            return SimpleNamespace(status='ready', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)

        started = time.monotonic()
        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert time.monotonic() - started < 2
        assert str(tmp_path) not in str(exc_info.value)
        assert 'timed out' not in str(exc_info.value).lower()
        assert is_retryable_error(exc_info.value)[0] is False
        assert events == ['retrieve:asset-1', 'delete:asset-1']

    def test_delayed_final_poll_is_bounded(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from pixeltable.functions import twelvelabs

        # The final retrieve is the readiness recheck. It has to start before the deadline and
        # stay inside the budget: a 30ms budget with a 90ms second retrieve used to finish only
        # after that retrieve returned.
        budget = 0.25
        hang = 1.0
        media = large_media_client.media
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', budget)
        monkeypatch.setattr(twelvelabs, '_ASSET_POLL_INITIAL_SECS', 30.0)
        monkeypatch.setattr(twelvelabs, '_ASSET_RETRIEVE_RPS', 1_000_000.0)
        events: list[str] = []
        calls = 0

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            events.append('upload')
            return SimpleNamespace(asset_id='asset-1')

        async def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            events.append(f'retrieve:{asset_id}')
            if calls == 1:
                return SimpleNamespace(status='processing', error=None)
            await asyncio.sleep(hang)
            events.append('retrieve-finished')
            return SimpleNamespace(status='ready', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)

        started = time.monotonic()
        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        elapsed = time.monotonic() - started
        assert elapsed < hang
        assert elapsed < budget + 0.35
        assert calls == 2
        assert 'retrieve-finished' not in events
        assert 'create' not in events
        assert str(tmp_path) not in str(exc_info.value)
        assert 'timed out' not in str(exc_info.value).lower()
        assert is_retryable_error(exc_info.value)[0] is False
        assert events == ['upload', 'retrieve:asset-1', 'retrieve:asset-1', 'delete:asset-1']

    def test_retrieve_rate_limit_retries_until_ready(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        monkeypatch.setattr(twelvelabs, '_ASSET_POLL_INITIAL_SECS', 0.01)
        monkeypatch.setattr(twelvelabs, '_ASSET_RETRIEVE_RPS', 1_000_000.0)
        events: list[str] = []
        sleeps: list[float] = []
        calls = 0

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-1')

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            events.append(f'retrieve:{asset_id}')
            if calls == 1:
                raise _RetrieveRateLimitError('0.05')
            return SimpleNamespace(status='ready', error=None)

        def sleep(delay: float) -> None:
            sleeps.append(delay)

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)])

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        vector = asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert vector.shape == (512,)
        assert twelvelabs._asset_polls.waiting == 0
        assert sleeps == pytest.approx([0.05])
        assert events == ['retrieve:asset-1', 'retrieve:asset-1', 'create', 'delete:asset-1']

    def test_embed_create_retry_reuses_ready_asset(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        # One invocation creates once and raises the retryable error. RequestRateScheduler
        # paces the next attempt. That attempt reuses the ready asset instead of uploading.
        media = large_media_client.media
        uploads = 0
        creates = 0
        asset_ids: list[str] = []
        sleeps: list[float] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            nonlocal uploads
            uploads += 1
            return SimpleNamespace(asset_id='asset-ready')

        def retrieve(asset_id: str) -> SimpleNamespace:
            return SimpleNamespace(status='ready', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            nonlocal creates
            creates += 1
            asset_ids.append(kwargs['video'].media_source.asset_id)
            if creates == 1:
                raise _RetrieveRateLimitError('0.05')
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)])

        def sleep(delay: float) -> None:
            sleeps.append(delay)

        def delete(asset_id: str) -> None:
            assert asset_id == 'asset-ready'

        client = large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pytest.raises(_RetrieveRateLimitError, match='too many requests'):
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert uploads == 1
        assert creates == 1
        assert asset_ids == ['asset-ready']
        assert sleeps == []
        assert client.assets.delete.await_count == 0

        vector = asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert vector.shape == (512,)
        assert uploads == 1
        assert creates == 2
        assert asset_ids == ['asset-ready', 'asset-ready']
        assert sleeps == []
        assert client.assets.delete.await_count == 1

    def test_embed_create_retry_stops_when_bounded(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.exec.expr_eval.schedulers import RequestRateScheduler
        from pixeltable.functions import twelvelabs

        # Each call creates once. The asset is held for RequestRateScheduler.MAX_RETRIES
        # retries, then deleted so a later call uploads again instead of leaking it.
        media = large_media_client.media
        uploads = 0
        creates = 0
        sleeps: list[float] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            nonlocal uploads
            uploads += 1
            return SimpleNamespace(asset_id='asset-ready')

        def retrieve(asset_id: str) -> SimpleNamespace:
            return SimpleNamespace(status='ready', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            nonlocal creates
            creates += 1
            assert kwargs['video'].media_source.asset_id == 'asset-ready'
            raise _RetrieveRateLimitError('0.01')

        def sleep(delay: float) -> None:
            sleeps.append(delay)

        client = large_media_client.install(
            upload_file=upload_file, retrieve=retrieve, create=create, delete=lambda asset_id: None
        )
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        attempts = RequestRateScheduler.MAX_RETRIES + 1
        for attempt in range(attempts):
            with pytest.raises(_RetrieveRateLimitError, match='too many requests'):
                asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
            assert creates == attempt + 1
            assert uploads == 1
        assert sleeps == []
        assert client.assets.delete.await_count == 1

        with pytest.raises(_RetrieveRateLimitError, match='too many requests'):
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert uploads == 2
        assert creates == attempts + 1

    def test_embed_create_nonretryable_does_not_reuse_asset(self, large_media_client: _LargeMediaClient) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        uploads = 0

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            nonlocal uploads
            uploads += 1
            return SimpleNamespace(asset_id=f'asset-{uploads}')

        def retrieve(asset_id: str) -> SimpleNamespace:
            return SimpleNamespace(status='ready', error=None)

        def create(**kwargs: Any) -> SimpleNamespace:
            raise RuntimeError('rejected')

        client = large_media_client.install(
            upload_file=upload_file, retrieve=retrieve, create=create, delete=lambda asset_id: None
        )

        for expected_uploads in (1, 2):
            with pytest.raises(RuntimeError, match='rejected'):
                asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
            assert uploads == expected_uploads
        assert client.assets.delete.await_count == 2

    def test_retrieve_rate_limit_past_deadline_is_not_retried(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        events: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-1')

        def retrieve(asset_id: str) -> SimpleNamespace:
            events.append(f'retrieve:{asset_id}')
            raise _RetrieveRateLimitError('10000')

        def sleep(delay: float) -> None:
            raise AssertionError(f'waited {delay}s after a retry delay past the deadline')

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing') as exc_info:
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        message = str(exc_info.value).lower()
        assert 'timed out' not in message
        assert '429' not in message
        assert 'too many requests' not in message
        assert is_retryable_error(exc_info.value)[0] is False
        assert events == ['retrieve:asset-1', 'delete:asset-1']

    def test_retrieve_poll_gap_tracks_inflight_assets(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        # 40 other assets are already waiting. This call adds one more, past the 5s backoff ceiling.
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 40)
        events: list[str] = []
        sleeps: list[float] = []
        statuses = ['processing', 'ready']

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-1')

        def retrieve(asset_id: str) -> SimpleNamespace:
            status = statuses.pop(0)
            events.append(f'retrieve:{status}')
            return SimpleNamespace(status=status, error=None)

        def sleep(delay: float) -> None:
            sleeps.append(delay)

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 512)])

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        vector = asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert vector.shape == (512,)
        assert sleeps == pytest.approx([41 / twelvelabs._ASSET_RETRIEVE_RPS])
        assert sleeps[0] > twelvelabs._ASSET_POLL_MAX_SECS
        assert twelvelabs._asset_polls.waiting == 40
        assert events == ['retrieve:processing', 'retrieve:ready', 'create', 'delete:asset-1']

    def test_retrieve_poll_gap_drops_ready_assets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # 100 assets, 99 ready on the first poll, and 4 other polls already in flight.
        # The next gap uses those 5 assets (1s). Counting the original 104 would sleep 20.8s.
        asset_ids = [f'asset-{i}' for i in range(100)]
        still_processing = asset_ids[-1]
        others_waiting = 4
        seen: set[str] = set()
        retrieved: list[str] = []
        sleeps: list[float] = []
        waiting_during_sleep: list[int] = []
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', others_waiting)

        def retrieve(asset_id: str) -> SimpleNamespace:
            first_poll = asset_id not in seen
            seen.add(asset_id)
            retrieved.append(asset_id)
            status = 'processing' if asset_id == still_processing and first_poll else 'ready'
            return SimpleNamespace(status=status, error=None)

        def sleep(delay: float) -> None:
            sleeps.append(delay)
            waiting_during_sleep.append(twelvelabs._asset_polls.waiting)

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        asyncio.run(twelvelabs._wait_for_assets(client, asset_ids))

        still_waiting = others_waiting + 1
        expected_pause = max(twelvelabs._ASSET_POLL_INITIAL_SECS, still_waiting / twelvelabs._ASSET_RETRIEVE_RPS)
        assert waiting_during_sleep == [still_waiting]
        assert sleeps == pytest.approx([expected_pause])
        assert retrieved.count(still_processing) == 2
        assert len(retrieved) == len(asset_ids) + 1
        assert twelvelabs._asset_polls.waiting == others_waiting

    def test_retrieve_poll_count_restored_when_remaining_asset_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # One asset is ready on the first poll and the other fails on the next. finally must drop only
        # the asset still outstanding, and the gap before that failure uses the reduced count.
        others_waiting = 4
        phase = 0
        sleeps: list[float] = []
        waiting_during_sleep: list[int] = []
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', others_waiting)

        def retrieve(asset_id: str) -> SimpleNamespace:
            if phase == 0:
                status = 'processing' if asset_id == 'pending-429' else 'ready'
                return SimpleNamespace(status=status, error=None)
            return SimpleNamespace(status='failed', error=SimpleNamespace(message='processing timed out'))

        def sleep(delay: float) -> None:
            nonlocal phase
            sleeps.append(delay)
            waiting_during_sleep.append(twelvelabs._asset_polls.waiting)
            phase = 1

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='TwelveLabs asset failed') as exc_info:
            asyncio.run(twelvelabs._wait_for_assets(client, ['ready-asset', 'pending-429']))
        message = str(exc_info.value)
        assert 'pending-429' not in message
        assert '429' not in message
        assert 'timed out' not in message.lower()
        assert is_retryable_error(exc_info.value)[0] is False
        cause = exc_info.value.__cause__
        assert isinstance(cause, twelvelabs._FailedAssetError)
        assert cause.asset_id == 'pending-429'
        assert cause.reason == 'processing timed out'
        still_waiting = others_waiting + 1
        expected_pause = max(twelvelabs._ASSET_POLL_INITIAL_SECS, still_waiting / twelvelabs._ASSET_RETRIEVE_RPS)
        assert waiting_during_sleep == [still_waiting]
        assert sleeps == pytest.approx([expected_pause])
        assert twelvelabs._asset_polls.waiting == others_waiting

    def test_retrieve_error_deletes_asset(
        self, large_media_client: _LargeMediaClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixeltable.functions import twelvelabs

        media = large_media_client.media
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', 0.2)
        monkeypatch.setattr(twelvelabs, '_ASSET_POLL_INITIAL_SECS', 0.01)
        monkeypatch.setattr(twelvelabs, '_ASSET_RETRIEVE_RPS', 1_000_000.0)
        events: list[str] = []

        def upload_file(file_path: str, file_type: str) -> SimpleNamespace:
            return SimpleNamespace(asset_id='asset-1')

        def retrieve(asset_id: str) -> SimpleNamespace:
            events.append(f'retrieve:{asset_id}')
            raise _RetrieveRejectedError()

        def create(**kwargs: Any) -> SimpleNamespace:
            events.append('create')
            raise AssertionError('embed should not run before the asset is ready')

        def delete(asset_id: str) -> None:
            events.append(f'delete:{asset_id}')

        large_media_client.install(upload_file=upload_file, retrieve=retrieve, create=create, delete=delete)

        with pytest.raises(_RetrieveRejectedError, match='asset missing'):
            asyncio.run(twelvelabs.embed.py_fns[3](str(media), model_name='marengo3.0'))
        assert events == ['retrieve:asset-1', 'delete:asset-1']

    def test_retrieve_not_found_with_retry_text_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # is_retryable_error() matches the "429" substring even though the status is 404.
        # The status is authoritative: a missing asset is not polled until the deadline.
        error = _RetrieveRejectedError('asset ee429ff was not found')
        assert '429' in str(error)
        assert is_retryable_error(error)[0] is True
        assert twelvelabs._retrieve_should_retry(error) == (False, None)
        calls = 0
        clock = {'t': 0.0}
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', 1.0)
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            raise error

        def sleep(delay: float) -> None:
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pytest.raises(_RetrieveRejectedError, match='ee429ff'):
            asyncio.run(twelvelabs._wait_for_assets(client, ['ee429ff']))
        assert calls == 1
        assert clock['t'] == 0
        assert twelvelabs._asset_polls.waiting == 0

    def test_deadline_poll_raises_when_still_processing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # The pace for this many waits is longer than the budget. The sleep stops early for a
        # bounded retrieve. A pending result is polled once more in the trailing second, then raised.
        others_waiting = 5_000
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', others_waiting)
        clock = {'t': 0.0}
        sleeps: list[float] = []
        retrieves: list[str] = []

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            retrieves.append(asset_id)
            return SimpleNamespace(status='processing', error=None)

        def sleep(delay: float) -> None:
            sleeps.append(delay)
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing after 600s') as exc_info:
            asyncio.run(twelvelabs._wait_for_assets(client, ['ee429ff']))
        message = str(exc_info.value)
        assert 'ee429ff' not in message
        assert '429' not in message
        assert 'timed out' not in message.lower()
        assert is_retryable_error(exc_info.value)[0] is False
        assert retrieves == ['ee429ff', 'ee429ff', 'ee429ff']
        budget = twelvelabs._ASSET_READY_TIMEOUT_SECS
        early = budget - twelvelabs._ASSET_FINAL_POLL_SECS
        followup = twelvelabs._ASSET_FINAL_POLL_SECS - twelvelabs._ASSET_FOLLOWUP_POLL_SECS
        assert sleeps == pytest.approx([early, followup])
        assert clock['t'] == pytest.approx(budget - twelvelabs._ASSET_FOLLOWUP_POLL_SECS)
        assert twelvelabs._asset_polls.waiting == others_waiting

    def test_ready_within_budget_polled_at_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # One asset among 2048 waits. The pace is ~410s, so the second sleep would reach the 600s
        # deadline. It stops early enough for a bounded retrieve. The asset is ready at 450s and
        # must be seen by that retrieve.
        others_waiting = 2048
        ready_after = 450.0
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', others_waiting)
        clock = {'t': 0.0}
        sleeps: list[float] = []
        retrieves = 0

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal retrieves
            retrieves += 1
            status = 'ready' if clock['t'] >= ready_after else 'processing'
            return SimpleNamespace(status=status, error=None)

        def sleep(delay: float) -> None:
            sleeps.append(delay)
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        asyncio.run(twelvelabs._wait_for_assets(client, ['asset-429']))
        pace = (others_waiting + 1) / twelvelabs._ASSET_RETRIEVE_RPS
        budget = twelvelabs._ASSET_READY_TIMEOUT_SECS
        reserved = twelvelabs._ASSET_FINAL_POLL_SECS
        assert pace < budget - reserved
        assert sleeps == pytest.approx([pace, budget - pace - reserved])
        assert retrieves == 3
        assert clock['t'] == pytest.approx(budget - reserved)
        assert clock['t'] > ready_after
        assert twelvelabs._asset_polls.waiting == others_waiting

    def test_ready_in_final_window_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # Backoff reaches the early final retrieve at 595s of 600. The asset is still
        # processing there and ready at 597s. That retrieve must not be the deadline.
        budget = twelvelabs._ASSET_READY_TIMEOUT_SECS
        early = budget - twelvelabs._ASSET_FINAL_POLL_SECS
        ready_at = early + 2.0
        followup = budget - twelvelabs._ASSET_FOLLOWUP_POLL_SECS
        assert ready_at == pytest.approx(597.0)
        assert early < ready_at < followup < budget
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)
        clock = {'t': 0.0}
        polled_at: list[float] = []

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            polled_at.append(clock['t'])
            status = 'ready' if clock['t'] >= ready_at else 'processing'
            return SimpleNamespace(status=status, error=None)

        def sleep(delay: float) -> None:
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        asyncio.run(twelvelabs._wait_for_assets(client, ['asset-1']))
        assert early in polled_at
        assert polled_at[-1] == pytest.approx(followup)
        assert polled_at[-1] >= ready_at
        assert clock['t'] == pytest.approx(followup)
        assert clock['t'] <= budget
        assert twelvelabs._asset_polls.waiting == 0

    def test_final_window_waits_out_retry_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # The early final retrieve returns 503 with Retry-After 4.5s. The asset is ready
        # after that delay and before the deadline. Polling on the 1s trailing budget
        # (sooner than Retry-After) would still see the error and raise.
        budget = twelvelabs._ASSET_READY_TIMEOUT_SECS
        early = budget - twelvelabs._ASSET_FINAL_POLL_SECS
        retry_after = 4.5
        ready_at = early + retry_after
        assert ready_at < budget
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)
        clock = {'t': 0.0}
        polled_at: list[float] = []

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            polled_at.append(clock['t'])
            if clock['t'] >= ready_at:
                return SimpleNamespace(status='ready', error=None)
            if clock['t'] >= early:
                raise _RetrieveRateLimitError(str(retry_after))
            return SimpleNamespace(status='processing', error=None)

        def sleep(delay: float) -> None:
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        asyncio.run(twelvelabs._wait_for_assets(client, ['asset-1']))
        assert early in polled_at
        assert polled_at[-1] >= ready_at
        assert polled_at[-1] == pytest.approx(early + retry_after)
        assert clock['t'] < budget
        assert twelvelabs._asset_polls.waiting == 0

    def test_sustained_waits_ready_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # 2048 single-asset waits arrive at the request-rate scheduler's default 10/s. Each asset
        # is ready 450s after it starts, inside the 600s budget. At waiting=2048 the pace is ~410s;
        # sleeping out the rest of the budget without a retrieve used to drop 1569 of these assets.
        n_assets = 2048
        arrivals_per_sec = 10.0
        ready_after = 450.0
        budget = twelvelabs._ASSET_READY_TIMEOUT_SECS
        asset_ids = [f'asset-{i}' for i in range(n_assets)]
        clock = _VirtualClock(n_assets)
        started: dict[str, float] = {}
        finished: dict[str, float] = {}
        peak = 0
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal peak
            peak = max(peak, twelvelabs._asset_polls.waiting)
            status = 'ready' if clock.now >= started[asset_id] + ready_after else 'processing'
            return SimpleNamespace(status=status, error=None)

        async def run_one(i: int) -> None:
            asset_id = asset_ids[i]
            try:
                lag = i / arrivals_per_sec - clock.now
                if lag > 0:
                    await clock.sleep(lag)
                started[asset_id] = clock.now
                await twelvelabs._wait_for_assets(client, [asset_id])
                finished[asset_id] = clock.now
            finally:
                clock.task_finished()

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', clock.monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', clock.sleep)

        async def run() -> list[BaseException | None]:
            tasks = [asyncio.create_task(run_one(i)) for i in range(n_assets)]
            return list(await asyncio.gather(*tasks, return_exceptions=True))

        # No asyncio.wait_for here: the virtual clock replaces time.monotonic, which is also the
        # event-loop clock, so a real timeout would expire as soon as simulated time passes it.
        outcomes = asyncio.run(run())
        failures = [outcome for outcome in outcomes if outcome is not None]
        assert failures == []
        assert peak == n_assets
        assert len(finished) == n_assets
        for asset_id, start in started.items():
            assert finished[asset_id] >= start + ready_after - 1e-6
            assert finished[asset_id] <= start + budget + 1e-6
        assert twelvelabs._asset_polls.waiting == 0

    @pytest.mark.parametrize('status_code', [408, 503, 504])
    def test_retryable_retrieve_status_polls_again(self, monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        calls = 0
        error = _RetrieveStatusError(status_code)
        assert is_retryable_error(error)[0] is True

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return SimpleNamespace(status='ready', error=None)

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock())
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)

        asyncio.run(twelvelabs._wait_for_assets(client, ['asset-1']))
        assert calls == 2
        assert twelvelabs._asset_polls.waiting == 0

    def test_retryable_retrieve_past_deadline_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('twelvelabs')
        from pixeltable.functions import twelvelabs

        # A 503 is retryable, so the scheduler would upload again if this helper re-raised it.
        # Keep polling through the early final retrieve. The trailing second is that retrieve's
        # budget, and a pending result there is a non-retryable readiness error.
        budget = 12.0
        monkeypatch.setattr(twelvelabs, '_ASSET_READY_TIMEOUT_SECS', budget)
        monkeypatch.setattr(twelvelabs, '_ASSET_POLL_INITIAL_SECS', 1.0)
        monkeypatch.setattr(twelvelabs, '_ASSET_POLL_MAX_SECS', 5.0)
        monkeypatch.setattr(twelvelabs, '_ASSET_RETRIEVE_RPS', 1_000_000.0)
        monkeypatch.setattr(twelvelabs._asset_polls, 'waiting', 0)
        clock = {'t': 0.0}
        retrieves = 0

        def monotonic() -> float:
            return clock['t']

        def retrieve(asset_id: str) -> SimpleNamespace:
            nonlocal retrieves
            retrieves += 1
            raise _RetrieveStatusError(503)

        def sleep(delay: float) -> None:
            clock['t'] += delay

        client: Any = SimpleNamespace(assets=SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve)))
        monkeypatch.setattr(twelvelabs.time, 'monotonic', monotonic)
        monkeypatch.setattr(twelvelabs.asyncio, 'sleep', AsyncMock(side_effect=sleep))

        with pxt_raises(pxt.ErrorCode.PROVIDER_TIMEOUT, match='still processing') as exc_info:
            asyncio.run(twelvelabs._wait_for_assets(client, ['asset-429']))
        assert 'asset-429' not in str(exc_info.value)
        assert '503' not in str(exc_info.value)
        assert is_retryable_error(exc_info.value)[0] is False
        assert retrieves == 5
        assert clock['t'] == pytest.approx(budget - twelvelabs._ASSET_FOLLOWUP_POLL_SECS)
        assert twelvelabs._asset_polls.waiting == 0

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
