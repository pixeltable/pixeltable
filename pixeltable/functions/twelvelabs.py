"""
Pixeltable UDFs
that wrap various endpoints from the TwelveLabs API. In order to use them, you must
first `pip install twelvelabs` and configure your TwelveLabs credentials, as described in
the [Working with TwelveLabs](https://docs.pixeltable.com/howto/providers/working-with-twelvelabs) tutorial.
"""

import asyncio
import logging
import os
import time
from base64 import b64encode
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Literal, NoReturn, Sequence

import numpy as np

import pixeltable as pxt
from pixeltable import env, type_system as ts
from pixeltable.runtime import get_runtime
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import exponential_backoff, is_retryable_error
from pixeltable.utils.image import to_base64

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import twelvelabs
    from twelvelabs.wrapper.multipart_upload_client_wrapper import UploadResult


TWELVELABS_INLINE_LIMIT_BYTES = 2 * 2**20

# Direct uploads stay in `processing` until validation finishes. Large media can take minutes.
_ASSET_POLL_INITIAL_SECS = 1.0
_ASSET_POLL_MAX_SECS = 5.0
_ASSET_READY_TIMEOUT_SECS = 600.0
# RequestRateScheduler meters embed() calls, not these polls. 5/s is half of the default
# TwelveLabs budget (600 RPM), leaving room for uploads and embed calls.
_ASSET_RETRIEVE_RPS = 5.0
# A paced sleep stops this far before the readiness deadline. The retrieve after that sleep
# uses the leftover time as its wait_for timeout, so the last poll is cancelled at the deadline
# instead of running with no timeout. An asset that is ready in that window is still accepted.
_ASSET_FINAL_POLL_SECS = 5.0
# HTTP statuses is_retryable_error() treats as retryable. Any other explicit status is final.
_RETRYABLE_RETRIEVE_STATUSES = frozenset({408, 429, 503, 504})
_RETRYABLE_RETRIEVE_STATUS_NAMES = {
    'REQUEST_TIMEOUT': 408,
    'TOO_MANY_REQUESTS': 429,
    'SERVICE_UNAVAILABLE': 503,
    'GATEWAY_TIMEOUT': 504,
    'NOT_FOUND': 404,
}
# RequestRateScheduler.MAX_RETRIES is 3. Retrying create while the ready asset is still held
# avoids deleting it and uploading the file again for those attempts.
_ASSET_CREATE_RETRIES = 3
_ASSET_CREATE_MAX_BACKOFF_SECS = 60.0


class _AssetPolls:
    """Process-wide count of assets _wait_for_assets() is still polling."""

    def __init__(self) -> None:
        self.waiting = 0


_asset_polls = _AssetPolls()


@env.register_client('twelvelabs', credential_param='api_key')
def _(api_key: str) -> 'twelvelabs.AsyncTwelveLabs':
    import twelvelabs

    return twelvelabs.AsyncTwelveLabs(api_key=api_key)


def _twelvelabs_client() -> 'twelvelabs.AsyncTwelveLabs':
    return get_runtime().get_client('twelvelabs')


def _coerce_http_status(value: Any) -> int | None:
    """Return an HTTP status code, or None when `value` is not one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if 100 <= value <= 599 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            code = int(text)
            return code if 100 <= code <= 599 else None
        return _RETRYABLE_RETRIEVE_STATUS_NAMES.get(text.upper())
    return None


def _http_status(exc: Exception) -> int | None:
    """HTTP status from an exception or its response, when the provider set one."""
    objects: list[Any] = [exc]
    response = getattr(exc, 'response', None)
    if response is not None and response is not exc:
        objects.append(response)
    for obj in objects:
        for attr in ('status_code', 'status', 'code'):
            status = _coerce_http_status(getattr(obj, attr, None))
            if status is not None:
                return status
    return None


def _assets_not_ready(
    pending: Sequence[tuple[str, Any]], *, blocked_by_provider_delay: bool = False
) -> pxt.ExternalServiceError:
    """Deadline failure for assets that are still not `ready`.

    Asset ids stay out of the message. `is_retryable_error()` treats substrings such as "429" and
    "timed out" as transport failures, and RequestRateScheduler would upload the file again after
    this asset was deleted.
    """
    assert len(pending) > 0
    budget = f'{_ASSET_READY_TIMEOUT_SECS:g}s'
    if blocked_by_provider_delay:
        summary = f'TwelveLabs assets still processing; provider delay exceeds the {budget} readiness deadline'
    else:
        summary = f'TwelveLabs assets still processing after {budget}'
    return pxt.ExternalServiceError(pxt.ErrorCode.PROVIDER_TIMEOUT, summary, provider='twelvelabs')


class _FailedAssetError(Exception):
    """Asset id and provider reason for a failed upload.

    Raised only as the `__cause__` of the fixed-wording `ExternalServiceError`. `is_retryable_error()`
    reads `str()` of the exception it is given and does not walk `__cause__`, so this text cannot make
    RequestRateScheduler upload the file again.
    """

    def __init__(self, asset_id: str, reason: str | None) -> None:
        self.asset_id = asset_id
        self.reason = reason
        super().__init__(asset_id if reason is None else f'{asset_id}: {reason}')


def _failure_reason(result: Any) -> str | None:
    error = getattr(result, 'error', None)
    if error is None:
        return None
    message = getattr(error, 'message', None)
    if message:
        return str(message)
    text = str(error)
    return text or None


def _raise_asset_failed(asset_id: str, result: Any) -> NoReturn:
    reason = _failure_reason(result)
    # The scheduler logs this exception chain. The warning is there when that chain is not printed.
    _logger.warning('TwelveLabs asset %s failed: %s', asset_id, reason if reason is not None else 'unknown')
    err = pxt.ExternalServiceError(pxt.ErrorCode.PROVIDER_ERROR, 'TwelveLabs asset failed', provider='twelvelabs')
    raise err from _FailedAssetError(asset_id, reason)


def _retrieve_should_retry(exc: BaseException) -> tuple[bool, float | None]:
    """Whether another poll can still succeed, plus the provider retry delay when one was given.

    An explicit HTTP status decides. `is_retryable_error()` still matches substrings such as "429"
    when the status is a non-retryable 4xx, which would poll a missing asset until the deadline.
    Message text is used only when the provider did not set a status.
    """
    if not isinstance(exc, Exception):
        return False, None
    status = _http_status(exc)
    if status is not None:
        if status not in _RETRYABLE_RETRIEVE_STATUSES:
            return False, None
        return True, is_retryable_error(exc)[1]
    return is_retryable_error(exc)


def _classify_retrieve_results(
    asset_ids: Sequence[str], results: Sequence[Any]
) -> tuple[list[tuple[str, Any]], float | None]:
    """Split a retrieve round into assets that still need another poll, and the longest retry delay."""
    still_pending: list[tuple[str, Any]] = []
    rate_limit_delay: float | None = None
    for asset_id, result in zip(asset_ids, results):
        if isinstance(result, BaseException):
            retryable, retry_delay = _retrieve_should_retry(result)
            if retryable:
                if retry_delay is not None and (rate_limit_delay is None or retry_delay > rate_limit_delay):
                    rate_limit_delay = retry_delay
                still_pending.append((asset_id, None))
                continue
            raise result
        status = result.status
        if status == 'ready':
            continue
        if status == 'failed':
            _raise_asset_failed(asset_id, result)
        still_pending.append((asset_id, status))
    return still_pending, rate_limit_delay


async def _retrieve_within(client: 'twelvelabs.AsyncTwelveLabs', asset_ids: Sequence[str], timeout: float) -> list[Any]:
    """Retrieve `asset_ids`, cancelling the round if it is still running after `timeout` seconds."""
    if timeout <= 0:
        raise _assets_not_ready([(asset_id, None) for asset_id in asset_ids])
    retrieve_round = asyncio.gather(
        *(client.assets.retrieve(asset_id) for asset_id in asset_ids), return_exceptions=True
    )
    try:
        return await asyncio.wait_for(retrieve_round, timeout=timeout)
    except TimeoutError:
        raise _assets_not_ready([(asset_id, None) for asset_id in asset_ids]) from None


async def _poll_pending(
    client: 'twelvelabs.AsyncTwelveLabs', asset_ids: Sequence[str], deadline: float
) -> tuple[list[tuple[str, Any]], float | None]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _assets_not_ready([(asset_id, None) for asset_id in asset_ids])
    results = await _retrieve_within(client, asset_ids, remaining)
    return _classify_retrieve_results(asset_ids, results)


def _release_ready(n_waiting: int, polled: int, still_pending: Sequence[tuple[str, Any]]) -> int:
    """Drop assets that left this poll from the shared pace count. Returns the count still held."""
    released = polled - len(still_pending)
    _asset_polls.waiting -= released
    return n_waiting - released


async def _wait_for_assets(client: 'twelvelabs.AsyncTwelveLabs', asset_ids: list[str]) -> None:
    """Poll `assets.retrieve` until every asset is `ready`.

    Raises when an asset fails or is still not ready at `_ASSET_READY_TIMEOUT_SECS`. Every retrieve,
    including the last one, is bounded by the time left in that budget. A paced sleep that would run
    up to the deadline stops `_ASSET_FINAL_POLL_SECS` early and is followed by one retrieve in the
    leftover time. The timeout is raised only when that status is still not ready. HTTP 429 and other
    retryable retrieve errors are polled again until that deadline instead of failing the upload.
    """
    pending = list(asset_ids)
    deadline = time.monotonic() + _ASSET_READY_TIMEOUT_SECS
    delay = _ASSET_POLL_INITIAL_SECS
    n_waiting = len(pending)
    _asset_polls.waiting += n_waiting
    try:
        while pending:
            still_pending, rate_limit_delay = await _poll_pending(client, pending, deadline)
            n_waiting = _release_ready(n_waiting, len(pending), still_pending)
            if not still_pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _assets_not_ready(still_pending)
            # A Retry-After past the deadline cannot succeed in time. Fail now so the request-rate
            # pool is not held until the deadline, and do not let the 429 become a scheduler retry.
            if rate_limit_delay is not None and rate_limit_delay >= remaining:
                raise _assets_not_ready(still_pending, blocked_by_provider_delay=True)
            pause = max(delay, _asset_polls.waiting / _ASSET_RETRIEVE_RPS)
            if rate_limit_delay is not None:
                pause = max(pause, rate_limit_delay)
            # Leave a positive timeout for one last retrieve. Sleeping until the deadline would make
            # that wait_for timeout 0, and awaiting the retrieve without one holds the upload past it.
            reserved = min(_ASSET_FINAL_POLL_SECS, remaining)
            if pause < remaining - reserved:
                await asyncio.sleep(pause)
                delay = min(delay * 2, _ASSET_POLL_MAX_SECS)
                pending = [asset_id for asset_id, _status in still_pending]
                continue
            sleep_for = remaining - reserved
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            final_ids = [asset_id for asset_id, _status in still_pending]
            final_pending, _rate_limit_delay = await _poll_pending(client, final_ids, deadline)
            n_waiting = _release_ready(n_waiting, len(final_ids), final_pending)
            if final_pending:
                raise _assets_not_ready(final_pending)
            return
    finally:
        _asset_polls.waiting -= n_waiting


@asynccontextmanager
async def _asset_uploads(input_type: Literal['audio', 'video'], files: list[str]) -> AsyncIterator[list[str]]:
    """
    Context manager that uploads files, waits until each TwelveLabs asset is ready, and deletes the assets after use.

    Returns:
        A list of asset IDs corresponding to the uploaded files.
    """
    if len(files) == 0:
        yield []
        return

    client = _twelvelabs_client()
    uploaded: list[str] = []

    try:
        tasks: list[Coroutine[Any, Any, 'UploadResult']] = []
        for file in files:
            tasks.append(client.multipart_upload.upload_file(file_path=file, file_type=input_type))  # type: ignore[attr-defined]
        upload_results = await asyncio.gather(*tasks)
        uploaded = [u.asset_id for u in upload_results]
        await _wait_for_assets(client, uploaded)

        yield uploaded

    finally:
        await asyncio.gather(*[client.assets.delete(asset_id) for asset_id in uploaded], return_exceptions=True)


@pxt.udf(resource_pool='request-rate:twelvelabs')
async def embed(text: str, image: pxt.Image | None = None, *, model_name: str) -> pxt.Array[np.float32] | None:
    """
    Creates an embedding vector for the given text, audio, image, or video input.

    Each UDF signature corresponds to one of the four supported input types. If text is specified, it is possible to
    specify an image as well. Both `marengo3.0` and `marengo3.5` return 512-dimensional vectors that can be used
    with an embedding index. Embeddings from different model versions are incompatible; re-embed existing
    content when changing models.

    Marengo 3.5 uses the synchronous `multi_input` API. Images, audio, and video must be at most 32 MB.
    Audio and video must also be at most 30 seconds. Split longer files with `audio_splitter` or
    `video_splitter` before embedding them. The `start_sec`,
    `end_sec`, and `embedding_option` parameters are supported only with Marengo 3.0.

    Equivalent to the TwelveLabs Embed API:
    <https://docs.twelvelabs.io/sdk-reference/python/create-embeddings-v-2/create-sync-embeddings>

    Request throttling:
    Applies the rate limit set in the config (section `twelvelabs`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install twelvelabs`
    - Marengo 3.5 requires `twelvelabs>=1.3.6`.

    Args:
        model_name: The name of the model to use (`marengo3.0` or `marengo3.5`).
        text: The text to embed.
        image: If specified, the embedding will be created from both the text and the image.

    Returns:
        The embedding.

    Examples:
        Add a computed column `embed` for an embedding of a string column `input`:

        >>> tbl.add_computed_column(
        ...     embed=embed(model_name='marengo3.5', text=tbl.input)
        ... )

        Add an embedding index for cross-modal search:

        >>> tbl.add_embedding_index(
        ...     tbl.input, embedding=embed.using(model_name='marengo3.5')
        ... )
    """
    env.Env.get().require_package('twelvelabs')
    import twelvelabs

    if model_name == 'marengo3.5':
        b64_str = None if image is None else to_base64(image, format=('png' if image.has_transparency_data else 'jpeg'))
        return await _embed_multi_input(text=text, input_type='image', b64_str=b64_str)

    cl = _twelvelabs_client()
    res: twelvelabs.EmbeddingSuccessResponse
    if image is None:
        # Text-only
        res = await cl.embed.v_2.create(
            input_type='text', model_name=model_name, text=twelvelabs.TextInputRequest(input_text=text)
        )
    else:
        b64str = to_base64(image, format=('png' if image.has_transparency_data else 'jpeg'))
        res = await cl.embed.v_2.create(
            input_type='text_image',
            model_name=model_name,
            text_image=twelvelabs.TextImageInputRequest(
                media_source=twelvelabs.MediaSource(base_64_string=b64str), input_text=text
            ),
        )
    if not res.data:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR, f"Didn't receive embedding for text: {text}\n{res}", provider='twelvelabs'
        )
    vector = res.data[0].embedding
    return np.array(vector, dtype='float32')


@embed.overload
async def _(image: pxt.Image, *, model_name: str) -> pxt.Array[np.float32] | None:
    env.Env.get().require_package('twelvelabs')
    import twelvelabs

    cl = _twelvelabs_client()
    b64_str = to_base64(image, format=('png' if image.has_transparency_data else 'jpeg'))
    if model_name == 'marengo3.5':
        return await _embed_multi_input(input_type='image', b64_str=b64_str)

    res = await cl.embed.v_2.create(
        input_type='image',
        model_name=model_name,
        image=twelvelabs.ImageInputRequest(media_source=twelvelabs.MediaSource(base_64_string=b64_str)),
    )
    if not res.data:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR, f"Didn't receive embedding for image: {image}\n{res}", provider='twelvelabs'
        )
    vector = res.data[0].embedding
    return np.array(vector, dtype='float32')


@embed.overload
async def _(
    audio: pxt.Audio,
    *,
    model_name: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    embedding_option: list[Literal['audio', 'transcription']] | None = None,
) -> pxt.Array[np.float32] | None:
    env.Env.get().require_package('twelvelabs')
    import twelvelabs

    if model_name == 'marengo3.5':
        return await _embed_av_multi_input(audio, 'audio', start_sec, end_sec, embedding_option)

    return await _embed_av_content(
        file_path=audio,
        input_type='audio',
        request_cls=twelvelabs.AudioInputRequest,
        model_name=model_name,
        start_sec=start_sec,
        end_sec=end_sec,
        embedding_option=embedding_option,
    )


@embed.overload
async def _(
    video: pxt.Video,
    *,
    model_name: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    embedding_option: list[Literal['visual', 'audio', 'transcription']] | None = None,
) -> pxt.Array[np.float32] | None:
    env.Env.get().require_package('twelvelabs')
    import twelvelabs

    if model_name == 'marengo3.5':
        return await _embed_av_multi_input(video, 'video', start_sec, end_sec, embedding_option)

    return await _embed_av_content(
        file_path=video,
        input_type='video',
        request_cls=twelvelabs.VideoInputRequest,
        model_name=model_name,
        start_sec=start_sec,
        end_sec=end_sec,
        embedding_option=embedding_option,
    )


async def _embed_multi_input(
    *, text: str | None = None, input_type: Literal['image', 'audio', 'video'], b64_str: str | None = None
) -> pxt.Array[np.float32]:
    env.Env.get().require_package('twelvelabs', min_version=[1, 3, 6])
    import twelvelabs

    media_sources = None
    if b64_str is not None:
        # The API's limit applies to the decoded media, not the base64 payload.
        size_bytes = len(b64_str) * 3 // 4 - (len(b64_str) - len(b64_str.rstrip('=')))
        if size_bytes > 32 * 2**20:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT, 'Marengo 3.5 synchronous embeddings require media of at most 32 MB.'
            )
        media_sources = [twelvelabs.MultiInputMediaSource(media_type=input_type, base_64_string=b64_str)]
    # Pass only fields that are set. An explicit None is serialized as JSON null, and the sync schema
    # marks input_text and media_sources optional, not nullable.
    request: dict[str, Any] = {}
    if text is not None:
        request['input_text'] = text
    if media_sources is not None:
        request['media_sources'] = media_sources
    res = await _twelvelabs_client().embed.v_2.create(
        input_type='multi_input',
        model_name='marengo3.5',
        embedding_dimension=512,
        multi_input=twelvelabs.MultiInputRequest(**request),
    )
    if not res.data or len(res.data) != 1 or len(res.data[0].embedding) != 512:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            'Expected a single 512-dimensional embedding from Marengo 3.5.',
            provider='twelvelabs',
        )
    return np.array(res.data[0].embedding, dtype='float32')


async def _embed_av_multi_input(
    file_path: str,
    input_type: Literal['audio', 'video'],
    start_sec: float | None,
    end_sec: float | None,
    embedding_option: Sequence[str] | None,
) -> pxt.Array[np.float32]:
    if start_sec is not None or end_sec is not None or embedding_option is not None:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            'Marengo 3.5 synchronous embeddings do not support start_sec, end_sec, or embedding_option. '
            'Split the media with audio_splitter or video_splitter before embedding it.',
        )
    b64_str = await asyncio.to_thread(_encode_short_media, file_path, input_type)
    return await _embed_multi_input(input_type=input_type, b64_str=b64_str)


async def _create_with_ready_asset(client: Any, create_kwargs: dict[str, Any]) -> Any:
    """Call `embed.v_2.create` for an asset that is already ready.

    A retryable create failure used to leave this context, which deletes the asset. The scheduler
    then uploads the file again and waits for readiness on every retry. The retries stay here, on
    the same asset, until one succeeds or the bound is used up.
    """
    for attempt in range(_ASSET_CREATE_RETRIES + 1):
        try:
            return await client.embed.v_2.create(**create_kwargs)
        except Exception as exc:
            retryable, retry_delay = is_retryable_error(exc)
            if not retryable or attempt >= _ASSET_CREATE_RETRIES:
                raise
            delay = exponential_backoff(attempt) if retry_delay is None or retry_delay < 0 else retry_delay
            await asyncio.sleep(min(delay, _ASSET_CREATE_MAX_BACKOFF_SECS))
    raise AssertionError('embed create retry loop exited without a result')


def _encode_short_media(file_path: str, input_type: Literal['audio', 'video']) -> str:
    if os.stat(file_path).st_size > 32 * 2**20:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            'Marengo 3.5 synchronous embeddings require media of at most 32 MB. '
            'Split the media with audio_splitter or video_splitter before embedding it.',
        )
    duration = (
        av_utils.get_audio_duration(file_path) if input_type == 'audio' else av_utils.get_video_duration(file_path)
    )
    if duration is None:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, f'Cannot determine {input_type} duration.')
    if duration > 30:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'Marengo 3.5 synchronous embeddings require {input_type} of at most 30 seconds; '
            f'got {duration:.2f} seconds. '
            'Split the media with audio_splitter or video_splitter before embedding it.',
        )
    with open(file_path, 'rb') as fp:
        return b64encode(fp.read()).decode('utf-8')


async def _embed_av_content(
    file_path: str,
    input_type: Literal['audio', 'video'],
    request_cls: type['twelvelabs.AudioInputRequest'] | type['twelvelabs.VideoInputRequest'],
    model_name: str,
    start_sec: float | None,
    end_sec: float | None,
    embedding_option: Sequence[str] | None,
) -> pxt.Array[np.float32] | None:
    import twelvelabs

    cl = _twelvelabs_client()
    size_bytes = os.stat(file_path).st_size
    res: twelvelabs.EmbeddingSuccessResponse

    if size_bytes > TWELVELABS_INLINE_LIMIT_BYTES:
        async with _asset_uploads(input_type=input_type, files=[file_path]) as asset_ids:
            create_kwargs = {
                'input_type': input_type,
                'model_name': model_name,
                input_type: request_cls(
                    media_source=twelvelabs.MediaSource(asset_id=asset_ids[0]),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    embedding_option=embedding_option,
                ),
            }
            res = await _create_with_ready_asset(cl, create_kwargs)
    else:
        with open(file_path, 'rb') as fp:
            b64_str = b64encode(fp.read()).decode('utf-8')
        create_kwargs = {
            'input_type': input_type,
            'model_name': model_name,
            input_type: request_cls(
                media_source=twelvelabs.MediaSource(base_64_string=b64_str),
                start_sec=start_sec,
                end_sec=end_sec,
                embedding_option=embedding_option,
            ),
        }
        res = await cl.embed.v_2.create(**create_kwargs)  # type: ignore[arg-type]

    if not res.data:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            f"Didn't receive embedding for {input_type}: {file_path}\n{res}",
            provider='twelvelabs',
        )
    vector = res.data[0].embedding
    return np.array(vector, dtype='float32')


@embed.conditional_return_type
def _(model_name: str) -> ts.ArrayType:
    if model_name == 'Marengo-retrieval-2.7':
        return ts.ArrayType(shape=(1024,), dtype=np.dtype('float32'))
    if model_name in ('marengo3.0', 'marengo3.5'):
        return ts.ArrayType(shape=(512,), dtype=np.dtype('float32'))
    return ts.ArrayType(dtype=np.dtype('float32'))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
