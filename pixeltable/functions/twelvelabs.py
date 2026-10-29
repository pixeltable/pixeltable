"""
Pixeltable UDFs
that wrap various endpoints from the TwelveLabs API. In order to use them, you must
first `pip install twelvelabs` and configure your TwelveLabs credentials, as described in
the [Working with TwelveLabs](https://docs.pixeltable.com/howto/providers/working-with-twelvelabs) tutorial.
"""

import asyncio
import os
import time
from base64 import b64encode
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Literal, Sequence, TypeVar

import numpy as np

import pixeltable as pxt
from pixeltable import env, type_system as ts
from pixeltable.runtime import get_runtime
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import is_retryable_error
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import twelvelabs


TWELVELABS_INLINE_LIMIT_BYTES = 2 * 2**20

# Marengo 3.5 limits on decoded media, in decimal MB (the stricter reading of TwelveLabs' "MB")
_SYNC_MAX_BYTES = 32 * 10**6
_SYNC_MAX_SECS = 30.0
_TASK_INLINE_MAX_BYTES = 36 * 10**6

_POLL_INTERVAL_SECS = 5.0
_POLL_TIMEOUT_SECS = 600.0

_Polled = TypeVar('_Polled', 'twelvelabs.AssetDetail', 'twelvelabs.EmbeddingTaskResponse')


@env.register_client('twelvelabs', credential_param='api_key')
def _(api_key: str) -> 'twelvelabs.AsyncTwelveLabs':
    import twelvelabs

    class _ClosableClient(twelvelabs.AsyncTwelveLabs):
        """twelvelabs.AsyncTwelveLabs has no close() (its __aexit__() does nothing); Runtime.close_clients() needs
        one."""

        async def close(self) -> None:
            # the client creates its httpx client in its constructor
            await self._client_wrapper.httpx_client.httpx_client.aclose()

    return _ClosableClient(api_key=api_key)


def _twelvelabs_client() -> 'twelvelabs.AsyncTwelveLabs':
    return get_runtime().get_client('twelvelabs')


async def _wait_until_ready(retrieve: Callable[[], Awaitable[_Polled]], name: str) -> _Polled:
    deadline = time.monotonic() + _POLL_TIMEOUT_SECS
    while True:
        try:
            result = await retrieve()
        except Exception as exc:
            if not is_retryable_error(exc)[0]:
                raise
        else:
            if result.status == 'ready':
                return result
            if result.status == 'failed':
                reason = result.error.message if result.error is not None else 'unknown error'
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_BAD_REQUEST, f'{name} failed: {reason}', provider='twelvelabs'
                )
        if time.monotonic() > deadline:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_TIMEOUT,
                f'{name} was not ready after {_POLL_TIMEOUT_SECS:g} seconds.',
                provider='twelvelabs',
            )
        await asyncio.sleep(_POLL_INTERVAL_SECS)


@asynccontextmanager
async def _asset_upload(input_type: Literal['audio', 'video'], file: str) -> AsyncIterator[str]:
    """Uploads `file` and yields its asset id once TwelveLabs can use it. Deletes the asset on exit."""
    client = _twelvelabs_client()
    uploaded = await client.multipart_upload.upload_file(file_path=file, file_type=input_type)  # type: ignore[attr-defined]
    asset_id: str = uploaded.asset_id
    try:
        await _wait_until_ready(lambda: client.assets.retrieve(asset_id), f'TwelveLabs asset {asset_id}')
        yield asset_id
    finally:
        # a failed delete must not replace the embedding or the error
        with suppress(Exception):
            await client.assets.delete(asset_id)


@pxt.udf(resource_pool='request-rate:twelvelabs')
async def embed(text: str, image: pxt.Image | None = None, *, model_name: str) -> pxt.Array[np.float32] | None:
    """
    Creates an embedding vector for the given text, audio, image, or video input.

    Each UDF signature corresponds to one of the four supported input types. If text is specified, it is possible to
    specify an image as well. Both `marengo3.0` and `marengo3.5` return 512-dimensional vectors that can be used
    with an embedding index. Embeddings from different model versions are incompatible; re-embed existing
    content when changing models.

    With Marengo 3.5, text, images of at most 32 MB, and audio or video of at most 30 seconds and 32 MB use the
    synchronous `multi_input` API. Longer or larger audio and video use an asynchronous embedding task, which
    returns one vector for the whole file: the audio of an audio file, or the visuals of a video file without its
    audio. Files over 36 MB are uploaded to TwelveLabs for the task and deleted afterwards. TwelveLabs recommends
    whole-file vectors for clips of up to 10 to 30 seconds; to search within long media, split it with
    `audio_splitter` or `video_splitter` first. The `start_sec`, `end_sec`, and `embedding_option` parameters are
    supported only with Marengo 3.0.

    Equivalent to the TwelveLabs Embed API:
    <https://docs.twelvelabs.io/sdk-reference/python/create-embeddings-v-2>

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
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f"Didn't receive embedding for text: {text}\n{res}",
            provider='twelvelabs',
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
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f"Didn't receive embedding for image: {image}\n{res}",
            provider='twelvelabs',
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
        return await _embed_av_marengo35(audio, 'audio', start_sec, end_sec, embedding_option)

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
        return await _embed_av_marengo35(video, 'video', start_sec, end_sec, embedding_option)

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
        if size_bytes > _SYNC_MAX_BYTES:
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
    return _marengo35_vector(res.data)


def _marengo35_vector(data: Sequence['twelvelabs.EmbeddingData'] | None) -> pxt.Array[np.float32]:
    if not data or len(data) != 1 or len(data[0].embedding) != 512:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            'Expected a single 512-dimensional embedding from Marengo 3.5.',
            provider='twelvelabs',
        )
    return np.array(data[0].embedding, dtype='float32')


async def _embed_av_marengo35(
    file_path: str,
    input_type: Literal['audio', 'video'],
    start_sec: float | None,
    end_sec: float | None,
    embedding_option: Sequence[str] | None,
) -> pxt.Array[np.float32]:
    if start_sec is not None or end_sec is not None or embedding_option is not None:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            'Marengo 3.5 embeddings do not support start_sec, end_sec, or embedding_option. '
            'Split the media with audio_splitter or video_splitter before embedding it.',
        )
    b64_str = await asyncio.to_thread(_encode_short_media, file_path, input_type)
    if b64_str is not None:
        return await _embed_multi_input(input_type=input_type, b64_str=b64_str)
    # too large or too long for the synchronous API
    return await _embed_av_task(file_path, input_type)


def _encode_short_media(file_path: str, input_type: Literal['audio', 'video']) -> str | None:
    if os.stat(file_path).st_size > _SYNC_MAX_BYTES:
        return None
    duration = (
        av_utils.get_audio_duration(file_path) if input_type == 'audio' else av_utils.get_video_duration(file_path)
    )
    if duration is None:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, f'Cannot determine {input_type} duration.')
    if duration > _SYNC_MAX_SECS:
        return None
    return _encode_file(file_path)


def _encode_file(file_path: str) -> str:
    with open(file_path, 'rb') as fp:
        return b64encode(fp.read()).decode('utf-8')


async def _embed_av_task(file_path: str, input_type: Literal['audio', 'video']) -> pxt.Array[np.float32]:
    import twelvelabs

    if os.stat(file_path).st_size > _TASK_INLINE_MAX_BYTES:
        async with _asset_upload(input_type, file_path) as asset_id:
            return await _run_embedding_task(input_type, twelvelabs.MediaSource(asset_id=asset_id))
    b64_str = await asyncio.to_thread(_encode_file, file_path)
    return await _run_embedding_task(input_type, twelvelabs.MediaSource(base_64_string=b64_str))


async def _run_embedding_task(
    input_type: Literal['audio', 'video'], media_source: 'twelvelabs.MediaSource'
) -> pxt.Array[np.float32]:
    import twelvelabs

    request_cls = twelvelabs.AsyncAudioInputRequest if input_type == 'audio' else twelvelabs.AsyncVideoInputRequest
    request: dict[str, Any] = {
        input_type: request_cls(
            media_source=media_source,
            # the task returns one asset vector per option; a fused video vector would require time_based_metadata
            embedding_option=['audio' if input_type == 'audio' else 'visual'],
            embedding_scope=['asset'],
        )
    }
    tasks = _twelvelabs_client().embed.v_2.tasks
    task = await tasks.create(input_type=input_type, model_name='marengo3.5', embedding_dimension=512, **request)
    result = await _wait_until_ready(lambda: tasks.retrieve(task.id), f'TwelveLabs embedding task {task.id}')
    return _marengo35_vector(result.data)


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
        async with _asset_upload(input_type, file_path) as asset_id:
            create_kwargs = {
                'input_type': input_type,
                'model_name': model_name,
                input_type: request_cls(
                    media_source=twelvelabs.MediaSource(asset_id=asset_id),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    embedding_option=embedding_option,
                ),
            }
            res = await cl.embed.v_2.create(**create_kwargs)  # type: ignore[arg-type]
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
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
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
