"""
Pixeltable UDFs
that wrap various endpoints from the TwelveLabs API. In order to use them, you must
first `pip install twelvelabs` and configure your TwelveLabs credentials, as described in
the [Working with TwelveLabs](https://docs.pixeltable.com/howto/providers/working-with-twelvelabs) tutorial.
"""

import asyncio
import os
from base64 import b64encode
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Iterator, Literal, Sequence, TypedDict

import numpy as np
from tenacity import retry, retry_if_result, wait_exponential

import pixeltable as pxt
from pixeltable import env, exprs, type_system as ts
from pixeltable.config import Config
from pixeltable.runtime import get_runtime
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import retry_if_retryable_error
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import twelvelabs
    from twelvelabs.wrapper.multipart_upload_client_wrapper import UploadResult


TWELVELABS_INLINE_LIMIT_BYTES = 2 * 2**20
# https://docs.twelvelabs.io/docs/concepts/upload-methods
_ASYNC_INLINE_LIMIT_BYTES = 36 * 10**6
_DEFAULT_EMBEDDING_DIMENSIONS = {'Marengo-retrieval-2.7': 1024, 'marengo3.0': 512, 'marengo3.5': 512}


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


@asynccontextmanager
async def _asset_uploads(input_type: Literal['audio', 'video'], files: list[str]) -> AsyncIterator[list[str]]:
    """
    Context manager that makes uploaded files temporarily available to Twelvelabs models, deleting them from the server
    after use.

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

        for asset_id in uploaded:
            await _wait_for_asset(client, asset_id)

        yield uploaded

    finally:
        await asyncio.gather(*[client.assets.delete(asset_id) for asset_id in uploaded], return_exceptions=True)


def _timeout() -> float:
    timeout = Config.get().get_float_value('timeout', section='twelvelabs')
    if timeout is None:
        return 600
    if timeout <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_CONFIGURATION, 'twelvelabs.timeout must be positive.')
    return timeout


async def _wait_for_asset(client: 'twelvelabs.AsyncTwelveLabs', asset_id: str) -> None:
    timeout = _timeout()
    try:
        async with asyncio.timeout(timeout):
            await _poll_until_ready(client, asset_id)
    except TimeoutError as e:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_TIMEOUT,
            f'Timed out after {timeout:g} seconds waiting for TwelveLabs asset {asset_id} to become ready. '
            'Increase timeout in the [twelvelabs] config section to extend the timeout; '
            'TWELVELABS_TIMEOUT overrides the config file.',
            provider='twelvelabs',
        ) from e


@retry(
    retry=retry_if_result(lambda ready: not ready) | retry_if_retryable_error,
    wait=wait_exponential(multiplier=2, min=2, max=30),
)
async def _poll_until_ready(client: 'twelvelabs.AsyncTwelveLabs', asset_id: str) -> bool:
    asset = await client.assets.retrieve(asset_id)
    assert asset.status in ('processing', 'ready', 'failed'), f'Unexpected TwelveLabs asset status: {asset.status!r}'
    if asset.status == 'failed':
        message = '' if asset.error is None else f': {asset.error.message}'
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            f'TwelveLabs asset {asset_id} processing failed{message}',
            provider='twelvelabs',
        )
    return asset.status == 'ready'


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
    `video_splitter` before embedding them, or use `embed_iterator()` for asynchronous content embeddings.
    The `start_sec`, `end_sec`, and `embedding_option` parameters are supported only with Marengo 3.0.

    Equivalent to the TwelveLabs Embed API:
    <https://docs.twelvelabs.io/sdk-reference/python/create-embeddings-v-2/create-sync-embeddings>

    Request throttling:
    Applies the rate limit set in the config (section `twelvelabs`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    With Marengo 3.0, audio and video files over 2 MiB are uploaded first. The UDF waits for each uploaded
    asset to become ready, subject to `timeout` in the `twelvelabs` config section.
    `TWELVELABS_TIMEOUT` overrides the config file.

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
                pxt.ErrorCode.INVALID_ARGUMENT, 'embed() with Marengo 3.5 requires media of at most 32 MB.'
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
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
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
            'embed() with Marengo 3.5 does not support start_sec, end_sec, or embedding_option. '
            'Split the media with audio_splitter or video_splitter before embedding it, '
            'or use embed_iterator(), which supports these options.',
        )
    b64_str = await asyncio.to_thread(_encode_short_media, file_path, input_type)
    return await _embed_multi_input(input_type=input_type, b64_str=b64_str)


def _encode_short_media(file_path: str, input_type: Literal['audio', 'video']) -> str:
    if os.stat(file_path).st_size > 32 * 2**20:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            'embed() with Marengo 3.5 requires media of at most 32 MB. '
            'Split the media with audio_splitter or video_splitter before embedding it, '
            'or use embed_iterator(), which accepts longer media.',
        )
    duration = (
        av_utils.get_audio_duration(file_path) if input_type == 'audio' else av_utils.get_video_duration(file_path)
    )
    if duration is None:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, f'Cannot determine {input_type} duration.')
    if duration > 30:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'embed() with Marengo 3.5 requires {input_type} of at most 30 seconds; '
            f'got {duration:.2f} seconds. '
            'Split the media with audio_splitter or video_splitter before embedding it, '
            'or use embed_iterator(), which accepts longer media.',
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


class EmbeddingResult(TypedDict):
    """One embedding record returned by `embed_iterator()`."""

    embedding: pxt.Array[(None,), np.float32]
    embedding_option: str | None
    """Provider modality label, such as `visual`, `audio`, or `fused`."""
    embedding_scope: str | None
    """Provider scope label: `clip` for a segment or `asset` for the entire file."""
    start_sec: float | None
    """Start of the embedded segment in seconds, or None if not returned by the provider."""
    end_sec: float | None
    """End of the embedded segment in seconds, or None if not returned by the provider."""
    embedding_uncertainty: pxt.Array[(None,), np.float32] | None
    """Optional per-dimension uncertainty; higher values indicate lower confidence in that dimension."""
    metadata: pxt.Json | None
    """Metadata for the input file, such as duration and requested modalities/scopes, shared by all its embeddings."""


@pxt.iterator
def embed_iterator(
    *,
    model_name: str,
    audio: pxt.Audio | None = None,
    video: pxt.Video | None = None,
    media_kwargs: dict[str, Any] | None = None,
    embedding_dimension: int | None = None,
    embedding_uncertainty: bool | None = None,
) -> Iterator[EmbeddingResult]:
    """
    Creates asynchronous audio or video embeddings, yielding one row per provider embedding record.

    Wraps the TwelveLabs `embed.v_2.tasks.create` and `retrieve` endpoints:
    <https://docs.twelvelabs.io/sdk-reference/python/create-embeddings-v-2/create-async-embeddings>

    Embeds audio and video of any duration supported by the model. By default, returns embeddings for individual
    segments and the whole file, with separate visual and audio embeddings for Marengo 3.5 video. Use `media_kwargs`
    to choose modalities, scopes, and segmentation. The resulting vectors can be indexed for similarity search;
    use `embed()` to create query embeddings.

    The iterator waits up to 600 seconds per input file, including upload processing. Set
    `timeout` in the `twelvelabs` config section to change this limit;
    `TWELVELABS_TIMEOUT` overrides the config file.

    The iterator processes input files sequentially and does not use the UDF request-rate resource pool.

    __Requirements:__

    - `pip install 'twelvelabs>=1.3.6'`

    Args:
        model_name: The embedding model (`marengo3.0` or `marengo3.5`).
        audio: The audio file to embed. Specify exactly one of `audio` and `video`. A `None` value yields no rows.
        video: The video file to embed. A `None` value yields no rows.
        media_kwargs: Additional fields for the provider's `AsyncAudioInputRequest` or `AsyncVideoInputRequest`,
            such as `start_sec`, `end_sec`, `embedding_option`, `embedding_scope`, `embedding_type`,
            `segmentation`, and `time_based_metadata`. `media_source` is managed by Pixeltable and cannot be set here.
        embedding_dimension: Output dimension. Must be constant. Omit to use the model's default; specify this
            when indexing embeddings from an unrecognized model.
        embedding_uncertainty: Whether to request per-dimension uncertainty. For audio and video, set
            `media_kwargs['embedding_scope']` without `asset`; the default scope includes it and the request fails.

    Examples:
        Embed videos and index every returned vector:

        >>> v = pxt.create_view(
        ...     'video_embeddings',
        ...     videos,
        ...     iterator=embed_iterator(video=videos.video, model_name='marengo3.5'),
        ... )
        >>> v.add_embedding_index(
        ...     v.embedding, embedding=embed.using(model_name='marengo3.5')
        ... )

        Limit processing to the first minute by passing provider options:

        >>> clips = pxt.create_view(
        ...     'first_minute_embeddings',
        ...     videos,
        ...     iterator=embed_iterator(
        ...         video=videos.video,
        ...         model_name='marengo3.5',
        ...         media_kwargs={'end_sec': 60.0, 'embedding_scope': ['clip']},
        ...     ),
        ... )

        Search the indexed embeddings:

        >>> score = v.embedding.similarity(string='a dog barking')
        >>> v.select(v.video, v.embedding_option, v.start_sec, score=score).order_by(
        ...     score, asc=False
        ... ).limit(10).collect()
    """
    file_path = audio if audio is not None else video
    if file_path is None:
        return
    if media_kwargs is not None and 'media_source' in media_kwargs:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'media_kwargs cannot override media_source.')
    env.Env.get().require_package('twelvelabs', min_version=[1, 3, 6])
    input_type: Literal['audio', 'video'] = 'audio' if audio is not None else 'video'
    task_options = {'embedding_dimension': embedding_dimension, 'embedding_uncertainty': embedding_uncertainty}
    result = get_runtime().run_coro(
        _create_embedding_task(
            file_path,
            input_type,
            model_name,
            media_kwargs if media_kwargs is not None else {},
            {k: v for k, v in task_options.items() if v is not None},
        )
    )
    if not result.data:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, 'TwelveLabs returned no embedding data.', provider='twelvelabs'
        )
    metadata = None if result.metadata is None else result.metadata.model_dump(mode='json')
    for record in result.data:
        yield EmbeddingResult(
            embedding=np.array(record.embedding, dtype=np.float32),
            embedding_option=record.embedding_option,
            embedding_scope=record.embedding_scope,
            start_sec=record.start_sec,
            end_sec=record.end_sec,
            embedding_uncertainty=(
                None
                if record.embedding_uncertainty is None
                else np.array(record.embedding_uncertainty, dtype=np.float32)
            ),
            metadata=metadata,
        )


@embed_iterator.conditional_output_schema
def _(bound_args: dict[str, exprs.Expr]) -> dict[str, type]:
    if ('audio' in bound_args) == ('video' in bound_args):
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'Specify exactly one of audio and video.')
    dimension_expr = bound_args.get('embedding_dimension')
    dimension: int | None = None
    if dimension_expr is not None:
        if not isinstance(dimension_expr, exprs.Literal):
            raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'embedding_dimension must be constant.')
        if dimension_expr.val is not None:
            dimension = dimension_expr.val
    if dimension is not None and dimension <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'embedding_dimension must be positive.')
    if dimension is None:
        model_expr = bound_args.get('model_name')
        if isinstance(model_expr, exprs.Literal):
            dimension = _DEFAULT_EMBEDDING_DIMENSIONS.get(model_expr.val)
    return {
        **EmbeddingResult.__annotations__,
        'embedding': pxt.Array[(dimension,), np.float32],
        'embedding_uncertainty': pxt.Array[(dimension,), np.float32] | None,
    }


async def _create_embedding_task(
    file_path: str,
    input_type: Literal['audio', 'video'],
    model_name: str,
    options: dict[str, Any],
    task_options: dict[str, Any],
) -> 'twelvelabs.EmbeddingTaskResponse':
    import twelvelabs

    timeout = _timeout()

    async def embed_source(source: 'twelvelabs.MediaSource') -> 'twelvelabs.EmbeddingTaskResponse':
        request_cls = twelvelabs.AsyncAudioInputRequest if input_type == 'audio' else twelvelabs.AsyncVideoInputRequest
        request = request_cls(media_source=source, **options)
        client = _twelvelabs_client()
        kwargs = {input_type: request, **task_options}
        task = await client.embed.v_2.tasks.create(input_type=input_type, model_name=model_name, **kwargs)
        return await _poll_embedding_task(client, task.id)

    try:
        async with asyncio.timeout(timeout):
            if os.stat(file_path).st_size <= _ASYNC_INLINE_LIMIT_BYTES:

                def encode() -> str:
                    with open(file_path, 'rb') as fp:
                        return b64encode(fp.read()).decode('utf-8')

                b64_str = await asyncio.to_thread(encode)
                return await embed_source(twelvelabs.MediaSource(base_64_string=b64_str))
            async with _asset_uploads(input_type, [file_path]) as asset_ids:
                return await embed_source(twelvelabs.MediaSource(asset_id=asset_ids[0]))
    except TimeoutError as e:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_TIMEOUT,
            f'Timed out after {timeout:g}s waiting for TwelveLabs embeddings. '
            'Increase timeout in the [twelvelabs] config section to extend the timeout; '
            'TWELVELABS_TIMEOUT overrides the config file.',
            provider='twelvelabs',
        ) from e


@retry(
    retry=retry_if_result(lambda result: result.status != 'ready') | retry_if_retryable_error,
    wait=wait_exponential(multiplier=2, min=2, max=30),
)
async def _poll_embedding_task(
    client: 'twelvelabs.AsyncTwelveLabs', task_id: str
) -> 'twelvelabs.EmbeddingTaskResponse':
    result = await client.embed.v_2.tasks.retrieve(task_id)
    if result.status == 'failed':
        message = result.error.message if result.error is not None else 'Unknown error'
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR, f'TwelveLabs embedding task failed: {message}', provider='twelvelabs'
        )
    return result


@embed.conditional_return_type
def _(model_name: str) -> ts.ArrayType:
    dimension = _DEFAULT_EMBEDDING_DIMENSIONS.get(model_name)
    shape = None if dimension is None else (dimension,)
    return ts.ArrayType(shape=shape, dtype=np.dtype('float32'))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
