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
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Literal, Sequence

import numpy as np

import pixeltable as pxt
from pixeltable import env, type_system as ts
from pixeltable.runtime import get_runtime
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import twelvelabs
    from twelvelabs.wrapper.multipart_upload_client_wrapper import UploadResult


TWELVELABS_INLINE_LIMIT_BYTES = 2 * 2**20


@env.register_client('twelvelabs', credential_param='api_key')
def _(api_key: str) -> 'twelvelabs.AsyncTwelveLabs':
    import twelvelabs

    return twelvelabs.AsyncTwelveLabs(api_key=api_key)


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

    Marengo 3.5 uses the synchronous `multi_input` API. Audio and video must be at most 30 seconds and 32 MB.
    Split longer files with `audio_splitter` or `video_splitter` before embedding them. The `start_sec`,
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
    res = await _twelvelabs_client().embed.v_2.create(
        input_type='multi_input',
        model_name='marengo3.5',
        embedding_dimension=512,
        multi_input=twelvelabs.MultiInputRequest(input_text=text, media_sources=media_sources),
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
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'Cannot determine {input_type} duration: {file_path}'
        )
    if duration > 30:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'Marengo 3.5 synchronous embeddings require {input_type} of at most 30 seconds; '
            f'got {duration:.2f} seconds. '
            'Split the media with audio_splitter or video_splitter before embedding it.',
        )
    with open(file_path, 'rb') as fp:
        b64_str = b64encode(fp.read()).decode('utf-8')
    return await _embed_multi_input(input_type=input_type, b64_str=b64_str)


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
