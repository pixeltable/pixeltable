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
    specify an image as well, corresponding to the `text_image` embedding type in the TwelveLabs API. This is
    (currently) the only way to include more than one input type at a time.

    Equivalent to the TwelveLabs Embed API:
    <https://docs.twelvelabs.io/v1.3/docs/guides/create-embeddings>

    Request throttling:
    Applies the rate limit set in the config (section `twelvelabs`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install twelvelabs`

    Args:
        model_name: The name of the model to use. Check
            [the TwelveLabs documentation](https://docs.twelvelabs.io/v1.3/sdk-reference/python/create-embeddings-v-1/create-text-image-and-audio-embeddings)
            for available models.
        text: The text to embed.
        image: If specified, the embedding will be created from both the text and the image.

    Returns:
        The embedding.

    Examples:
        Add a computed column `embed` for an embedding of a string column `input`:

        >>> tbl.add_computed_column(
        ...     embed=embed(model_name='marengo3.0', text=tbl.input)
        ... )
    """
    env.Env.get().require_package('twelvelabs')
    import twelvelabs

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

    return await _embed_av_content(
        file_path=video,
        input_type='video',
        request_cls=twelvelabs.VideoInputRequest,
        model_name=model_name,
        start_sec=start_sec,
        end_sec=end_sec,
        embedding_option=embedding_option,
    )


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
    if model_name == 'marengo3.0':
        return ts.ArrayType(shape=(512,), dtype=np.dtype('float32'))
    return ts.ArrayType(dtype=np.dtype('float32'))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
