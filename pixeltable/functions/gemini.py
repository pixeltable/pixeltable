"""
Pixeltable UDFs
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-genai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://docs.pixeltable.com/howto/providers/working-with-gemini) tutorial.

Supports two authentication methods:

- Google AI Studio: set `GOOGLE_API_KEY` or `GEMINI_API_KEY` (or put `api_key` in the `gemini` section of
  the Pixeltable config file).
- Vertex AI: set `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_PROJECT` (and optionally
  `GOOGLE_CLOUD_LOCATION`), then authenticate via Application Default Credentials
  (e.g. `gcloud auth application-default login`).
"""

import asyncio
import base64
import io
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Sequence

import numpy as np
import PIL.Image
from tenacity import RetryCallState, retry, retry_if_result, stop_after_delay, wait_exponential

import pixeltable as pxt
from pixeltable import env, exceptions as excs, exprs, type_system as ts
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import exponential_backoff, parse_duration_str
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from google import genai

_logger = logging.getLogger('pixeltable')

# Max raw file size (bytes) for inline data; larger files use the Files API.
GEMINI_INLINE_LIMIT_BYTES = 4 * 2**20

# Placeholder key used in first pass for large file uploads.
_UPLOAD_PLACEHOLDER_KEY = '__google_genai_upload_ref__'


@env.register_client('gemini')
def _(api_key: str | None = None) -> 'genai.client.Client':
    from google import genai

    try:
        if api_key is not None:
            return genai.client.Client(api_key=api_key)
        # Vertex AI fall-through: rely on genai.client.Client to read its own env vars
        # (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION)
        return genai.client.Client()
    except Exception as e:
        raise excs.Error(
            'Gemini client not initialized. '
            'For the Gemini Developer API set GOOGLE_API_KEY or GEMINI_API_KEY, '
            'or set api_key in the [gemini] section of $PIXELTABLE_HOME/config.toml. '
            'For Vertex AI set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT, '
            'then authenticate via: gcloud auth application-default login'
        ) from e


def _genai_client() -> 'genai.client.Client':
    return get_runtime().get_client('gemini')


class GeminiRateLimitsInfo(env.RateLimitsInfo):
    def __init__(self) -> None:
        super().__init__(self._get_request_resources)

    def _get_request_resources(self) -> dict[str, int]:
        # TODO(PXT-996): Improve resource tracking for Gemini UDFs
        return {}

    def is_initialized(self) -> bool:
        return True

    def get_retry_delay(self, exc: Exception, attempt: int) -> float | None:
        if hasattr(exc, 'code') and exc.code == 429:
            try:
                for detail_dict in exc.details['error']['details']:  # type: ignore[attr-defined]
                    if detail_dict.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        delay = parse_duration_str(detail_dict['retryDelay'])
                        return delay
            except (AttributeError, KeyError, TypeError):
                return exponential_backoff(attempt)
        return None


@pxt.udf(is_deterministic=False)
async def generate_content(
    contents: pxt.Json,
    *,
    model: str,
    config: dict | None = None,
    tools: list[dict] | None = None,
    _runtime_ctx: env.RuntimeCtx | None = None,
) -> dict:
    """
    Generate content from the specified model.

    Request throttling:
    Applies the rate limit set in the config (section `gemini.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The input content to generate from. Can be a prompt, or a list containing images and text
            prompts, as described in: <https://ai.google.dev/gemini-api/docs/text-generation>
        model: The name of the model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateContentConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig>
        tools: An optional list of Pixeltable tools to use. It is also possible to specify tools manually via the
            `config['tools']` parameter, but at most one of `config['tools']` or `tools` may be used.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gemini-2.5-flash`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=generate_content(tbl.prompt, model='gemini-2.5-flash')
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_: types.GenerateContentConfig
    if config is None and tools is None:
        config_ = None
    else:
        if config is None:
            config_ = types.GenerateContentConfig()
        else:
            config_ = types.GenerateContentConfig(**config)
        if tools is not None:
            gemini_tools = [__convert_pxt_tool(tool) for tool in tools]
            config_.tools = [types.Tool(function_declarations=gemini_tools)]

    large_video_paths: list[str] = []
    client = _genai_client()

    contents = _process_media_contents(contents, large_video_paths)
    async with _gemini_file_uploads(large_video_paths) as uploaded:
        contents = _replace_upload_placeholders(contents, uploaded)
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config_)
        return response.model_dump(mode='json')


@asynccontextmanager
async def _gemini_file_uploads(files: list[str]) -> AsyncIterator[list['genai.types.File']]:
    """
    Context manager that makes uploaded files temporarily available to Gemini models, deleting them from the server
    after use.
    """
    client = _genai_client()
    uploaded: list['genai.types.File'] = []

    try:
        if len(files) > 0:
            tasks: list[Coroutine[Any, Any, 'genai.types.File']] = []
            for file in files:
                mime_type, _ = mimetypes.guess_type(file, strict=False)
                if mime_type is None:
                    raise excs.Error(f'Could not identify mime type of file: {file}')
                tasks.append(client.aio.files.upload(file=file, config={'mime_type': mime_type}))
            uploaded = await asyncio.gather(*tasks)
            # poll till server finished uploading files (state is ACTIVE)
            await _poll_until_active(async_client=client.aio, uploaded=uploaded, video_paths=files)

        yield uploaded

    finally:
        if len(uploaded) > 0:
            await asyncio.gather(*[client.aio.files.delete(name=f.name) for f in uploaded], return_exceptions=True)


def __convert_pxt_tool(pxt_tool: dict) -> dict:
    return {
        'name': pxt_tool['name'],
        'description': pxt_tool['description'],
        'parameters': {
            'type': 'object',
            'properties': pxt_tool['parameters']['properties'],
            'required': pxt_tool['required'],
        },
    }


def invoke_tools(tools: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenAI response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_gemini_response_to_pxt_tool_calls(response))


@pxt.udf
def _gemini_response_to_pxt_tool_calls(response: dict) -> dict | None:
    pxt_tool_calls: dict[str, list[dict]] = {}
    for part in response['candidates'][0]['content']['parts']:
        tool_call = part.get('function_call')
        if tool_call is not None:
            tool_name = tool_call['name']
            if tool_name not in pxt_tool_calls:
                pxt_tool_calls[tool_name] = []
            pxt_tool_calls[tool_name].append({'args': tool_call['args']})
    if len(pxt_tool_calls) == 0:
        return None
    return pxt_tool_calls


@generate_content.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(is_deterministic=False)
async def generate_images(
    prompt: str, *, model: str, config: dict | None = None, _runtime_ctx: env.RuntimeCtx | None = None
) -> PIL.Image.Image:
    """
    Generates images based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/image-generation>

    Request throttling:
    Applies the rate limit set in the config (section `imagen.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the images to generate.
        model: The model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateImagesConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig>

    Returns:
        The generated image.

    Examples:
        Add a computed column that applies the model `imagen-4.0-generate-001`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=generate_images(tbl.prompt, model='imagen-4.0-generate-001')
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai.types import GenerateImagesConfig

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_ = GenerateImagesConfig(**config) if config else None
    response = await _genai_client().aio.models.generate_images(model=model, prompt=prompt, config=config_)
    return response.generated_images[0].image._pil_image


@generate_images.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(is_deterministic=False)
async def generate_videos(
    prompt: str | None = None,
    image: PIL.Image.Image | None = None,
    *,
    model: str,
    config: dict | None = None,
    _runtime_ctx: env.RuntimeCtx | None = None,
) -> pxt.Video:
    """
    Generates videos based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/video>

    At least one of `prompt` or `image` must be provided.

    Request throttling:
    Applies the rate limit set in the config (section `veo.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the videos to generate.
        image: An image to use as the first frame of the video.
        model: The model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateVideosConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig>

    Returns:
        The generated video.

    Examples:
        Add a computed column that applies the model `veo-3.0-generate-001`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=generate_videos(tbl.prompt, model='veo-3.0-generate-001')
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    if prompt is None and image is None:
        raise excs.Error('At least one of `prompt` or `image` must be provided.')

    image_: types.Image | None = None
    if image is not None:
        with io.BytesIO() as buffer:
            image.save(buffer, format='webp')
            image_ = types.Image(image_bytes=buffer.getvalue(), mime_type='image/webp')

    config_ = types.GenerateVideosConfig(**config) if config else None

    operation = await _genai_client().aio.models.generate_videos(
        model=model, prompt=prompt, image=image_, config=config_
    )
    while not operation.done:
        await asyncio.sleep(3)
        operation = await _genai_client().aio.operations.get(operation)

    if operation.error:
        raise Exception(f'Video generation failed: {operation.error}')

    video = operation.response.generated_videos[0]

    video_bytes = await _genai_client().aio.files.download(file=video.video)  # type: ignore[arg-type]
    assert video_bytes is not None

    # Create a temporary file to store the video bytes
    output_path = TempStore.create_path(extension='.mp4')
    Path(output_path).write_bytes(video_bytes)
    return str(output_path)


@generate_videos.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(batch_size=4)
async def embed_content(
    contents: Batch[str], *, model: str, config: dict[str, Any] | None = None, use_batch_api: bool = False
) -> Batch[pxt.Array[(None,), np.float32]]:
    """
    Generate embeddings for text, images, video, and other content. For more information on Gemini embeddings API, see:
    <https://ai.google.dev/gemini-api/docs/embeddings>

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The string, image, audio, video, or document to embed.
        model: The Gemini model to use.
        config: Configuration for embedding generation, corresponding to keyword arguments of
            `genai.types.EmbedContentConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig>
        use_batch_api: If True, use [Gemini's Batch API](https://ai.google.dev/gemini-api/docs/batch-api) that provides
            a higher throughput at a lower cost at the expense of higher latency.

    Returns:
        The corresponding embedding vector.

    Examples:
        Add a computed column with embeddings to an existing table with a `text` column:

        >>> t.add_computed_column(
        ...     embedding=embed_content(t.text, model='gemini-embedding-001')
        ... )

        Add an embedding index on `text` column:

        >>> t.add_embedding_index(
        ...     t.text, embedding=embed_content.using(model='gemini-embedding-001')
        ... )
    """
    return await _embed_content(contents, model, config, use_batch_api)


@embed_content.overload
async def _(
    contents: Batch[PIL.Image.Image], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_content(contents, model, config, use_batch_api=False)


@embed_content.overload
async def _(
    contents: Batch[pxt.Audio], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config, use_batch_api=False)


@embed_content.overload
async def _(
    contents: Batch[pxt.Video], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config, use_batch_api=False)


@embed_content.overload
async def _(
    contents: Batch[pxt.Document], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config, use_batch_api=False)


async def _embed_file_content(
    contents: list[str], model: str, config: dict[str, Any] | None, use_batch_api: bool
) -> Batch[pxt.Array[(None,), np.float32]]:
    env.Env.get().require_package('google.genai')
    from google.genai import types

    large_files: list[str] = []
    for item in contents:
        size_bytes = os.stat(item).st_size
        if size_bytes > GEMINI_INLINE_LIMIT_BYTES:
            large_files.append(item)

    async with _gemini_file_uploads(large_files) as uploaded:
        upload_map = dict(zip(large_files, uploaded))
        contents_: list[types.ContentUnion] = []
        for item in contents:
            if item in upload_map:
                contents_.append(upload_map[item])
            else:
                mime_type, _ = mimetypes.guess_type(item, strict=False)
                if mime_type is None:
                    raise excs.Error(f'Could not identify mime type of file: {item}')

                try:
                    # TODO: Do this on a background thread.
                    data = Path(item).read_bytes()
                except (OSError, ValueError) as exc:
                    raise excs.Error(f'Error reading file for embedding: {item}') from exc

                contents_.append(types.Part.from_bytes(data=data, mime_type=mime_type))

        return await _embed_content(contents_, model, config, use_batch_api)


async def _embed_content(
    contents: Sequence['genai.types.ContentUnion'], model: str, config: dict[str, Any] | None, use_batch_api: bool
) -> Batch[pxt.Array[(None,), np.float32]]:
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    client = _genai_client()
    config_ = _embedding_config(config)

    if not use_batch_api:
        result = await client.aio.models.embed_content(model=model, contents=list(contents), config=config_)
        assert len(result.embeddings) == len(contents)
        return [np.array(emb.values, dtype=np.float32) for emb in result.embeddings]

    # Batch API
    batch_job = client.batches.create_embeddings(
        model=model,
        src=types.EmbeddingsBatchJobSource(inlined_requests=types.EmbedContentBatch(contents=contents, config=config_)),
    )

    await asyncio.sleep(3)
    i = 0
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        if batch_job.state in (
            types.JobState.JOB_STATE_SUCCEEDED,
            types.JobState.JOB_STATE_FAILED,
            types.JobState.JOB_STATE_CANCELLED,
            types.JobState.JOB_STATE_EXPIRED,
        ):
            break
        delay = min(10 + i * 2, 30)
        _logger.debug(
            f'Waiting for embedding batch job {batch_job.name} to complete. Latest state: {batch_job.state}. Sleeping'
            f' for {delay}s before the next attempt.'
        )
        await asyncio.sleep(delay)
        i += 1

    if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
        raise excs.Error(f'Embedding batch job did not succeed: {batch_job.state}. Error: {batch_job.error}')

    assert batch_job.error is None
    results = []
    for resp in batch_job.dest.inlined_embed_content_responses:
        assert resp.error is None
        results.append(np.array(resp.response.embedding.values, dtype=np.float32))
    return results


_DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL: dict[str, int] = {
    'gemini-embedding-001': 3072,
    'gemini-embedding-2-preview': 3072,
}


@embed_content.conditional_return_type
def _(model: str, config: dict | None) -> ts.ArrayType:
    config_ = _embedding_config(config)
    dim = config_.output_dimensionality
    if dim is None and model in _DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL:
        dim = _DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL.get(model)
    return ts.ArrayType((dim,), dtype=np.dtype('float32'), nullable=False)


@embed_content.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


def _embedding_config(config: dict | None) -> 'genai.types.EmbedContentConfig':
    from google.genai import types

    return types.EmbedContentConfig(**config) if config else types.EmbedContentConfig()


def _is_processing(remote_files: list['genai.types.File']) -> bool:
    from google.genai import types

    return any(file.state != types.FileState.ACTIVE for file in remote_files)


def _handle_polling_timeout(retry_state: RetryCallState) -> None:
    """Triggered when timeout is reached."""
    from google.genai import types

    remote_files: list[types.File] = retry_state.outcome.result()

    # Extract video_paths from the keyword arguments
    video_paths: list[str] = retry_state.kwargs.get('video_paths', [])
    stuck_details = []
    for i, file in enumerate(remote_files):
        if file.state != types.FileState.ACTIVE:
            path = video_paths[i] if i < len(video_paths) else 'Unknown path'
            stuck_details.append(f'{path} (ID: {file.name}, State: {file.state.name})')

    detail_str = '\n- '.join(stuck_details)
    raise excs.Error(
        f'Timeout: {len(stuck_details)}/{len(remote_files)} failed to upload large videos :\n- {detail_str}'
    )


@retry(
    retry=retry_if_result(_is_processing),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_delay(600),
    retry_error_callback=_handle_polling_timeout,
)
async def _poll_until_active(
    async_client: 'genai.client.AsyncClient', uploaded: list['genai.types.File'], video_paths: list[str]
) -> list['genai.types.File']:
    from google.genai import types

    # Collect statuses for all uploaded files
    remote_files = await asyncio.gather(*[async_client.files.get(name=f.name) for f in uploaded])
    for i, file in enumerate(remote_files):
        if file.state == types.FileState.FAILED:
            # Fail immediately
            raise excs.Error(f'Server processing failed for {video_paths[i]} ({file.name})')
    return remote_files


def _process_media_contents(data: Any, large_video_paths: list[str]) -> Any:
    """
    Recursively traverse a nested content structure (dict/list/str) and process video file paths.

    - Strings that are not local video file paths are returned unchanged.
    - Small video files (<= GEMINI_INLINE_LIMIT_BYTES * 0.75) are base64-encoded inline.
    - Large video files are queued for async upload and replaced with a placeholder dict
      (keyed by _UPLOAD_PLACEHOLDER_KEY) to be resolved later by _replace_upload_placeholders.

    Returns the same nested structure with video path strings replaced by inline_data or placeholder dicts.
    """
    if isinstance(data, dict):
        return {k: _process_media_contents(v, large_video_paths) for k, v in data.items()}
    if isinstance(data, list):
        return [_process_media_contents(v, large_video_paths) for v in data]
    if isinstance(data, str):
        # Check if string is a file path containing video
        mime_type, _ = mimetypes.guess_type(data, strict=False)
        if mime_type is None or not mime_type.lower().startswith('video/'):
            return data
        local_path = Path(data).expanduser()
        try:
            if not local_path.exists():
                return data
        except (OSError, ValueError):
            return data
        size_bytes = local_path.stat().st_size
        if size_bytes <= GEMINI_INLINE_LIMIT_BYTES * 3 // 4:  # scale by 0.75 to account for base64 expansion
            # TODO: Do this on a background thread.
            data_b64 = base64.b64encode(local_path.read_bytes()).decode('utf-8')
            return {'inline_data': {'mime_type': mime_type, 'data': data_b64}}
        # Record the large file for upload and insert a placeholder to be resolved later
        large_video_paths.append(str(local_path))
        return {_UPLOAD_PLACEHOLDER_KEY: {'task_id': len(large_video_paths) - 1, 'mime_type': mime_type}}
    return data


def _replace_upload_placeholders(obj: Any, uploaded: list['genai.types.File']) -> Any:
    """
    Recursively traverse a nested content structure (dict/list/str) and resolve upload placeholders.

    Returns the same nested structure with all placeholders replaced by file_data dicts.
    """
    if isinstance(obj, dict) and _UPLOAD_PLACEHOLDER_KEY in obj:
        idx = obj[_UPLOAD_PLACEHOLDER_KEY]['task_id']
        mime_type = obj[_UPLOAD_PLACEHOLDER_KEY]['mime_type']
        f = uploaded[idx]
        return {'file_data': {'file_uri': f.uri, 'mime_type': mime_type}}
    if isinstance(obj, dict):
        return {k: _replace_upload_placeholders(v, uploaded) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_upload_placeholders(v, uploaded) for v in obj]
    return obj


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
