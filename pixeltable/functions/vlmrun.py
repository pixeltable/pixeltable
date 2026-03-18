"""
Pixeltable UDFs that wrap the VLM Run API.

In order to use them, you must first ``pip install vlmrun`` and ``pip install openai``,
and configure your VLM Run API key as described in the
`VLM Run documentation <https://docs.vlm.run/>`_.
"""

import asyncio
import copy
import io
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    import openai

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client registration
# ---------------------------------------------------------------------------


@register_client('vlmrun')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://agent.vlm.run/v1/openai',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _vlmrun_client() -> 'openai.AsyncOpenAI':
    return Env.get().get_client('vlmrun')


# Lazy-initialized sync VLMRun client for file upload & artifact download.
_sync_client_lock = threading.Lock()
_sync_client_cache: dict[str, Any] = {}


def _vlmrun_sync_client() -> Any:
    """Return a cached synchronous ``VLMRun`` client (lazy, thread-safe)."""
    if 'client' in _sync_client_cache:
        return _sync_client_cache['client']
    with _sync_client_lock:
        if 'client' in _sync_client_cache:
            return _sync_client_cache['client']
        from vlmrun.client import VLMRun  # type: ignore[import-untyped]

        api_key = _vlmrun_client().api_key
        client = VLMRun(api_key=api_key, base_url='https://agent.vlm.run/v1')
        _sync_client_cache['client'] = client
        return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.mp4', '.mov', '.avi', '.mkv', '.webm'}


async def _upload_files(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan *messages* for ``file_path`` entries, upload each via the sync SDK, and replace with ``file_id``."""
    processed = copy.deepcopy(messages)
    for message in processed:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'input_file':
                continue
            if 'file_id' in item:
                raise ValueError(
                    "Passing 'file_id' directly is not supported. "
                    "Use 'file_path' instead; files are uploaded automatically."
                )
            file_path_str = item.get('file_path')
            if file_path_str is None:
                continue
            fp = Path(file_path_str)
            ext = fp.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f'Unsupported file format: {ext}. Supported formats: {", ".join(sorted(_SUPPORTED_EXTENSIONS))}'
                )
            uploaded = await asyncio.to_thread(lambda p=fp: _vlmrun_sync_client().files.upload(file=p))
            item.pop('file_path')
            item['file_id'] = uploaded.id
    return processed


async def _download_artifact(
    object_id: str, session_id: str, *, poll_interval: float = 5.0, timeout: float = 600.0
) -> bytes:
    """Download an artifact's raw bytes, polling until ready or *timeout* seconds elapse.

    Works around an SDK bug where ``artifacts.get()`` uses a path-based URL
    (``/artifacts/{session_id}/{object_id}``) instead of query parameters.
    """
    import time

    import requests

    def _fetch() -> bytes:
        client = _vlmrun_sync_client()
        url = f'{client.base_url}/artifacts'
        headers = {'Authorization': f'Bearer {client.api_key}'}
        params = {'object_id': object_id, 'session_id': session_id}
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            attempt += 1
            resp = requests.get(url, params=params, headers=headers, timeout=120)
            if resp.status_code == 200:
                data = resp.content
                # For url_ artifacts the body is a signed URL — follow it
                if data.startswith(b'http'):
                    actual = requests.get(data.decode('utf-8').strip(), timeout=120)
                    actual.raise_for_status()
                    return actual.content
                return data
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f'Artifact {object_id} not ready after {timeout}s ({attempt} attempts, status {resp.status_code})'
                )
            _logger.warning(
                'Artifact %s attempt %d returned status %d, retrying...', object_id, attempt, resp.status_code
            )
            time.sleep(poll_interval)

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# UDFs
# ---------------------------------------------------------------------------


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_completions(
    messages: list, *, model: str = 'vlmrun-orion-1:auto', model_kwargs: dict[str, Any] | None = None
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the VLM Run chat completions API endpoint.
    For additional details, see: <https://docs.vlm.run/>

    Files referenced via ``file_path`` in message content items are automatically
    uploaded before the API call.

    Request throttling:
    Applies the rate limit set in the config (section ``vlmrun``, key ``rate_limit``).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - ``pip install vlmrun openai``

    Args:
        messages: A list of messages for the chat conversation.  Supports
            ``{'type': 'input_file', 'file_path': '/path/to/file'}`` content items;
            files are uploaded automatically.
        model: The model to use.  Options: ``'vlmrun-orion-1:fast'``,
            ``'vlmrun-orion-1:auto'``, ``'vlmrun-orion-1:pro'``.
        model_kwargs: Additional keyword args for the VLM Run API.

    Returns:
        A dictionary containing the API response.

    Examples:
        Describe an image:

        >>> messages = [{'role': 'user', 'content': [
        ...     {'type': 'text', 'text': 'Describe this image'},
        ...     {'type': 'input_file', 'file_path': t.image.localpath},
        ... ]}]
        >>> t.add_computed_column(response=vlmrun.chat_completions(messages))
    """
    processed = await _upload_files(messages)
    kwargs = dict(model_kwargs) if model_kwargs else {}
    result = await _vlmrun_client().chat.completions.with_raw_response.create(messages=processed, model=model, **kwargs)
    raw = json.loads(result.text)
    raw.pop('session_id', None)
    return raw


@pxt.udf(resource_pool='request-rate:vlmrun')
async def generate_image(
    prompt: str,
    *,
    file_path: str | None = None,
    model: str = 'vlmrun-orion-1:auto',
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> PIL.Image.Image:
    """
    Generates or edits an image using VLM Run.

    When *file_path* is ``None``, generates a new image from the text prompt.
    When *file_path* is provided, edits the referenced image according to the prompt.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - ``pip install vlmrun openai``

    Args:
        prompt: Text prompt describing the image to generate or the edit to apply.
        file_path: Optional local path to an input image for editing.  For Image
            columns, use ``.localpath``.  Omit for text-to-image generation.
        model: The model to use.  Defaults to ``'vlmrun-orion-1:auto'``.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact.  Defaults to 600.

    Returns:
        The generated or edited image.

    Examples:
        Generate an image from text:

        >>> tbl.add_computed_column(image=vlmrun.generate_image(tbl.prompt))

        Edit an existing image:

        >>> tbl.add_computed_column(edited=vlmrun.generate_image(
        ...     'Blur all faces', file_path=tbl.image.localpath
        ... ))
    """
    from pydantic import BaseModel as _BaseModel
    from vlmrun.types.refs import ImageRef  # type: ignore[import-untyped]

    # Build messages
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    if file_path is not None:
        content.append({'type': 'input_file', 'file_path': file_path})
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'Always return a generated image in your response.'},
        {'role': 'user', 'content': content},
    ]

    # Build kwargs
    kwargs = dict(model_kwargs) if model_kwargs else {}
    schema = type('_Out', (_BaseModel,), {'__annotations__': {'image': ImageRef}}).model_json_schema()
    kwargs['response_format'] = {'type': 'json_schema', 'schema': schema}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = ['image-gen']
    kwargs['extra_body'] = extra_body

    # Upload files and call API
    processed = await _upload_files(messages)
    result = await _vlmrun_client().chat.completions.with_raw_response.create(messages=processed, model=model, **kwargs)
    raw = json.loads(result.text)

    # Extract session_id (ephemeral) and artifact ID
    session_id = raw.get('session_id')
    if not session_id:
        raise RuntimeError('VLM Run did not return a session_id for artifact retrieval')

    response_content = raw['choices'][0]['message']['content']
    parsed = json.loads(response_content)
    artifact_data = parsed['image']
    artifact_id = artifact_data['id'] if isinstance(artifact_data, dict) else artifact_data

    # Download and return image
    data = await _download_artifact(artifact_id, session_id, timeout=timeout)
    img = PIL.Image.open(io.BytesIO(data))
    img.load()
    return img


@pxt.udf(resource_pool='request-rate:vlmrun')
async def annotate_image(
    prompt: str,
    *,
    file_path: str,
    model: str = 'vlmrun-orion-1:auto',
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> PIL.Image.Image:
    """
    Annotates an image with bounding boxes, keypoints, or segmentation masks.

    Uses VLM Run's ``viz`` toolset to overlay visual annotations on an input
    image based on the text prompt.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - ``pip install vlmrun openai``

    Args:
        prompt: Text prompt describing what to annotate (e.g. ``'Draw bounding
            boxes around all people'``).
        file_path: Local path to the input image.  For Image columns, use
            ``.localpath``.
        model: The model to use.  Defaults to ``'vlmrun-orion-1:auto'``.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact.  Defaults to 600.

    Returns:
        The annotated image.

    Examples:
        Annotate objects in an image:

        >>> tbl.add_computed_column(annotated=vlmrun.annotate_image(
        ...     'Draw bounding boxes around all people',
        ...     file_path=tbl.image.localpath,
        ... ))
    """
    from pydantic import BaseModel as _BaseModel
    from vlmrun.types.refs import ImageRef  # type: ignore[import-untyped]

    # Build messages
    content: list[dict[str, Any]] = [
        {'type': 'text', 'text': prompt},
        {'type': 'input_file', 'file_path': file_path},
    ]
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'Always return an annotated image in your response.'},
        {'role': 'user', 'content': content},
    ]

    # Build kwargs
    kwargs = dict(model_kwargs) if model_kwargs else {}
    schema = type('_Out', (_BaseModel,), {'__annotations__': {'image': ImageRef}}).model_json_schema()
    kwargs['response_format'] = {'type': 'json_schema', 'schema': schema}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = ['viz']
    kwargs['extra_body'] = extra_body

    # Upload files and call API
    processed = await _upload_files(messages)
    result = await _vlmrun_client().chat.completions.with_raw_response.create(messages=processed, model=model, **kwargs)
    raw = json.loads(result.text)

    # Extract session_id (ephemeral) and artifact ID
    session_id = raw.get('session_id')
    if not session_id:
        raise RuntimeError('VLM Run did not return a session_id for artifact retrieval')

    response_content = raw['choices'][0]['message']['content']
    parsed = json.loads(response_content)
    artifact_data = parsed['image']
    artifact_id = artifact_data['id'] if isinstance(artifact_data, dict) else artifact_data

    # Download and return image
    data = await _download_artifact(artifact_id, session_id, timeout=timeout)
    img = PIL.Image.open(io.BytesIO(data))
    img.load()
    return img


@pxt.udf(resource_pool='request-rate:vlmrun')
async def generate_video(
    prompt: str,
    *,
    file_path: str | None = None,
    model: str = 'vlmrun-orion-1:auto',
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> pxt.Video:
    """
    Generates a video from a text prompt using VLM Run.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - ``pip install vlmrun openai``

    Args:
        prompt: Text prompt describing the video to generate.
        file_path: Optional local path to an input file.  For Image/Video
            columns, use ``.localpath``.
        model: The model to use.  Defaults to ``'vlmrun-orion-1:auto'``.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact.  Defaults to 600.

    Returns:
        A video file.

    Examples:
        Add a computed column that generates videos from text prompts:

        >>> tbl.add_computed_column(video=vlmrun.generate_video(tbl.prompt))
    """
    from pydantic import BaseModel as _BaseModel
    from vlmrun.types.refs import VideoRef  # type: ignore[import-untyped]

    # Build messages
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    if file_path is not None:
        content.append({'type': 'input_file', 'file_path': file_path})
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'Always return a generated video in your response.'},
        {'role': 'user', 'content': content},
    ]

    # Build kwargs
    kwargs = dict(model_kwargs) if model_kwargs else {}
    schema = type('_Out', (_BaseModel,), {'__annotations__': {'video': VideoRef}}).model_json_schema()
    kwargs['response_format'] = {'type': 'json_schema', 'schema': schema}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = ['video']
    kwargs['extra_body'] = extra_body

    # Upload files and call API
    processed = await _upload_files(messages)
    result = await _vlmrun_client().chat.completions.with_raw_response.create(messages=processed, model=model, **kwargs)
    raw = json.loads(result.text)

    # Extract session_id (ephemeral) and artifact ID
    session_id = raw.get('session_id')
    if not session_id:
        raise RuntimeError('VLM Run did not return a session_id for artifact retrieval')

    response_content = raw['choices'][0]['message']['content']
    parsed = json.loads(response_content)
    artifact_data = parsed['video']
    artifact_id = artifact_data['id'] if isinstance(artifact_data, dict) else artifact_data

    # Download and write video (videos are generated asynchronously, so retry)
    data = await _download_artifact(artifact_id, session_id, poll_interval=10.0, timeout=timeout)
    path = TempStore.create_path(extension='.mp4')
    path.write_bytes(data)
    return str(path)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
