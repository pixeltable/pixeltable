"""
Pixeltable UDFs that wrap the VLM Run API.

In order to use them, you must first `pip install vlmrun openai`
and configure your VLM Run API key as described in the
[VLM Run documentation](https://docs.vlm.run/).
"""

import asyncio
import copy
import functools
import io
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    import openai

_logger = logging.getLogger(__name__)

_DEFAULT_MODEL = 'vlmrun-orion-2:auto'

# ---------------------------------------------------------------------------
# Client registration
# ---------------------------------------------------------------------------


@register_client('vlmrun', credential_param='api_key')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://agent.vlm.run/v1/openai',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _vlmrun_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('vlmrun')


# Lazy-initialized sync `VLMRun` client. The OpenAI-compatible async client above only covers
# chat completions; file uploads go through the VLM Run Files API, which is only exposed by the
# `vlmrun` SDK's synchronous client (wrapped in `asyncio.to_thread` at the call sites).
_sync_client_lock = threading.Lock()
_sync_client_cache: dict[str, Any] = {}


def _vlmrun_sync_client() -> Any:
    """Return a cached synchronous ``VLMRun`` client (lazy, thread-safe)."""
    if 'client' in _sync_client_cache:
        return _sync_client_cache['client']
    with _sync_client_lock:
        if 'client' in _sync_client_cache:
            return _sync_client_cache['client']
        Env.get().require_package('vlmrun')
        from vlmrun.client import VLMRun  # type: ignore[import-untyped]

        api_key = _vlmrun_client().api_key
        client = VLMRun(api_key=api_key, base_url='https://agent.vlm.run/v1')
        _sync_client_cache['client'] = client
        return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Media formats accepted by the Orion chat completions API (audio is not supported by Orion).
_SUPPORTED_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.bmp',
    '.tiff',
    '.webp',  # images
    '.mp4',
    '.mov',
    '.avi',
    '.mkv',
    '.webm',  # videos
    '.pdf',
    '.doc',
    '.docx',  # documents
}

# Content item types whose value may be a Pixeltable media column reference.
_MEDIA_ITEM_TYPES = ('image_url', 'video_url', 'file_url')


async def _upload_file(file_path: str | Path) -> str:
    """Upload a local file to VLM Run and return its file id.

    The file id is used only for the duration of the API call and is never stored in table state.
    The SDK deduplicates uploads by MD5, so re-uploading the same file on recompute is cheap.
    """
    fp = Path(file_path)
    ext = fp.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'Unsupported file format: {ext}. Supported formats: {", ".join(sorted(_SUPPORTED_EXTENSIONS))}',
        )
    try:
        uploaded = await asyncio.to_thread(lambda: _vlmrun_sync_client().files.upload(file=fp))
    except Exception as exc:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR, f'File upload to VLM Run failed for {fp.name}: {exc}', provider='vlmrun'
        ) from exc
    return str(uploaded.id)


async def _upload_image(image: PIL.Image.Image) -> str:
    """Write a PIL image to a temporary file and upload it, returning its file id."""
    path = TempStore.create_path(extension='.png')

    def _save() -> None:
        image.save(path, format='PNG')

    await asyncio.to_thread(_save)
    try:
        return await _upload_file(path)
    finally:
        path.unlink(missing_ok=True)


async def _resolve_messages(messages: list) -> list:
    """Resolve Pixeltable media references in *messages* to VLM Run ``input_file`` items.

    Media columns embedded in message content arrive as `PIL.Image.Image` objects (image columns)
    or local path strings (video/document columns). Each is uploaded ephemerally and replaced with
    ``{'type': 'input_file', 'file_id': ...}``. HTTP(S) URLs are passed through unchanged.
    """
    processed = copy.deepcopy(messages)
    for message in processed:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            if item_type == 'input_file':
                if 'file_id' in item:
                    continue  # already an uploaded file reference
                file_path = item.pop('file_path', None)
                if file_path is None:
                    raise pxt.RequestError(
                        pxt.ErrorCode.INVALID_ARGUMENT,
                        "'input_file' content items must contain a 'file_id' or 'file_path'",
                    )
                item['file_id'] = await _upload_file(file_path)
                continue
            if item_type not in _MEDIA_ITEM_TYPES:
                continue
            value = item.get(item_type)
            if isinstance(value, PIL.Image.Image):
                # resolved pxt.Image column reference
                item.clear()
                item['type'] = 'input_file'
                item['file_id'] = await _upload_image(value)
            elif isinstance(value, str) and not value.startswith(('http://', 'https://', 'data:')):
                # resolved pxt.Video / pxt.Document column reference (a local file path)
                item.clear()
                item['type'] = 'input_file'
                item['file_id'] = await _upload_file(value)
            elif isinstance(value, str):
                item[item_type] = {'url': value}
            # dicts ({'url': ...}) pass through unchanged
    return processed


@functools.lru_cache(maxsize=None)
def _image_response_schema() -> str:
    """JSON schema (serialized) for responses containing a single generated image artifact."""
    from pydantic import BaseModel
    from vlmrun.types.refs import ImageRef  # type: ignore[import-untyped]

    class _ImageOut(BaseModel):
        image: ImageRef

    return json.dumps(_ImageOut.model_json_schema())


@functools.lru_cache(maxsize=None)
def _video_response_schema() -> str:
    """JSON schema (serialized) for responses containing a video URL.

    Uses a plain URL field instead of ``VideoRef`` so the API returns a direct download
    link rather than an artifact id that requires polling.
    """
    from pydantic import BaseModel, Field

    class _VideoOut(BaseModel):
        url: str = Field(..., description='The URL of the generated video')

    return json.dumps(_VideoOut.model_json_schema())


@functools.lru_cache(maxsize=None)
def _document_response_schema() -> str:
    """JSON schema (serialized) for responses containing a single generated document artifact."""
    from pydantic import BaseModel
    from vlmrun.types.refs import DocumentRef

    class _DocumentOut(BaseModel):
        document: DocumentRef

    return json.dumps(_DocumentOut.model_json_schema())


def _classify_media_ref(media_ref: str) -> tuple[str, str]:
    """Classify a media reference from a structured response as an artifact id or a direct URL.

    The API may return a direct URL or an artifact reference of the form ``<type>_<6-hex-chars>``,
    optionally with a file extension appended (e.g. ``vid_a1b2c3``, ``doc_a1b2c3.pdf``, ``url_f41446``).
    Returns ``('artifact', <bare id>)`` or ``('url', <url>)``.
    """
    ref = media_ref.strip()
    m = re.fullmatch(r'([a-z]+_[0-9a-f]{6})(?:\.\w+)?', ref)
    if m is not None:
        return 'artifact', m.group(1)
    if ref.startswith(('http://', 'https://')):
        return 'url', ref
    raise pxt.ExternalServiceError(
        pxt.ErrorCode.PROVIDER_ERROR,
        f'VLM Run returned an unrecognized media reference: {media_ref!r}',
        provider='vlmrun',
    )


def _parse_artifact_ref(raw: dict[str, Any], key: str) -> str:
    """Extract an artifact reference from a structured chat response, validating the shape."""
    try:
        response_content = raw['choices'][0]['message']['content']
        parsed = json.loads(response_content)
        artifact_data = parsed[key]
        return artifact_data['id'] if isinstance(artifact_data, dict) else str(artifact_data)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            f'VLM Run returned an unexpected response shape: {exc}\n{raw}',
            provider='vlmrun',
        ) from exc


async def _download_artifact(object_id: str, session_id: str, *, timeout: float = 600.0) -> bytes:
    """Download an artifact's raw bytes, polling with exponential backoff until ready or *timeout*.

    Fetches the ``/artifacts`` endpoint directly rather than using the SDK's
    ``artifacts.get()``. The SDK asserts the response ``Content-Type`` matches the
    artifact type and raises when VLM Run serves a mismatched type (e.g. an
    ``img_`` artifact sent as ``application/octet-stream``); fetching raw bytes
    avoids that crash and lets us do our own readiness polling.
    """
    import requests

    def _fetch() -> bytes:
        client = _vlmrun_sync_client()
        url = f'{client.base_url}/artifacts'
        headers = {'Authorization': f'Bearer {client.api_key}'}
        params = {'object_id': object_id, 'session_id': session_id}
        deadline = time.monotonic() + timeout
        delay = 0.25
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=120)
                if resp.status_code == 200:
                    data = resp.content
                    # For url_ artifacts the body is a signed URL — follow it
                    if data.startswith(b'http'):
                        actual = requests.get(data.decode('utf-8').strip(), timeout=120)
                        actual.raise_for_status()
                        return actual.content
                    return data
                if resp.status_code in (401, 403):
                    # auth failures won't resolve by polling
                    raise pxt.ExternalServiceError(
                        pxt.ErrorCode.PROVIDER_ERROR,
                        f'Artifact request for {object_id} failed with status {resp.status_code}: {resp.text}',
                        provider='vlmrun',
                        status_code=resp.status_code,
                    )
            except requests.RequestException as exc:
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_ERROR, f'Artifact download failed for {object_id}: {exc}', provider='vlmrun'
                ) from exc
            if time.monotonic() >= deadline:
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_TIMEOUT,
                    f'Artifact {object_id} not ready after {timeout}s ({attempt} attempts, status {resp.status_code})',
                    provider='vlmrun',
                    status_code=resp.status_code,
                )
            _logger.debug(
                'Artifact %s attempt %d returned status %d, retrying in %.2fs',
                object_id,
                attempt,
                resp.status_code,
                delay,
            )
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 16.0)

    return await asyncio.to_thread(_fetch)


async def _poll_redirect(location: str, *, timeout: float) -> dict[str, Any]:
    """Poll a 303 redirect target until the completion result is ready.

    VLM Run's chat endpoint responds with `303 See Other` when a request outlives the
    gateway's HTTP window (~150s); the Location URL serves the result once the job finishes,
    re-redirecting or returning 202 until then.
    """
    headers = {'Authorization': f'Bearer {_vlmrun_client().api_key}'}
    url = str(httpx.URL('https://agent.vlm.run').join(location))
    deadline = time.monotonic() + timeout
    consecutive_5xx = 0
    async with httpx.AsyncClient(follow_redirects=False, timeout=120.0) as http:
        while True:
            try:
                resp = await http.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                # the result URL long-polls, holding the connection while the job runs;
                # a timeout just means "not done yet" — reconnect until the deadline
                if time.monotonic() >= deadline:
                    raise pxt.ExternalServiceError(
                        pxt.ErrorCode.PROVIDER_TIMEOUT,
                        f'Long-running request did not complete within {timeout}s',
                        provider='vlmrun',
                    ) from None
                continue
            if resp.status_code in (200, 201) and resp.content:  # completion body is ready
                return resp.json()
            if resp.status_code in (202, 204, 303) or resp.status_code >= 500:
                # 202/204/303: result not ready yet (204 = No Content is returned repeatedly
                # while the job runs, and an empty body would break resp.json()). 5xx: the
                # polling endpoint intermittently errors while the job is still running;
                # tolerate a bounded number in a row.
                if resp.status_code >= 500:
                    consecutive_5xx += 1
                    if consecutive_5xx > 5:
                        raise pxt.ExternalServiceError(
                            pxt.ErrorCode.PROVIDER_ERROR,
                            f'Polling long-running request failed with status {resp.status_code}: {resp.text[:500]}',
                            provider='vlmrun',
                            status_code=resp.status_code,
                        )
                else:
                    consecutive_5xx = 0
                    if resp.status_code == 303:
                        url = str(httpx.URL(url).join(resp.headers.get('location', url)))
                if time.monotonic() >= deadline:
                    raise pxt.ExternalServiceError(
                        pxt.ErrorCode.PROVIDER_TIMEOUT,
                        f'Long-running request did not complete within {timeout}s',
                        provider='vlmrun',
                        status_code=resp.status_code,
                    )
                await asyncio.sleep(2.0)
                continue
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR,
                f'Polling long-running request failed with status {resp.status_code}: {resp.text[:500]}',
                provider='vlmrun',
                status_code=resp.status_code,
            )


async def _chat_create(messages: list, model: str, kwargs: dict[str, Any], *, timeout: float = 600.0) -> dict[str, Any]:
    """Call the chat completions endpoint, transparently handling long-running-request redirects."""
    import openai

    try:
        result = await _vlmrun_client().chat.completions.with_raw_response.create(
            messages=messages, model=model, **kwargs
        )
        return json.loads(result.text)
    except openai.APIStatusError as exc:
        location = exc.response.headers.get('location')
        if exc.response.status_code == 303 and location is not None:
            return await _poll_redirect(location, timeout=timeout)
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            f'VLM Run API request failed with status {exc.response.status_code}: {exc.message}',
            provider='vlmrun',
            status_code=exc.response.status_code,
        ) from exc
    except openai.APITimeoutError as exc:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_TIMEOUT, f'VLM Run API request timed out: {exc}', provider='vlmrun'
        ) from exc
    except openai.APIError as exc:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR, f'VLM Run API request failed: {exc}', provider='vlmrun'
        ) from exc


async def _generate_image_impl(
    prompt: str,
    image: PIL.Image.Image | None,
    *,
    toolset: str,
    system_message: str,
    model: str,
    model_kwargs: dict[str, Any] | None,
    timeout: float,
) -> PIL.Image.Image:
    """Shared implementation for image generation/annotation: call API, download the image artifact."""
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    if image is not None:
        content.append({'type': 'input_file', 'file_id': await _upload_image(image)})
    messages: list = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': content}]

    kwargs = dict(model_kwargs) if model_kwargs else {}
    kwargs['response_format'] = {'type': 'json_schema', 'schema': json.loads(_image_response_schema())}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = [toolset]
    kwargs['extra_body'] = extra_body

    raw = await _chat_create(messages, model, kwargs, timeout=timeout)

    session_id = raw.get('session_id')
    if not session_id:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            'VLM Run did not return a session_id for artifact retrieval',
            provider='vlmrun',
        )
    artifact_id = _parse_artifact_ref(raw, 'image')

    data = await _download_artifact(artifact_id, session_id, timeout=timeout)
    img = PIL.Image.open(io.BytesIO(data))
    img.load()
    return img


# ---------------------------------------------------------------------------
# UDFs
# ---------------------------------------------------------------------------


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def chat_completions(
    messages: list, *, model: str = _DEFAULT_MODEL, model_kwargs: dict[str, Any] | None = None
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the VLM Run chat completions API endpoint.
    For additional details, see: <https://docs.vlm.run/>

    Pixeltable media columns can be referenced directly in message content items:
    image columns via `{'type': 'image_url', 'image_url': tbl.img}`, video columns via
    `{'type': 'video_url', 'video_url': tbl.video}`, and document columns via
    `{'type': 'file_url', 'file_url': tbl.doc}`. Referenced media is uploaded to VLM Run
    automatically before the API call.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun openai`

    Args:
        messages: A list of messages for the chat conversation. Media content items may
            reference Pixeltable columns directly (see above).
        model: The model to use. Options include the Orion 2 tiers `'vlmrun-orion-2:fast'`,
            `'vlmrun-orion-2:auto'`, and `'vlmrun-orion-2:pro'`, as well as specific
            backbones (e.g. Qwen 3.6, Gemma 4, GPT-5.5, Claude Opus 4.8, Kimi 2.6); see
            <https://docs.vlm.run/> for the full list of model variants.
        model_kwargs: Additional keyword args for the VLM Run API, such as `response_format`
            (for structured JSON output) and `extra_body` (e.g. `{'toolsets': [...]}`).

    Returns:
        A dictionary containing the API response.

    Examples:
        Describe an image column `img` of the table `tbl`:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'type': 'text', 'text': 'Describe this image'},
        ...             {'type': 'image_url', 'image_url': tbl.img},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(response=vlmrun.chat_completions(messages))

        Extract structured data from a document column `doc` into a JSON column:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {
        ...                 'type': 'text',
        ...                 'text': 'Extract the invoice number and total',
        ...             },
        ...             {'type': 'file_url', 'file_url': tbl.doc},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(
        ...     invoice=vlmrun.chat_completions(
        ...         messages,
        ...         model_kwargs={'response_format': {'type': 'json_object'}},
        ...     )
        ... )
    """
    Env.get().require_package('vlmrun')
    Env.get().require_package('openai')
    processed = await _resolve_messages(messages)
    kwargs = dict(model_kwargs) if model_kwargs else {}
    raw = await _chat_create(processed, model, kwargs)
    if 'choices' not in raw:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_ERROR,
            f'VLM Run returned an unexpected response shape (no choices):\n{raw}',
            provider='vlmrun',
        )
    raw.pop('session_id', None)
    return raw


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_image(
    prompt: str,
    *,
    image: pxt.Image | None = None,
    model: str = _DEFAULT_MODEL,
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> PIL.Image.Image:
    """
    Generates or edits an image using VLM Run.

    When *image* is `None`, generates a new image from the text prompt.
    When *image* is provided, edits the given image according to the prompt.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun openai`

    Args:
        prompt: Text prompt describing the image to generate or the edit to apply.
        image: Optional input image to edit. Omit for text-to-image generation.
        model: The model to use. Defaults to `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact. Defaults to 600.

    Returns:
        The generated or edited image.

    Examples:
        Generate an image from text:

        >>> tbl.add_computed_column(image=vlmrun.generate_image(tbl.prompt))

        Edit an existing image column `img`:

        >>> tbl.add_computed_column(
        ...     edited=vlmrun.generate_image('Blur all faces', image=tbl.img)
        ... )
    """
    Env.get().require_package('vlmrun')
    Env.get().require_package('openai')
    return await _generate_image_impl(
        prompt,
        image,
        toolset='image-gen',
        system_message='Always return a generated image in your response.',
        model=model,
        model_kwargs=model_kwargs,
        timeout=timeout,
    )


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def annotate_image(
    prompt: str,
    image: pxt.Image,
    *,
    model: str = _DEFAULT_MODEL,
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> PIL.Image.Image:
    """
    Annotates an image with bounding boxes, keypoints, or segmentation masks.

    Uses VLM Run's `viz` toolset to overlay visual annotations on an input
    image based on the text prompt.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun openai`

    Args:
        prompt: Text prompt describing what to annotate (e.g. `'Draw bounding
            boxes around all people'`).
        image: The input image to annotate.
        model: The model to use. Defaults to `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact. Defaults to 600.

    Returns:
        The annotated image.

    Examples:
        Annotate objects in an image column `img`:

        >>> tbl.add_computed_column(
        ...     annotated=vlmrun.annotate_image(
        ...         'Draw bounding boxes around all people', tbl.img
        ...     )
        ... )
    """
    Env.get().require_package('vlmrun')
    Env.get().require_package('openai')
    return await _generate_image_impl(
        prompt,
        image,
        toolset='viz',
        system_message='Always return an annotated image in your response.',
        model=model,
        model_kwargs=model_kwargs,
        timeout=timeout,
    )


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_video(
    prompt: str,
    *,
    image: pxt.Image | None = None,
    video: pxt.Video | None = None,
    model: str = _DEFAULT_MODEL,
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> pxt.Video:
    """
    Generates a video from a text prompt using VLM Run.

    An input image or video can optionally be provided as a starting point for generation.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun openai`

    Args:
        prompt: Text prompt describing the video to generate.
        image: Optional input image to use as a starting point.
        video: Optional input video to use as a starting point.
        model: The model to use. Defaults to `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact. Defaults to 600.

    Returns:
        A video file.

    Examples:
        Add a computed column that generates videos from text prompts:

        >>> tbl.add_computed_column(video=vlmrun.generate_video(tbl.prompt))
    """
    Env.get().require_package('vlmrun')
    Env.get().require_package('openai')
    import requests

    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    if image is not None:
        content.append({'type': 'input_file', 'file_id': await _upload_image(image)})
    if video is not None:
        content.append({'type': 'input_file', 'file_id': await _upload_file(video)})
    messages: list = [
        {'role': 'system', 'content': 'Always return a generated video in your response.'},
        {'role': 'user', 'content': content},
    ]

    kwargs = dict(model_kwargs) if model_kwargs else {}
    kwargs['response_format'] = {'type': 'json_schema', 'schema': json.loads(_video_response_schema())}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = ['video']
    kwargs['extra_body'] = extra_body

    raw = await _chat_create(messages, model, kwargs, timeout=timeout)

    session_id = raw.get('session_id')
    video_ref = _parse_artifact_ref(raw, 'url')
    ref_kind, ref = _classify_media_ref(video_ref)

    if ref_kind == 'artifact':
        # Artifact reference — resolve via the artifacts endpoint
        if not session_id:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR,
                'VLM Run did not return a session_id for artifact retrieval',
                provider='vlmrun',
            )
        data = await _download_artifact(ref, session_id, timeout=timeout)
    else:
        # Direct URL — download the video
        def _download_video() -> bytes:
            resp = requests.get(ref, timeout=timeout)
            resp.raise_for_status()
            return resp.content

        data = await asyncio.to_thread(_download_video)
    path = TempStore.create_path(extension='.mp4')
    path.write_bytes(data)
    return str(path)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_document(
    prompt: str,
    document: pxt.Document,
    *,
    model: str = _DEFAULT_MODEL,
    model_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> pxt.Document:
    """
    Transforms a document and returns the edited document (e.g. redaction).

    Uses VLM Run's `document` toolset to produce a new document from an input
    document according to the text prompt — for example, redacting PII or
    removing specific content.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun openai`

    Args:
        prompt: Text prompt describing the transformation (e.g. `'Redact all
            personally identifiable information'`).
        document: The input document to transform.
        model: The model to use. Defaults to `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional keyword args for the VLM Run API.
        timeout: Maximum seconds to wait for the artifact. Defaults to 600.

    Returns:
        The transformed document.

    Examples:
        Redact PII from a document column `doc`:

        >>> tbl.add_computed_column(
        ...     redacted=vlmrun.generate_document(
        ...         'Redact all names and email addresses', tbl.doc
        ...     )
        ... )
    """
    Env.get().require_package('vlmrun')
    Env.get().require_package('openai')
    import requests

    content: list[dict[str, Any]] = [
        {'type': 'text', 'text': prompt},
        {'type': 'input_file', 'file_id': await _upload_file(document)},
    ]
    messages: list = [
        {'role': 'system', 'content': 'Always return a generated document in your response.'},
        {'role': 'user', 'content': content},
    ]

    kwargs = dict(model_kwargs) if model_kwargs else {}
    kwargs['response_format'] = {'type': 'json_schema', 'schema': json.loads(_document_response_schema())}
    extra_body = kwargs.pop('extra_body', {})
    extra_body['toolsets'] = ['document']
    kwargs['extra_body'] = extra_body

    raw = await _chat_create(messages, model, kwargs, timeout=timeout)

    session_id = raw.get('session_id')
    doc_ref = _parse_artifact_ref(raw, 'document')
    ref_kind, ref = _classify_media_ref(doc_ref)

    if ref_kind == 'artifact':
        # Artifact reference — resolve via the artifacts endpoint
        if not session_id:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR,
                'VLM Run did not return a session_id for artifact retrieval',
                provider='vlmrun',
            )
        data = await _download_artifact(ref, session_id, timeout=timeout)
    else:
        # Direct URL — download the document
        def _download_doc() -> bytes:
            resp = requests.get(ref, timeout=timeout)
            resp.raise_for_status()
            return resp.content

        data = await asyncio.to_thread(_download_doc)
    path = TempStore.create_path(extension='.pdf')
    path.write_bytes(data)
    return str(path)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
