"""
Pixeltable UDFs that wrap the VLM Run Gateway API.

The VLM Run Gateway is an OpenAI-compatible API in front of vision and speech models for OCR, visual question
answering, detection, segmentation, pose estimation, embeddings and speech-to-text. In order to use it, you must
first `pip install openai` and configure your VLM Run API key, either as the `VLMRUN_API_KEY` environment variable
or as `api_key` in the `vlmrun` section of the Pixeltable config file. For details, see the
[VLM Run Gateway documentation](https://docs.vlm.run/gateway/introduction).
"""

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import pathlib
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env, register_client
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.documents import get_document_handle
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import openai

_logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = 'https://gateway.vlm.run/v1/openai'
_PROVIDER = 'vlmrun'

# Longer than the Gateway's 600s default deadline per model, so that the Gateway, not the client, reports a timeout.
# Uploads get the same allowance: sending a 244 MB file takes about 85s.
_TIMEOUT_S = 660.0

_MediaKind = Literal['image', 'document', 'video', 'audio']


class TranscriptionWord(TypedDict):
    word: str
    start: float
    end: float


class TranscriptionSegment(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionResponse(TypedDict, total=False):
    # 'json' and 'verbose_json' return all of these except possibly segments and words; 'text', 'srt' and 'vtt'
    # return only text.
    text: str
    duration: float
    language: str | None
    segments: list[TranscriptionSegment] | None
    words: list[TranscriptionWord] | None
    # usage.cost is the cost of the call in USD
    usage: dict | None
    model: str


_EMBEDDING_DIMENSIONS = {'qwen/qwen3-vl-embedding-2b': 2048, 'google/embeddinggemma-2': 768}


# The key is read from the `vlmrun` section, so one VLM Run key serves every VLM Run integration. The client keeps its
# own name because its `base_url` and `rate_limit` belong to the Gateway: `VLMRUN_BASE_URL` is the vlmrun SDK's
# setting for api.vlm.run.
@register_client('vlmrun_gateway', credential_param=None)
def _(base_url: str | None = None) -> 'openai.AsyncOpenAI':
    Env.get().require_package('openai', [3])

    import httpx2
    import openai

    api_key = Config.get().get_string_value('api_key', section='vlmrun')
    if api_key is None:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            '`vlmrun_gateway` client not initialized: no VLM Run API key is configured.\n'
            'To fix this, set the `VLMRUN_API_KEY` environment variable, or put `api_key` in the `vlmrun` section '
            'of $PIXELTABLE_HOME/config.toml.',
        )
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or _DEFAULT_BASE_URL,
        # Pixeltable's scheduler retries rate limits and transient errors; SDK retries on top of it would multiply
        # the attempts on a slow model (600s each by default).
        max_retries=0,
        # No connect or pool timeout: a request can wait in the connection queue for a long time, and httpx starts the
        # connect clock when the request enters the queue.
        http_client=httpx2.AsyncClient(
            timeout=httpx2.Timeout(connect=None, read=_TIMEOUT_S, write=_TIMEOUT_S, pool=None)
        ),
    )


def _client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('vlmrun_gateway')


_PNG_MODES = frozenset({'1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I;16'})


def _image_url(image: PIL.Image.Image) -> str:
    if image.mode not in _PNG_MODES:
        image = image.convert('RGBA' if image.has_transparency_data else 'RGB')
    return f'data:image/png;base64,{to_base64(image, format="png")}'


# The MIME type to send when a file's name doesn't give one, such as media cached from a URL without an extension.
# The Gateway decodes the bytes itself; the type only needs to name the right kind of media.
_DEFAULT_MIME: dict[str, str] = {
    'image': 'image/png',
    'document': 'application/pdf',
    'video': 'video/mp4',
    'audio': 'audio/mpeg',
}


def _file_mime(path: str, kind: _MediaKind) -> str:
    if kind == 'document':
        handle = get_document_handle(path)
        handle.close()
        if handle.format != ts.DocumentType.DocumentFormat.PDF:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'The VLM Run Gateway reads PDF documents only; {pathlib.Path(path).name!r} is {handle.format.name}.',
            )
        return 'application/pdf'
    mime, _ = mimetypes.guess_type(path)
    return mime if mime is not None and mime.startswith(f'{kind}/') else _DEFAULT_MIME[kind]


def _file_data_uri(path: str, kind: _MediaKind) -> str:
    mime = _file_mime(path, kind)
    return f'data:{mime};base64,{base64.b64encode(pathlib.Path(path).read_bytes()).decode("ascii")}'


_MEDIA_PART_TYPES = frozenset({'image_url', 'document_url', 'file_url', 'video_url', 'audio_url'})


def _media_url(value: Any, part_type: str) -> Any:
    """Convert the value of a media content part into the Gateway's `{'url': ...}` form.

    Only image columns (which arrive as PIL images) are encoded here. Strings must be http(s) or `data:` URLs: inside
    `messages`, a document or video column's local path can't be told apart from a string column, and reading it
    would upload any local file a string names.
    """
    if isinstance(value, dict):
        return {**value, **_media_url(value['url'], part_type)} if 'url' in value else value
    if isinstance(value, PIL.Image.Image):
        return {'url': _image_url(value)}
    if isinstance(value, str):
        if not value[:8].lower().startswith(('http://', 'https://', 'data:')):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'A `{part_type}` content part must be an image column, or an http(s) or data: URL. For a media '
                'column inserted from a public URL, pass its `.fileurl`; to read a local document or video, use '
                '`ocr()` or `analyze()`.',
            )
        return {'url': value}
    raise excs.RequestError(
        excs.ErrorCode.INVALID_ARGUMENT, f'Unsupported value for a `{part_type}` content part: {type(value).__name__}'
    )


def _resolve_messages(messages: list) -> list:
    """Return a copy of `messages` with image values encoded as data URIs.

    Returns a copy: built-in UDFs must not modify their arguments.
    """
    resolved = copy.deepcopy(messages)
    for message in resolved:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get('type') in _MEDIA_PART_TYPES and part['type'] in part:
                part[part['type']] = _media_url(part[part['type']], part['type'])
    return resolved


def _reject_reserved(
    kwargs: dict[str, Any], keys: tuple[str, ...], hint: str = '', *, args: dict[str, str] | None = None
) -> None:
    """Reject `model_kwargs` keys that the function sets; `args` maps keys that come from a UDF argument to it."""
    args = {'model': 'model', **(args or {})}
    # model_kwargs['extra_body'] is merged over the request too, so it can't contain these keys either.
    nested = kwargs.get('extra_body')
    for key in keys:
        if key in kwargs or (isinstance(nested, dict) and key in nested):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'`{key}` cannot be set in `model_kwargs`; pass it as the `{args[key]}` argument.'
                if key in args
                else f'`{key}` cannot be set in `model_kwargs`; this function sets it.{hint}',
            )


def _body(model_kwargs: dict[str, Any] | None, **extra: Any) -> dict[str, Any] | None:
    """Return the `extra_body` for a JSON request.

    `model_kwargs` go in `extra_body` because the OpenAI SDK rejects keyword arguments it doesn't know, and the
    Gateway accepts many (`method`, `document_dpi`, sampling fields such as `top_k`). The SDK merges `extra_body`
    into the request JSON, so known fields such as `max_tokens` work the same way. `extra` contains the fields that
    the UDF's own arguments set; they win over `model_kwargs`.
    """
    body = dict(model_kwargs or {})
    body.update(body.pop('extra_body', None) or {})
    body.update(extra)
    return body or None


def _error_details(exc: 'openai.APIStatusError') -> tuple[str, str | None, str | None]:
    """Return (message, code, param) from a Gateway error body.

    The Gateway wraps most errors as `{"detail": {"error": {...}}}`, which the OpenAI SDK doesn't recognize: it leaves
    `exc.code` empty, and recent SDK versions drop the message from `str(exc)` too.
    """
    body = exc.body
    err: Any = None
    if isinstance(body, dict):
        detail = body.get('detail', body)
        err = detail.get('error', detail) if isinstance(detail, dict) else detail
    if isinstance(err, dict):
        return str(err.get('message') or exc.message), err.get('code'), err.get('param')
    if isinstance(err, str):
        return err, None, None
    text = exc.response.text
    # Under load, the Gateway's load balancer answers with an HTML error page, which doesn't belong in an error message.
    return (exc.message if '<html' in text[:200].lower() else text[:500] or exc.message), None, None


def _retry_after(exc: 'openai.APIStatusError') -> float | None:
    try:
        return float(exc.response.headers['retry-after'])
    except (KeyError, ValueError):
        return None


def _to_pxt_error(exc: 'openai.APIStatusError', message: str, code: str | None, param: str | None) -> excs.Error:
    """Translate a Gateway error into an `ExternalServiceError` that keeps the Gateway's message and request id.

    The error code decides whether Pixeltable's scheduler retries: it retries `RATE_LIMITED` and
    `PROVIDER_OVERLOADED`, and `PROVIDER_ERROR` when the chained SDK error has a retryable status.
    """
    status = exc.status_code
    request_id = exc.response.headers.get('x-request-id', '').split(',')[0].strip()
    error_code: excs.ErrorCode
    if status == 429:
        error_code = excs.ErrorCode.RATE_LIMITED
    elif status == 503:
        error_code = excs.ErrorCode.PROVIDER_OVERLOADED
    elif status == 504 and code == 'inference_timeout':
        # The run used up the model's whole deadline. The Gateway docs suggest a smaller input or streaming, not
        # resending the same request.
        error_code = excs.ErrorCode.PROVIDER_TIMEOUT
        message = (
            message.rstrip('. ')
            + '. Try a smaller input, such as fewer PDF pages (`pages` in `ocr()`) or a shorter video.'
        )
    elif status in (401, 402, 403):
        # Pixeltable has no billing error code; an account problem that a retry can't fix is the closest.
        error_code = excs.ErrorCode.PROVIDER_AUTH_ERROR
    elif status in (408, 504):
        error_code = excs.ErrorCode.PROVIDER_ERROR
    elif 400 <= status < 500:
        error_code = excs.ErrorCode.PROVIDER_BAD_REQUEST
    else:
        error_code = excs.ErrorCode.PROVIDER_ERROR
    label = ', '.join(str(x) for x in (status, code, f'param={param}' if param else None) if x)
    suffix = f' (request id: {request_id})' if request_id else ''
    return excs.ExternalServiceError(
        error_code,
        f'VLM Run Gateway error ({label}): {message}{suffix}',
        retry_after=_retry_after(exc),
        provider=_PROVIDER,
        status_code=status,
        provider_error_code=code,
    )


async def _call(fn: Any, **kwargs: Any) -> Any:
    """Call an OpenAI SDK method, translating Gateway errors into Pixeltable errors.

    The Gateway asks clients to retry 500s and transient 502s (`model_error`), which the scheduler doesn't retry, so
    those get one retry here. So does a 502 without a Gateway error code, which the front end returns under load.
    """
    import openai

    for attempt in range(2):
        try:
            return await fn(**kwargs)
        except openai.APITimeoutError as exc:
            # The Gateway gives up on a model after its own deadline; past ours, a retry would most likely hang again.
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_TIMEOUT,
                f'The VLM Run Gateway request timed out (no response, or no upload progress, within '
                f'{_TIMEOUT_S:.0f}s).',
                provider=_PROVIDER,
            ) from exc
        except openai.APIConnectionError as exc:
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR, f'Could not reach the VLM Run Gateway: {exc}', provider=_PROVIDER
            ) from exc
        except openai.APIStatusError as exc:
            message, code, param = _error_details(exc)
            transient = exc.status_code == 500 or (
                exc.status_code == 502 and (code is None or code == 'model_error' or 'retry' in message.lower())
            )
            if transient and attempt == 0:
                _logger.debug(f'VLM Run Gateway returned {exc.status_code} ({code}); retrying once')
                await asyncio.sleep(2.0)
                continue
            raise _to_pxt_error(exc, message, code, param) from exc
    raise AssertionError('unreachable')


def _parse_json_content(response: Any) -> dict:
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run Gateway returned a non-JSON reply in JSON mode: {str(content)[:200]!r}',
            provider=_PROVIDER,
        ) from exc
    if not isinstance(result, dict):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run Gateway returned a JSON {type(result).__name__} instead of an object: {str(content)[:200]!r}',
            provider=_PROVIDER,
        )
    failed = [p.get('page_id') for p in result.get('pages') or [] if isinstance(p, dict) and p.get('status') == 'error']
    if failed:
        _logger.warning(f'VLM Run Gateway could not read page(s) {failed}; they are marked "status": "error".')
    return result


async def _structured_read(
    part: dict[str, Any], model: str, method: str | None, extra: dict[str, Any], model_kwargs: dict[str, Any] | None
) -> dict:
    """Run a JSON-mode read of a single media part and return the Gateway's JSON result."""
    _reject_reserved(
        model_kwargs or {},
        ('model', 'messages', 'response_format'),
        ' Use `chat_completions()` for full control of the request.',
    )
    if method is not None:
        extra['method'] = method
    response = await _call(
        _client().chat.completions.create,
        model=model,
        messages=[{'role': 'user', 'content': [part]}],
        response_format={'type': 'json_object'},
        extra_body=_body(model_kwargs, **extra),
    )
    return _parse_json_content(response)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun_gateway')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the VLM Run Gateway `chat/completions` API endpoint.
    For additional details, see: <https://docs.vlm.run/gateway/api-reference/post-chat-completions>

    Image columns can be referenced directly in message content parts, as `{'type': 'image_url', 'image_url': tbl.img}`;
    they are sent inline as base64. Document and video content parts (`document_url`, `video_url`) take an
    http(s) or `data:` URL, which the Gateway fetches itself; for media inserted from a public URL, that is
    `tbl.doc.fileurl`. To read a local document or video column, use `ocr()` or `analyze()`.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun_gateway`, key `rate_limit`). If no rate limit is
    configured, uses a default of 600 RPM, which is above the Gateway's limits of 240 requests per minute and
    10,000 per hour per account; set `rate_limit = 160` to stay under both.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages for the chat conversation. Media content parts may reference Pixeltable
            columns directly (see above).
        model: The model to use, such as `'qwen/qwen3.5-0.8b'`. For the available models, see
            <https://vlm.run/gateway/models>.
        model_kwargs: Additional keyword args for the Gateway `chat/completions` API, such as `response_format`,
            `max_tokens`, or Gateway fields like `method`, `method_params` and `document_dpi`. For details, see
            <https://docs.vlm.run/gateway/extra-kwargs>.
        tools: An optional list of Pixeltable tools to use for the request. Supported by some models only, such as
            `'qwen/qwen3.8-27b'` and the partner Gemini, Kimi, MiniMax and Muse models.
        tool_choice: An optional Pixeltable tool choice to use for the request.

    Returns:
        A dictionary containing the API response, including the cost of the call in USD in `usage.cost`.

    Examples:
        Ask a question about an image column `img` of the table `tbl`:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'type': 'text', 'text': 'What is in this image?'},
        ...             {'type': 'image_url', 'image_url': tbl.img},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(messages, model='qwen/qwen3.5-0.8b')
        ... )

        Extract fields from a receipt image as JSON that matches a schema:

        >>> receipt_messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {
        ...                 'type': 'text',
        ...                 'text': 'What is the total on this receipt?',
        ...             },
        ...             {'type': 'image_url', 'image_url': tbl.img},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(
        ...     receipt=chat_completions(
        ...         receipt_messages,
        ...         model='qwen/qwen3.8-27b',
        ...         model_kwargs={
        ...             'response_format': {
        ...                 'type': 'json_schema',
        ...                 'json_schema': {
        ...                     'name': 'receipt',
        ...                     'schema': {
        ...                         'type': 'object',
        ...                         'properties': {'total': {'type': 'number'}},
        ...                     },
        ...                 },
        ...             }
        ...         },
        ...     )
        ... )
    """
    _reject_reserved(model_kwargs or {}, ('model', 'messages'), args={'messages': 'messages'})
    kwargs: dict[str, Any] = {}
    if tools is not None:
        kwargs['tools'] = [{'type': 'function', 'function': tool} for tool in tools]
    if tool_choice is not None:
        if tool_choice['auto']:
            kwargs['tool_choice'] = 'auto'
        elif tool_choice['required']:
            kwargs['tool_choice'] = 'required'
        else:
            assert tool_choice['tool'] is not None
            kwargs['tool_choice'] = {'type': 'function', 'function': {'name': tool_choice['tool']}}
        if not tool_choice['parallel_tool_calls']:
            kwargs['parallel_tool_calls'] = False

    resolved = await asyncio.to_thread(_resolve_messages, messages)
    response = await _call(
        _client().chat.completions.create, model=model, messages=resolved, extra_body=_body(model_kwargs, **kwargs)
    )
    return response.model_dump(mode='json')


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun_gateway')
async def ocr(
    document: pxt.Document,
    *,
    model: str = 'paddlepaddle/paddleocr-vl-1.6',
    method: str | None = None,
    pages: list | None = None,
    dpi: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Reads the text and layout of a PDF document or an image with a VLM Run Gateway OCR model.

    Each PDF page is rasterized and read separately. For details, see
    <https://docs.vlm.run/gateway/guides/document-ocr>

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun_gateway`, key `rate_limit`). If no rate limit is
    configured, uses a default of 600 RPM, which is above the Gateway's limits of 240 requests per minute and
    10,000 per hour per account; set `rate_limit = 160` to stay under both.

    __Requirements:__

    - `pip install openai`

    Args:
        document: The PDF document to read. Other document formats raise an error. An image (`pxt.Image`) is
            also accepted.
        model: The OCR model to use, such as `'paddlepaddle/paddleocr-vl-1.6'` (general purpose),
            `'paddleocr/pp-ocrv6'` (text lines with bounding boxes) or `'nvidia/nemotron-3.5-parse'` (layout boxes,
            with `method='layout'`).
            For the available models, see <https://vlm.run/gateway/models>.
        method: The model entrypoint to run, such as `'markdown'`, `'ocr'` or `'layout'`. If omitted, the
            model's default method is used. For each model's methods, see <https://docs.vlm.run/gateway/methods>.
        pages: The zero-based pages to read, as page indices and `[start, stop]` ranges that exclude `stop`,
            for example `[0, [2, 4]]` for pages 0, 2 and 3. Selected pages are renumbered from 0. Not available for
            images.
        dpi: The resolution at which PDF pages are rasterized (Gateway default: 96). Raise it, for example to 150,
            for small text or dense tables. Not available for images.
        model_kwargs: Additional Gateway fields, such as `method_params` or `precision`. For details, see
            <https://docs.vlm.run/gateway/extra-kwargs>.

    Returns:
        The Gateway's JSON result, unchanged. Its shape depends on the input and the method:

        - For a PDF, `document_npages` is the number of pages read and `pages` is a list of page records, each
            with `page_id`, `page_width`, `page_height` and `content`, where `content` is
            `{'object': 'doc.page.blocks', 'items': [...]}`. A page the Gateway could not read has
            `'status': 'error'` and no `content`; the other pages are still returned.
        - For an image, `content` is a string for text methods such as `'markdown'`, or
            `{'object': ..., 'items': [...]}` for region methods such as `'ocr'` on `'paddleocr/pp-ocrv6'`.
        - `'vlm-run/ocr:auto'` with `method='markdown'` returns the whole PDF as one `content` string.

    Examples:
        Read each page of a PDF column `doc` of the table `tbl` as Markdown:

        >>> tbl.add_computed_column(pages=ocr(tbl.doc))

        Read the first 10 pages at a higher resolution:

        >>> tbl.add_computed_column(pages=ocr(tbl.doc, pages=[[0, 10]], dpi=150))

        Get text lines with bounding boxes from an image column `img`:

        >>> tbl.add_computed_column(
        ...     lines=ocr(tbl.img, model='paddleocr/pp-ocrv6', method='ocr')
        ... )
    """
    extra: dict[str, Any] = {}
    if pages is not None:
        extra['document_pages'] = pages
    if dpi is not None:
        extra['document_dpi'] = dpi
    url = await asyncio.to_thread(_file_data_uri, document, 'document')
    return await _structured_read(
        {'type': 'document_url', 'document_url': {'url': url}}, model, method, extra, model_kwargs
    )


@ocr.overload
async def _(
    document: PIL.Image.Image,
    *,
    model: str = 'paddlepaddle/paddleocr-vl-1.6',
    method: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    url = await asyncio.to_thread(_image_url, document)
    return await _structured_read({'type': 'image_url', 'image_url': {'url': url}}, model, method, {}, model_kwargs)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun_gateway')
async def analyze(
    image: PIL.Image.Image,
    *,
    model: str,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Runs a VLM Run Gateway vision model on an image or a video: object detection, segmentation, pose estimation
    or captioning.

    For details, see <https://docs.vlm.run/gateway/visual-intelligence>

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun_gateway`, key `rate_limit`). If no rate limit is
    configured, uses a default of 600 RPM, which is above the Gateway's limits of 240 requests per minute and
    10,000 per hour per account; set `rate_limit = 160` to stay under both.

    __Requirements:__

    - `pip install openai`

    Args:
        image: The image to analyze. A video (`pxt.Video`) is also accepted by video-capable models, such as
            `'facebook/sam3.1'` with `method='track'` or `'usyd-community/vitpose-plus-large'`.
        model: The model to use, such as `'facebook/sam3.1'` (segmentation), `'microsoft/florence-2-base-ft'`
            (detection and captions), `'usyd-community/vitpose-plus-large'` (body pose) or `'geopavlakos/hamer'`
            (hand pose). For the available models, see <https://vlm.run/gateway/models>.
        method: The model entrypoint to run, such as `'segment'` or `'od'`. If omitted, the model's default method
            is used. For each model's methods, see <https://docs.vlm.run/gateway/methods>.
        method_params: Arguments for the method, such as `{'prompt': 'car'}` for `'facebook/sam3.1'` segmentation.
        model_kwargs: Additional Gateway fields, such as `precision`, `video_fps` or `video_max_frames`.
            `'facebook/sam3.1'` with `method='track'` reads `video_fps` and `video_max_frames` from `method_params`
            instead. For details, see <https://docs.vlm.run/gateway/extra-kwargs>.

    Returns:
        The Gateway's JSON result, unchanged: the input's metadata (such as `image_width` and `image_height`) and
        `content`, which is `{'object': ..., 'items': [...]}` (for a video, also `frames`), or a string for caption
        methods. Image coordinates are fractions of the image width and height, and pose keypoints can fall outside
        0 to 1; `'geopavlakos/hamer'` also returns metric 3D joints in `kpts_xyz`.
        For each `object` type, see <https://docs.vlm.run/gateway/response-types>.

    Examples:
        Segment the cars in an image column `img` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     cars=analyze(
        ...         tbl.img,
        ...         model='facebook/sam3.1',
        ...         method='segment',
        ...         method_params={'prompt': 'car'},
        ...     )
        ... )
    """
    extra = {'method_params': method_params} if method_params is not None else {}
    url = await asyncio.to_thread(_image_url, image)
    return await _structured_read({'type': 'image_url', 'image_url': {'url': url}}, model, method, extra, model_kwargs)


@analyze.overload
async def _(
    image: pxt.Video,
    *,
    model: str,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    extra = {'method_params': method_params} if method_params is not None else {}
    url = await asyncio.to_thread(_file_data_uri, image, 'video')
    return await _structured_read({'type': 'video_url', 'video_url': {'url': url}}, model, method, extra, model_kwargs)


async def _embed(inputs: list, model: str, model_kwargs: dict[str, Any] | None) -> list[np.ndarray]:
    _reject_reserved(model_kwargs or {}, ('model', 'input', 'encoding_format'), args={'input': 'input'})
    response = await _call(
        _client().embeddings.create, model=model, input=inputs, encoding_format='float', extra_body=_body(model_kwargs)
    )
    data = sorted(response.data, key=lambda d: d.index)
    if len(data) != len(inputs):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run Gateway returned {len(data)} embeddings for {len(inputs)} inputs.',
            provider=_PROVIDER,
        )
    return [np.array(d.embedding, dtype=np.float32) for d in data]


async def _embed_files(
    paths: list[str], kind: Literal['document', 'video', 'audio'], model: str, model_kwargs: dict[str, Any] | None
) -> list[np.ndarray]:
    """Embed media files one per request.

    `batch_size` is shared by every overload, and a batch of 32 large files would be one multi-GB request body.
    """
    part_type = f'{kind}_url'
    # Check every file before sending anything, so that a bad file fails the batch before any request is billed.
    await asyncio.to_thread(lambda: [_file_mime(path, kind) for path in paths])
    results: list[np.ndarray] = []
    for path in paths:
        url = await asyncio.to_thread(_file_data_uri, path, kind)
        # Each input is its own list of content parts; a flat list of parts would be pooled into a single vector.
        results.extend(await _embed([[{'type': part_type, part_type: {'url': url}}]], model, model_kwargs))
    return results


@pxt.udf(batch_size=32, resource_pool='request-rate:vlmrun_gateway')
async def embeddings(
    input: Batch[str], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    """
    Creates an embedding vector for text, an image, a video, audio or a PDF document.

    Equivalent to the VLM Run Gateway `embeddings` API endpoint.
    For additional details, see: <https://docs.vlm.run/gateway/api-reference/post-embeddings>

    Text, images and other media share one vector space, so an index built on an image column can be searched with
    a text query. Text and images are sent up to 32 per request. Videos, audio and documents are sent one per
    request, so a media column can use more Gateway requests than the configured rate limit suggests. A document that
    isn't a PDF fails every row in its batch.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun_gateway`, key `rate_limit`). If no rate limit is
    configured, uses a default of 600 RPM, which is above the Gateway's limits of 240 requests per minute and
    10,000 per hour per account; set `rate_limit = 160` to stay under both.

    __Requirements:__

    - `pip install openai`

    Args:
        input: The text to embed. Images (`pxt.Image`), and for models that accept them, videos (`pxt.Video`),
            audio (`pxt.Audio`) and PDF documents (`pxt.Document`) are also accepted.
        model: The embedding model to use: `'qwen/qwen3-vl-embedding-2b'` (2048 dimensions; text, images and video) or
            `'google/embeddinggemma-2'` (768 dimensions; text, images, video, audio and PDFs). For other models,
            pass `dimensions` in `model_kwargs`; it sets the column type and is also sent to the Gateway. An
            embedding index over 2048-dimensional vectors needs `add_embedding_index()`'s default
            `precision='fp16'`, since `fp32` indexes allow at most 2000 dimensions.
        model_kwargs: Additional keyword args for the Gateway `embeddings` API, such as `dimensions` (768, 512, 256
            or 128 for `'google/embeddinggemma-2'`) and `prompt_name`. `'google/embeddinggemma-2'` prefixes text
            with a task (`prompt_name`, default `'Classification'`); for search indexes, use
            `'SentenceSimilarity'`, since the same function embeds both the query and the indexed values.

    Returns:
        An array containing the embedding vector.

    Examples:
        Add an embedding index to an image column `img` of the table `tbl`, and search it with text:

        >>> tbl.add_embedding_index(
        ...     tbl.img,
        ...     embedding=embeddings.using(model='qwen/qwen3-vl-embedding-2b'),
        ... )
        >>> sim = tbl.img.similarity(string='a red car')
        >>> tbl.order_by(sim, asc=False).limit(5).select(tbl.img).collect()
    """
    return await _embed(list(input), model, model_kwargs)


@embeddings.overload
async def _(
    input: Batch[PIL.Image.Image], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    urls = await asyncio.to_thread(lambda: [_image_url(img) for img in input])
    return await _embed([[{'type': 'image_url', 'image_url': {'url': u}}] for u in urls], model, model_kwargs)


@embeddings.overload
async def _(
    input: Batch[pxt.Video], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_files(list(input), 'video', model, model_kwargs)


@embeddings.overload
async def _(
    input: Batch[pxt.Audio], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_files(list(input), 'audio', model, model_kwargs)


@embeddings.overload
async def _(
    input: Batch[pxt.Document], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_files(list(input), 'document', model, model_kwargs)


@embeddings.conditional_return_type
def _(model: str, model_kwargs: dict[str, Any] | None = None) -> ts.ArrayType:
    # the same field precedence as the request, where `extra_body` overrides the top level
    dim = (_body(model_kwargs) or {}).get('dimensions') or _EMBEDDING_DIMENSIONS.get(model)
    return ts.ArrayType((dim,), dtype=np.dtype('float32'), nullable=False)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun_gateway')
async def transcriptions(
    audio: pxt.Audio, *, model: str = 'nvidia/parakeet-tdt-0.6b-v3', model_kwargs: dict[str, Any] | None = None
) -> TranscriptionResponse:
    """
    Transcribes speech in an audio file into text.

    Equivalent to the VLM Run Gateway `audio/transcriptions` API endpoint.
    For additional details, see: <https://docs.vlm.run/gateway/api-reference/post-audio-transcriptions>

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun_gateway`, key `rate_limit`). If no rate limit is
    configured, uses a default of 600 RPM, which is above the Gateway's limits of 240 requests per minute and
    10,000 per hour per account; set `rate_limit = 160` to stay under both.

    __Requirements:__

    - `pip install openai`

    Args:
        audio: The audio to transcribe.
        model: The transcription model to use. For the available models, see <https://vlm.run/gateway/models>.
        model_kwargs: Additional keyword args for the Gateway `audio/transcriptions` API, such as `language`,
            `response_format` (`'json'`, `'verbose_json'`, `'text'`, `'srt'` or `'vtt'`) and
            `timestamp_granularities` (`['segment']`, `['word']` or both; applies to `verbose_json` only).

    Returns:
        A dictionary with the transcript in `text`, plus `duration` and `usage` (including the cost in USD in
        `usage.cost`). For `verbose_json`, it also has `segments`, and `words` when `timestamp_granularities`
        includes `'word'`. For the `text`, `srt` and `vtt` formats, the dictionary contains only `text`.

    Examples:
        Transcribe an audio column `audio` of the table `tbl`:

        >>> tbl.add_computed_column(transcript=transcriptions(tbl.audio))
    """
    _reject_reserved(model_kwargs or {}, ('model', 'file'), args={'file': 'audio'})
    mime = _file_mime(audio, 'audio')
    p = pathlib.Path(audio)
    # A file cached from a URL can lack an extension; give the upload one.
    name = p.name if p.suffix else p.name + (mimetypes.guess_extension(mime) or '.mp3')
    data = await asyncio.to_thread(p.read_bytes)
    result = await _call(
        _client().audio.transcriptions.create, file=(name, data, mime), model=model, extra_body=_body(model_kwargs)
    )
    if isinstance(result, str):
        return {'text': result}
    return cast(TranscriptionResponse, result.model_dump(mode='json'))


__all__ = ['analyze', 'chat_completions', 'embeddings', 'ocr', 'transcriptions']


def __dir__() -> list[str]:
    return __all__
