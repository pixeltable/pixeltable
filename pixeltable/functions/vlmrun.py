"""
Pixeltable UDFs that wrap the [VLM Run](https://docs.vlm.run/) Orion agent API.

In order to use them, you must first `pip install vlmrun` and configure your VLM Run API key, by setting the
`VLMRUN_API_KEY` environment variable or `vlmrun.api_key` in the Pixeltable config file.
"""

import asyncio
import contextlib
import copy
import json
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Generator, Literal

import httpx
import PIL.Image
import pydantic

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import RetryInfo, exponential_backoff, is_retryable_error, redact_url
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    import vlmrun.client  # type: ignore[import-untyped]

_logger = logging.getLogger(__name__)

_BASE_URL = httpx.URL('https://agent.vlm.run/v1/')

# Media formats accepted by the Orion chat completions API
_SUPPORTED_EXTENSIONS = frozenset(
    {
        *('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'),
        *('.mp4', '.mov', '.avi', '.mkv', '.webm'),
        *('.pdf', '.doc', '.docx'),
    }
)

# Content part types whose value can come from a Pixeltable media column
_MEDIA_PART_TYPES = ('image_url', 'video_url', 'file_url')

# An artifact can lag the completion that references it, so a 404 is retried for this long before it counts as a bad id
_NOT_FOUND_GRACE = 60.0

# _poll() follows redirects itself, so it caps a chain of them as httpx does
_MAX_REDIRECTS = 20

_MAX_DETAIL_LEN = 500

_Artifact = Literal['image', 'video', 'document']

# An artifact id is a type prefix and this, as in img_a1b2c3
_ARTIFACT_ID_SUFFIX = r'_\w{6}'

# id prefix and file extension of each kind of generated artifact
_ARTIFACT_TYPES: dict[_Artifact, tuple[str, str]] = {
    'image': ('img', ''),
    'video': ('vid', '.mp4'),
    'document': ('doc', '.pdf'),
}


class _ApiKeyAuth(httpx.Auth):
    """Sends the API key to the API host only, never to a storage host or another host named in a redirect."""

    def __init__(self, api_key: str) -> None:
        self._header = f'Bearer {api_key}'

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if request.url.host == _BASE_URL.host:
            request.headers['Authorization'] = self._header
        yield request


async def _require_https(request: httpx.Request) -> None:  # noqa: RUF029  # AsyncClient awaits its event hooks
    # runs for every request the client sends, including each redirect it follows, so that a URL or a Location from
    # the API can't point it at a plain-http endpoint, such as one on an internal network
    if request.url.scheme != 'https':
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run returned a URL that is not https: {redact_url(str(request.url))}',
            provider='vlmrun',
        )


class _VlmRunClient:
    """Client for the VLM Run API. Uploads and deletes files with the vlmrun SDK. Chat completions, polling and
    downloads go through a connection pool of its own: the SDK's chat client follows the 303 that answers a
    long-running request, where this client polls the result URL instead."""

    timeout: float
    _sdk: 'vlmrun.client.VLMRun'
    _http: httpx.AsyncClient

    def __init__(self, api_key: str, timeout: float) -> None:
        from vlmrun.client import VLMRun

        self.timeout = timeout
        # require_api_key=False skips the key check over the network that the SDK otherwise makes here
        self._sdk = VLMRun(api_key=api_key, base_url=str(_BASE_URL).rstrip('/'), require_api_key=False)
        # no read timeout: a request runs for up to ~150s before the gateway redirects it, a long poll holds the
        # connection while the job runs, and deadline() bounds the whole call
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            auth=_ApiKeyAuth(api_key),
            timeout=httpx.Timeout(None, connect=30.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
            event_hooks={'request': [_require_https]},
        )

    async def close(self) -> None:
        await self._http.aclose()

    @contextlib.asynccontextmanager
    async def deadline(self) -> AsyncIterator[None]:
        """Bounds a UDF call, from its first upload to its last download, by the configured timeout."""
        try:
            async with asyncio.timeout(self.timeout):
                yield
        except TimeoutError:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_TIMEOUT,
                f'VLM Run call did not complete within {self.timeout}s; to allow more time, set `VLMRUN_TIMEOUT` '
                'or `timeout` in the `vlmrun` section of the config file',
                provider='vlmrun',
            ) from None

    async def _send(self, method: str, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR, f'VLM Run request failed: {exc!r}', provider='vlmrun'
            ) from exc

    async def upload(self, file: PIL.Image.Image | str) -> str:
        """Uploads an image, or the local file at the given path, and returns its file id."""
        if isinstance(file, PIL.Image.Image):
            # the SDK uploads from a path
            path = TempStore.create_path(extension='.png')
            try:
                await asyncio.to_thread(file.save, path, format='PNG')
                return await self._upload(path)
            finally:
                path.unlink(missing_ok=True)
        path = Path(file)
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            supported = ', '.join(sorted(_SUPPORTED_EXTENSIONS))
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'Unsupported file format: {path.suffix}. Supported formats: {supported}',
            )
        if not path.is_file():
            raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'Not a file: {file}')
        return await self._upload(path)

    async def _upload(self, path: Path) -> str:
        try:
            # force=True uploads a new file even if one with the same content exists: delete_files() deletes this
            # call's files when the call ends, and another call may be using an existing one
            uploaded = await asyncio.to_thread(self._sdk.files.upload, file=path, force=True)
        except Exception as exc:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR,
                f'Upload to VLM Run failed: {exc}',
                provider='vlmrun',
                status_code=getattr(exc, 'http_status', None),
            ) from exc
        if uploaded.id is None:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                'VLM Run returned an upload without an id',
                provider='vlmrun',
            )
        return uploaded.id

    async def delete_files(self, file_ids: list[str]) -> None:
        """Deletes uploaded files, logging rather than raising a failure: the call's result doesn't depend on it."""
        # each delete is bounded, as the SDK retries a failing one several times
        results = await asyncio.gather(
            *(asyncio.wait_for(asyncio.to_thread(self._sdk.files.delete, file_id), 30.0) for file_id in file_ids),
            return_exceptions=True,
        )
        for file_id, result in zip(file_ids, results):
            if isinstance(result, BaseException):
                _logger.debug(f'Deleting VLM Run file {file_id} failed: {result!r}')

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Creates a chat completion and returns the API response."""
        resp = await self._send('POST', 'openai/chat/completions', json=body)
        if resp.status_code == 303 and 'location' in resp.headers:
            # the gateway answers a request that runs longer than ~150s with the URL that will serve its result
            resp = await self._poll(resp.url.join(resp.headers['location']))
        return _json(resp)

    async def download(self, ref: str, session_id: str | None, dest: Path) -> None:
        """Downloads a generated artifact, given its id or URL, into `dest`."""
        if ref.startswith('https://'):
            await self._poll(ref, dest=dest)
            return
        if session_id is None:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                'VLM Run returned an artifact without a session_id',
                provider='vlmrun',
            )
        params = {'object_id': ref, 'session_id': session_id}
        if ref.startswith('url_'):
            # the body of a url_ artifact is the media's URL
            url = (await self._poll('artifacts', params=params)).text.strip()
            if not url.startswith('https://'):
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    f'VLM Run artifact {ref} holds a URL that is not https: {redact_url(url)[:_MAX_DETAIL_LEN]}',
                    provider='vlmrun',
                )
            await self._poll(url, dest=dest)
        else:
            await self._poll('artifacts', params=params, dest=dest)

    async def _poll(
        self, url: str | httpx.URL, *, params: dict[str, str] | None = None, dest: Path | None = None
    ) -> httpx.Response:
        """GETs `url` until it returns a non-empty result, and returns that response.

        The result's body is streamed into `dest` if given, and read into the response otherwise. A 429 or a dropped
        connection is retried here until deadline() ends the call; any other failure, a 5xx included, raises an error
        that the scheduler doesn't retry, since the job has already run and a rerun would pay for another generation.
        """
        start = time.monotonic()
        attempt = 0
        redirects = 0
        while True:
            delay: float | None = None
            try:
                async with self._http.stream('GET', url, params=params) as resp:
                    status = resp.status_code
                    if status in (200, 201) and dest is not None:
                        await _write_body(resp, dest)
                        if dest.stat().st_size > 0:
                            return resp
                    else:
                        await resp.aread()
                        if status in (200, 201) and len(resp.content.strip()) > 0:
                            return resp
                    if resp.is_redirect:
                        next_url = resp.url.join(resp.headers['location'])
                        # a redirect to the same URL means that the result isn't ready yet, so only new URLs count
                        if next_url != resp.url:
                            redirects += 1
                            if redirects > _MAX_REDIRECTS:
                                raise pxt.ExternalServiceError(
                                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                                    f'VLM Run redirected more than {_MAX_REDIRECTS} times, last to '
                                    f'{redact_url(str(next_url))[:_MAX_DETAIL_LEN]}',
                                    provider='vlmrun',
                                )
                        url, params = next_url, None
                    elif (
                        status in (200, 201, 202, 204)  # 200 and 201 with an empty body
                        or (status == 404 and time.monotonic() - start < _NOT_FOUND_GRACE)
                    ):
                        delay = _retry_info(resp).retry_delay
                    elif status >= 500:
                        raise pxt.ExternalServiceError(
                            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                            f'VLM Run request to {resp.url.path} failed with status {status}: {_error_detail(resp)}',
                            provider='vlmrun',
                            status_code=status,
                        )
                    else:
                        error = _status_error(resp)
                        if error.error_code not in (pxt.ErrorCode.RATE_LIMITED, pxt.ErrorCode.PROVIDER_OVERLOADED):
                            raise error
                        delay = error.retry_after
                    _logger.debug(f'VLM Run result at {resp.url.path} not ready (status {status})')
            except httpx.TransportError as exc:
                # a long poll can outlast the connection
                _logger.debug(f'Polling VLM Run result failed, will poll again: {exc!r}')
            # a Retry-After, as on a 429 or a 503, sets the least wait
            await asyncio.sleep(max(0.0 if delay is None else delay, exponential_backoff(attempt)))
            attempt += 1


@register_client('vlmrun', credential_param='api_key')
def _(api_key: str, timeout: float | None = None) -> _VlmRunClient:
    Env.get().require_package('vlmrun', [0, 9, 3])
    return _VlmRunClient(api_key, timeout=600.0 if timeout is None else timeout)


def _vlmrun_client() -> _VlmRunClient:
    return get_runtime().get_client('vlmrun')


async def _write_body(resp: httpx.Response, dest: Path) -> None:
    with dest.open('wb') as f:
        async for chunk in resp.aiter_bytes():
            await asyncio.to_thread(f.write, chunk)


def _retry_info(resp: httpx.Response) -> RetryInfo:
    return is_retryable_error(httpx.HTTPStatusError(f'status {resp.status_code}', request=resp.request, response=resp))


def _status_error(resp: httpx.Response) -> pxt.ExternalServiceError:
    status = resp.status_code
    retryable, retry_after = _retry_info(resp)
    code: pxt.ErrorCode
    if status in (401, 403):
        code = pxt.ErrorCode.PROVIDER_AUTH_ERROR
    elif status == 429:
        code = pxt.ErrorCode.RATE_LIMITED
    elif retryable:
        code = pxt.ErrorCode.PROVIDER_OVERLOADED
    elif status >= 500:
        code = pxt.ErrorCode.PROVIDER_ERROR
    else:
        code = pxt.ErrorCode.PROVIDER_BAD_REQUEST
    return pxt.ExternalServiceError(
        code,
        f'VLM Run request to {resp.url.path} failed with status {status}: {_error_detail(resp)}',
        retry_after=retry_after,
        provider='vlmrun',
        status_code=status,
    )


def _error_detail(resp: httpx.Response) -> str:
    """Returns the error message of a failed response, as the API reports it."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    message: Any = None
    if isinstance(body, dict):
        error = body.get('error')
        message = error.get('message') if isinstance(error, dict) else body.get('detail')
    return (resp.text if message is None else str(message))[:_MAX_DETAIL_LEN]


def _json(resp: httpx.Response) -> dict[str, Any]:
    if not resp.is_success:
        raise _status_error(resp)
    try:
        result = resp.json()
    except ValueError:
        result = None
    if not isinstance(result, dict):
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run returned a response that is not a JSON object: {resp.text[:_MAX_DETAIL_LEN]}',
            provider='vlmrun',
            status_code=resp.status_code,
        )
    return result


def _media_ref(ref: str, prefix: str) -> str:
    """Returns the artifact id or https URL in a reference to generated media with the given artifact id prefix.

    An artifact id can come with a file extension, as in `vid_a1b2c3.mp4`, and a `url_` artifact stands for any type.
    """
    ref = ref.strip()
    m = re.fullmatch(rf'((?:{prefix}|url){_ARTIFACT_ID_SUFFIX})(?:\.\w+)?', ref)
    if m is not None:
        return m.group(1)
    if ref.startswith('https://'):
        return ref
    raise pxt.ExternalServiceError(
        pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
        f'VLM Run returned an unrecognized media reference: {ref[:_MAX_DETAIL_LEN]!r}',
        provider='vlmrun',
    )


async def _resolve_messages(messages: list, client: _VlmRunClient, file_ids: list[str]) -> list:
    """Returns a copy of `messages` in which media from Pixeltable columns is uploaded and referenced by file id.

    The ids of the uploaded files are appended to `file_ids`.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get('type') not in _MEDIA_PART_TYPES:
                continue
            part_type = part['type']
            value = part.get(part_type)
            if isinstance(value, dict) and isinstance(value.get('url'), str):
                continue
            if value is None:
                raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`{part_type}` is null')
            # an image column produces a PIL image, and a video or document column a local path
            if isinstance(value, str) and (part_type == 'image_url' or len(urllib.parse.urlparse(value).scheme) > 1):
                part[part_type] = {'url': value}
            elif isinstance(value, (PIL.Image.Image, str)):
                file_ids.append(await client.upload(value))
                content[i] = {'type': 'input_file', 'file_id': file_ids[-1]}
            else:
                raise pxt.RequestError(
                    pxt.ErrorCode.INVALID_ARGUMENT,
                    f'`{part_type}` must be a media column, a URL or a dict with a `url`; got {type(value).__name__}',
                )
    return messages


def _response_format(artifact: _Artifact) -> tuple[str, dict[str, Any]]:
    """Returns the key of the generated artifact in the response, and a `response_format` that asks for it."""
    from vlmrun.types import refs  # type: ignore[import-untyped]

    key: str = artifact
    fields: dict[str, Any]
    if artifact == 'video':
        # asked for a VideoRef, the API returned made-up ids such as vid_abc123 in live tests, so this asks for the
        # video's URL instead
        key = 'url'
        fields = {key: (str, pydantic.Field(..., description='The URL of the generated video'))}
    else:
        fields = {key: ({'image': refs.ImageRef, 'document': refs.DocumentRef}[artifact], ...)}
    output_model = pydantic.create_model(f'{artifact.title()}Output', **fields)
    return key, {'type': 'json_schema', 'schema': output_model.model_json_schema()}


def _check_model_kwargs(model_kwargs: dict[str, Any] | None, reserved: tuple[str, ...]) -> None:
    if model_kwargs is None:
        return
    for key in reserved:
        if key in model_kwargs:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT, f'`model_kwargs` cannot contain `{key}`: this function sets it'
            )


async def _generate(
    prompt: str,
    inputs: list[PIL.Image.Image | str | None],
    *,
    artifact: _Artifact,
    toolset: str,
    system_message: str,
    model: str,
    model_kwargs: dict[str, Any] | None,
) -> Path:
    """Runs a generation request and returns the path of a TempStore file that holds the generated artifact."""
    _check_model_kwargs(model_kwargs, ('model', 'messages', 'response_format', 'toolsets'))
    body: dict[str, Any] = {} if model_kwargs is None else dict(model_kwargs)
    prefix, extension = _ARTIFACT_TYPES[artifact]
    key, response_format = _response_format(artifact)
    client = _vlmrun_client()
    file_ids: list[str] = []
    try:
        async with client.deadline():
            content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
            for input in inputs:
                if input is not None:
                    file_ids.append(await client.upload(input))
                    content.append({'type': 'input_file', 'file_id': file_ids[-1]})
            body.update(
                model=model,
                messages=[{'role': 'system', 'content': system_message}, {'role': 'user', 'content': content}],
                response_format=response_format,
                toolsets=[toolset],
            )
            result = await client.chat(body)
            try:
                ref = json.loads(result['choices'][0]['message']['content'])[key]
                if isinstance(ref, dict):
                    ref = ref['id']
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    f'VLM Run returned no {artifact}: {str(result)[:_MAX_DETAIL_LEN]}',
                    provider='vlmrun',
                ) from exc
            _logger.debug(f'VLM Run returned {artifact} {ref!r} in session {result.get("session_id")}')
            dest = TempStore.create_path(extension=extension)
            try:
                await client.download(_media_ref(str(ref), prefix), result.get('session_id'), dest)
            except BaseException:
                # including cancellation by deadline(): nothing downstream knows the path of a partial download
                dest.unlink(missing_ok=True)
                raise
            return dest
    finally:
        await client.delete_files(file_ids)


def _load_image(path: Path) -> PIL.Image.Image:
    try:
        img = PIL.Image.open(path)
        img.load()
        return img
    finally:
        path.unlink()


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def chat_completions(messages: list, *, model: str, model_kwargs: dict[str, Any] | None = None) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the VLM Run chat completions API endpoint.
    For additional details, see: <https://docs.vlm.run/agents/agent-chat-completions>

    Message content items can reference Pixeltable media columns directly: image columns via
    `{'type': 'image_url', 'image_url': tbl.img}`, video columns via `{'type': 'video_url', 'video_url': tbl.video}`,
    and document columns via `{'type': 'file_url', 'file_url': tbl.doc}`. Referenced media is uploaded to VLM Run
    for each call, and deleted from VLM Run when the call ends. A string in `video_url` or `file_url` that is not a
    URL is read as a local file path and uploaded, so don't put untrusted strings there.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate limit is configured,
    uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        messages: A list of messages for the chat conversation. Media content items can reference Pixeltable
            columns directly (see above).
        model: The model to use, such as `'vlmrun-orion-2:fast'`, `'vlmrun-orion-2:auto'` or `'vlmrun-orion-2:pro'`.
            For the full list, see <https://docs.vlm.run/>.
        model_kwargs: Additional parameters for the VLM Run chat completions API, such as `response_format` for
            structured JSON output or `toolsets` to select the agent's tools.

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
        >>> tbl.add_computed_column(
        ...     response=vlmrun.chat_completions(
        ...         messages, model='vlmrun-orion-2:auto'
        ...     )
        ... )

        Ask for the invoice number and total of a document column `doc` as a JSON string:

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
        ...         model='vlmrun-orion-2:auto',
        ...         model_kwargs={'response_format': {'type': 'json_object'}},
        ...     )
        ... )
    """
    _check_model_kwargs(model_kwargs, ('model', 'messages'))
    body: dict[str, Any] = {} if model_kwargs is None else dict(model_kwargs)
    client = _vlmrun_client()
    file_ids: list[str] = []
    try:
        async with client.deadline():
            body.update(model=model, messages=await _resolve_messages(messages, client, file_ids))
            result = await client.chat(body)
    finally:
        await client.delete_files(file_ids)
    if 'choices' not in result:
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'VLM Run returned a response without choices: {str(result)[:_MAX_DETAIL_LEN]}',
            provider='vlmrun',
        )
    return result


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_image(
    prompt: str, *, image: pxt.Image | None = None, model: str, model_kwargs: dict[str, Any] | None = None
) -> PIL.Image.Image:
    """
    Generates an image from a text prompt, or edits an image, using VLM Run's `image-gen` toolset.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun`

    Args:
        prompt: Text prompt describing the image to generate or the edit to apply.
        image: The image to edit. If it is omitted, or null in a row, the image is generated from the prompt alone.
        model: The model to use, such as `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional parameters for the VLM Run chat completions API. This function sets
            `response_format` and `toolsets` itself, so they cannot be passed.

    Returns:
        The generated or edited image.

    Examples:
        Generate an image from text:

        >>> tbl.add_computed_column(
        ...     image=vlmrun.generate_image(tbl.prompt, model='vlmrun-orion-2:auto')
        ... )

        Edit an existing image column `img`:

        >>> tbl.add_computed_column(
        ...     edited=vlmrun.generate_image(
        ...         'Blur all faces', image=tbl.img, model='vlmrun-orion-2:auto'
        ...     )
        ... )
    """
    path = await _generate(
        prompt,
        [image],
        artifact='image',
        toolset='image-gen',
        system_message='Always return a generated image in your response.',
        model=model,
        model_kwargs=model_kwargs,
    )
    return await asyncio.to_thread(_load_image, path)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def annotate_image(
    prompt: str, image: pxt.Image, *, model: str, model_kwargs: dict[str, Any] | None = None
) -> PIL.Image.Image:
    """
    Draws bounding boxes, keypoints or segmentation masks on an image, using VLM Run's `viz` toolset.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun`

    Args:
        prompt: Text prompt describing what to annotate, such as `'Draw bounding boxes around all people'`.
        image: The image to annotate.
        model: The model to use, such as `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional parameters for the VLM Run chat completions API. This function sets
            `response_format` and `toolsets` itself, so they cannot be passed.

    Returns:
        The annotated image.

    Examples:
        Annotate objects in an image column `img`:

        >>> tbl.add_computed_column(
        ...     annotated=vlmrun.annotate_image(
        ...         'Draw bounding boxes around all people',
        ...         tbl.img,
        ...         model='vlmrun-orion-2:auto',
        ...     )
        ... )
    """
    path = await _generate(
        prompt,
        [image],
        artifact='image',
        toolset='viz',
        system_message='Always return an annotated image in your response.',
        model=model,
        model_kwargs=model_kwargs,
    )
    return await asyncio.to_thread(_load_image, path)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_video(
    prompt: str,
    *,
    image: pxt.Image | None = None,
    video: pxt.Video | None = None,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Generates a video from a text prompt, using VLM Run's `video` toolset.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun`

    Args:
        prompt: Text prompt describing the video to generate.
        image: An image to start the video from.
        video: A video to start from. If both `image` and `video` are omitted, or null in a row, the video is
            generated from the prompt alone.
        model: The model to use, such as `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional parameters for the VLM Run chat completions API. This function sets
            `response_format` and `toolsets` itself, so they cannot be passed.

    Returns:
        The generated video.

    Examples:
        Generate videos from text prompts:

        >>> tbl.add_computed_column(
        ...     video=vlmrun.generate_video(tbl.prompt, model='vlmrun-orion-2:auto')
        ... )
    """
    path = await _generate(
        prompt,
        [image, video],
        artifact='video',
        toolset='video',
        system_message='Always return a generated video in your response.',
        model=model,
        model_kwargs=model_kwargs,
    )
    return str(path)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:vlmrun')
async def generate_document(
    prompt: str, document: pxt.Document, *, model: str, model_kwargs: dict[str, Any] | None = None
) -> pxt.Document:
    """
    Transforms a document, for example by redacting it, using VLM Run's `document` toolset.

    For additional details, see: <https://docs.vlm.run/>

    __Requirements:__

    - `pip install vlmrun`

    Args:
        prompt: Text prompt describing the transformation, such as
            `'Redact all personally identifiable information'`.
        document: The document to transform: a PDF, DOC or DOCX file.
        model: The model to use, such as `'vlmrun-orion-2:auto'`.
        model_kwargs: Additional parameters for the VLM Run chat completions API. This function sets
            `response_format` and `toolsets` itself, so they cannot be passed.

    Returns:
        The transformed document, as a PDF.

    Examples:
        Redact PII from a document column `doc`:

        >>> tbl.add_computed_column(
        ...     redacted=vlmrun.generate_document(
        ...         'Redact all names and email addresses',
        ...         tbl.doc,
        ...         model='vlmrun-orion-2:auto',
        ...     )
        ... )
    """
    path = await _generate(
        prompt,
        [document],
        artifact='document',
        toolset='document',
        system_message='Always return a generated document in your response.',
        model=model,
        model_kwargs=model_kwargs,
    )
    return str(path)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
