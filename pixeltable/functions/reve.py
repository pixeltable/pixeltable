"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap [Reve](https://app.reve.com/) image
generation API. In order to use them, the API key must be specified either with `REVE_API_KEY` environment variable,
or as `api_key` in the `reve` section of the Pixeltable config file.
"""

import asyncio
import atexit
import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, TypedDict

import aiohttp
import PIL.Image

import pixeltable as pxt
from pixeltable.env import register_client
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

_logger = logging.getLogger(__name__)


class ImageResponse(TypedDict):
    image: PIL.Image.Image
    layout: dict


class ReveRateLimitedError(Exception):
    pass


class ReveContentViolationError(Exception):
    pass


class ReveUnexpectedError(Exception):
    pass


_REVE_BASE_URL = 'https://api.reve.com'
_RETRY_AFTER_RE = re.compile(r'\d{1,2}')


class _ReveClient:
    """
    Client for interacting with the Reve API.
    """

    _request_headers: dict[str, str]
    _session: aiohttp.ClientSession | None

    def __init__(self, api_key: str):
        self._request_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        self._session = None  # defer session creation until we have a running event loop

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(base_url=_REVE_BASE_URL)
            session = self._session
            atexit.register(lambda: asyncio.run(session.close()))
        return self._session

    async def post(self, endpoint: str, *, payload: dict) -> dict[str, Any]:
        async with self._get_session().post(endpoint, json=payload, headers=self._request_headers) as resp:
            request_id = resp.headers.get('X-Reve-Request-Id')
            error_code = resp.headers.get('X-Reve-Error-Code')
            match resp.status:
                case 200:
                    if error_code is not None:
                        raise ReveUnexpectedError(
                            f'Reve request {request_id} returned an unexpected error {error_code}'
                        )
                    content_violation = resp.headers.get('X-Reve-Content-Violation', 'false')
                    if content_violation.lower() != 'false':
                        raise ReveContentViolationError(
                            f'Reve request {request_id} resulted in a content violation error'
                        )
                    if resp.content_type != 'application/json':
                        raise ReveUnexpectedError(
                            f'Reve request {request_id} expected content type application/json, got {resp.content_type}'
                        )
                    raw = await resp.read()
                    if len(raw) == 0:
                        _logger.error(f'Reve request {request_id} resulted in an empty response: {resp}')
                        raise ReveUnexpectedError(f'Reve request {request_id} resulted in an empty response')
                    _logger.debug(f'Reve request {request_id} successful')
                    return json.loads(raw)
                case 429:
                    # Try to honor the server-provided Retry-After value if present
                    # Note: Retry-After value can also be given in the form of HTTP Date, which we don't
                    # currently handle
                    retry_after_seconds = None
                    retry_after_header = resp.headers.get('Retry-After')
                    if retry_after_header is not None and _RETRY_AFTER_RE.fullmatch(retry_after_header):
                        retry_after_seconds = int(retry_after_header)
                    _logger.info(
                        f'Reve request {request_id} failed due to rate limiting, retry after header value: '
                        f'{retry_after_header}'
                    )
                    # This error message is formatted specifically so that RequestRateScheduler can extract the
                    # retry delay from it
                    raise ReveRateLimitedError(
                        f'Reve request {request_id} failed due to rate limiting (429). retry-after:'
                        f'{retry_after_seconds}'
                    )
                case _:
                    _logger.info(
                        f'Reve request {request_id} failed with status code {resp.status} and error code {error_code}: '
                        f'{resp}'
                    )
                    if resp.content_type == 'application/json':
                        json_body = await resp.text(errors='replace')
                        json_body = json_body if len(json_body) <= 1024 else json_body[:1024] + '...'
                        _logger.info(f'Response body: {json_body}')
                    raise ReveUnexpectedError(
                        f'Reve request failed with status code {resp.status} and error code {error_code}'
                    )


@register_client('reve', credential_param='api_key')
def _(api_key: str) -> _ReveClient:
    return _ReveClient(api_key=api_key)


def _client() -> _ReveClient:
    from pixeltable.runtime import get_runtime

    return get_runtime().get_client('reve')


# At the moment, Reve does not document their rate limits, and API responses do not include rate limits-related tokens.
@pxt.udf(is_deterministic=False, resource_pool='request-rate:reve')
async def create(
    prompt: str,
    *,
    references: list[PIL.Image.Image] | None = None,
    aspect_ratio: str | None = None,
    postprocessing: list[dict] | None = None,
    version: str | None = None,
    model_kwargs: dict | None = None,
) -> ImageResponse:
    """
    Creates an image from a text prompt, optionally guided by reference images.

    The prompt may include `<frame>0</frame>`, `<frame>1</frame>`, etc. tags to refer to the reference images.

    This UDF wraps the `https://api.reve.com/v2/image/create` endpoint. For more information, refer to the official
    [API documentation](https://api.reve.com/console/docs/v2/create).

    Args:
        prompt: prompt describing the desired image
        references: optional list of reference images to guide the model
        aspect_ratio: desired image aspect ratio, e.g. '3:2', '16:9', '1:1', etc.
        postprocessing: optional list of postprocessing operations to apply to the generated image
            e.g. `[{'process': 'effect', 'effect_name': 'low_light'}]`.
        version: specific model version to use. Latest if not specified.
        model_kwargs: additional keyword arguments to pass to the Reve API.

    Returns:
        A dictionary containing the generated image (`'image'` key) and layout metadata
        (`'layout'` key) returned by the Reve API.

    Examples:
        Add a computed column with generated images from text prompts:

        >>> t.add_computed_column(img=reve.create(t.prompt, aspect_ratio='1:1'))

        Remove the background from an image:

        >>> t.add_computed_column(
        ...     img=reve.create(
        ...         'Remove the background from <frame>0</frame> and replace it with a clean white background',
        ...         references=[t.product_img],
        ...     )
        ... )

        Add a computed column that generates product shots in the style of a brand reference image:

        >>> t.add_computed_column(
        ...     img=reve.create(
        ...         'Generate a product shot of <frame>0</frame> styled to match the aesthetic of <frame>1</frame>',
        ...         references=[t.product_img, t.brand_reference_img],
        ...     )
        ... )
    """
    payload: dict[str, Any] = {'prompt': prompt}
    if references is not None and len(references) > 0:
        payload['references'] = [{'data': to_base64(img)} for img in references]
    if aspect_ratio is not None:
        payload['aspect_ratio'] = aspect_ratio
    if version is not None:
        payload['version'] = version
    if postprocessing is not None:
        payload['postprocessing'] = postprocessing
    if model_kwargs is not None:
        payload.update(model_kwargs)

    body = await _client().post('/v2/image/create', payload=payload)
    for field in ('image', 'layout'):
        if field not in body:
            raise ReveUnexpectedError(f'Reve response missing {field} field')
    img_bytes = base64.b64decode(body['image'])
    img = PIL.Image.open(BytesIO(img_bytes))
    img.load()
    _logger.debug(f'Image bytes: {len(img_bytes)}, size: {img.size}, format: {img.format}, mode: {img.mode}')
    return ImageResponse(image=img, layout=body['layout'])


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
