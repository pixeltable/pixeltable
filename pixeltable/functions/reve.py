"""
Pixeltable [UDFs](https://docs.pixeltable.com/datastore/custom-functions) that wrap [Reve](https://app.reve.com/) image
generation API. In order to use them, the API key must be specified either with `REVE_API_KEY` environment variable,
or as `api_key` in the `reve` section of the Pixeltable config file.
"""

import asyncio
import atexit
import base64
import logging
import re
from io import BytesIO

import aiohttp
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

_logger = logging.getLogger('pixeltable')
_SUPPORTED_FORMATS = ('png', 'jpeg', 'webp')


class _ReveClient:
    api_key: str
    session: aiohttp.ClientSession

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = aiohttp.ClientSession(base_url='https://api.reve.com', loop=Env.get().event_loop)
        atexit.register(lambda: asyncio.run(self.session.close()))

    async def _post(self, endpoint: str, *, payload: dict, format: str) -> PIL.Image.Image:
        if format not in _SUPPORTED_FORMATS:
            raise pxt.Error(f'Unsupported image format: {format}. Supported formats: {_SUPPORTED_FORMATS}')
        requested_content_type = 'image/' + format
        request_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': requested_content_type,
        }

        async with self.session.post(endpoint, json=payload, headers=request_headers) as resp:
            request_id = resp.headers.get('X-Reve-Request-Id')
            error_code = resp.headers.get('X-Reve-Error-Code')
            match resp.status:
                case 200:
                    if error_code is not None:
                        raise pxt.Error(f'Reve request {request_id} returned an unexpected error {error_code}')
                    content_violation = resp.headers.get('X-Reve-Content-Violation', 'false')
                    if content_violation.lower() != 'false':
                        raise pxt.Error(f'Reve request {request_id} resulted in a content violation error')
                    if resp.content_type != requested_content_type:
                        raise pxt.Error(
                            f'Reve request {request_id} expected content type {requested_content_type}, '
                            f'got {resp.content_type}'
                        )

                    img_data = await resp.read()
                    if len(img_data) == 0:
                        raise pxt.Error(f'Reve request {request_id} resulted in an empty image')
                    img = PIL.Image.open(BytesIO(img_data))
                    _logger.debug(
                        f'Reve request {request_id} successful. Image bytes: {len(img_data)}, size: {img.size}'
                        f', format: {img.format}, mode: {img.mode}'
                    )
                    return img
                case 429:
                    # Try to honor the server-provided Retry-After value if present
                    # Note: Retry-After value can also be given in the form of HTTP Date, which we don't currently
                    # handle
                    retry_after_seconds = None
                    retry_after_header = resp.headers.get('Retry-After')
                    if retry_after_header is not None and re.fullmatch(r'\d{1,2}', retry_after_header):
                        retry_after_seconds = int(retry_after_header)
                    _logger.info(
                        f'Reve request {request_id} failed due to rate limiting, retry after header value: '
                        f'{retry_after_header}'
                    )
                    # This error message is formatted specifically so that RequestRateScheduler can extract the retry
                    # delay from it
                    raise pxt.Error(
                        f'Reve request {request_id} failed due to rate limiting (429). retry-after:'
                        f'{retry_after_seconds}'
                    )
                case _:
                    _logger.info(
                        f'Reve request {request_id} failed with status code {resp.status} and error code {error_code}'
                    )
                    raise pxt.Error(f'Reve request failed with status code {resp.status} and error code {error_code}')


@register_client('reve')
def _(api_key: str) -> _ReveClient:
    return _ReveClient(api_key=api_key)


def _client() -> _ReveClient:
    return Env.get().get_client('reve')


# TODO Regarding rate limiting: Reve appears to be going for a credits per minute rate limiting model, but does not
# currently communicate rate limit information in responses. Therefore neither of the currently implemented limiting
# strategies is a perfect match, but "request-rate" is the closest. Reve does not currently enforce the rate limits,
# but when it does, we can revisit this choice.
@pxt.udf(resource_pool='request-rate:reve')
async def create(
    prompt: str, *, format: str, aspect_ratio: str | None = None, version: str | None = None
) -> PIL.Image.Image:
    """
    Creates an image from a text prompt.

    This UDF wraps the `https://api.reve.com/v1/image/create` endpoint. For more information, refer to the official
    [API documentation](https://api.reve.com/console/docs/create).

    Args:
        prompt: prompt describing the desired image
        format: desired output image format
        aspect_ratio: desired image aspect ratio, e.g. '3:2', '16:9', '1:1', etc.
        version: specific model version to use. Latest if not specified.

    Returns:
        A generated image

    Examples:
        Add a computed column with generated square images to a table with text prompts:

        >>> t.add_computed_column(
        ...     img=reve.create(t.prompt, format='jpeg', aspect_ratio='1:1')
        ... )
    """
    payload = {'prompt': prompt}
    if aspect_ratio is not None:
        payload['aspect_ratio'] = aspect_ratio
    if version is not None:
        payload['version'] = version

    result = await _client()._post('/v1/image/create', payload=payload, format=format)
    return result


@pxt.udf(resource_pool='request-rate:reve')
async def edit(
    image: PIL.Image.Image, edit_instruction: str, *, output_format: str | None = None, version: str | None = None
) -> PIL.Image.Image:
    """
    Edits images based on a text prompt.

    This UDF wraps the `https://api.reve.com/v1/image/edit` endpoint. For more information, refer to the official
    [API documentation](https://api.reve.com/console/docs/edit)

    Args:
        image: image to edit
        edit_instruction: text prompt describing the desired edit
        output_format: desired output image format. If not specified, uses the input image
            format.
        version: specific model version to use. Latest if not specified.

    Returns:
        A generated image

    Examples:
        Add a computed column with catalog-ready images to the table with product pictures:

        >>> t.add_computed_column(
        ...     catalog_img=reve.edit(
        ...         t.product_img,
        ...         'Remove background and distractions from the product picture, improve lighting.'
        ...     )
        ... )
    """
    if image.format is None or image.format.lower() not in _SUPPORTED_FORMATS:
        raise pxt.Error(f'Input image format {image.format} is not supported. Supported formats: {_SUPPORTED_FORMATS}')
    if output_format is None:
        output_format = image.format.lower()

    payload = {'edit_instruction': edit_instruction, 'reference_image': to_base64(image)}
    if version is not None:
        payload['version'] = version

    result = await _client()._post('/v1/image/edit', payload=payload, format=output_format)
    return result


@pxt.udf(resource_pool='request-rate:reve')
async def remix(
    prompt: str,
    images: list[PIL.Image.Image],
    *,
    output_format: str,
    aspect_ratio: str | None = None,
    version: str | None = None,
) -> PIL.Image.Image:
    """
    Creates images based on a text prompt and reference images.

    The prompt may include `<img>0</img>`, `<img>1</img>`, etc. tags to refer to the images in the `images` argument.

    This UDF wraps the `https://api.reve.com/v1/image/remix` endpoint. For more information, refer to the official
    [API documentation](https://api.reve.com/console/docs/remix)

    Args:
        prompt: prompt describing the desired image
        images: list of reference images
        output_format: desired output image format
        aspect_ratio: desired image aspect ratio, e.g. '3:2', '16:9', '1:1', etc.
        version: specific model version to use. Latest by default.

    Returns:
        A generated image

    Examples:
        Add a computed column with promotional collages to a table with original images:

        >>> t.add_computed_column(
        ...     collage=reve.remix(
        ...         'Create a promotional collage by combining the picture of a hotel room from $0'
        ...         ' with city attractions from $1 and $2',
        ...         images=[t.room_img, t.city_attraction, t.city_skyline],
        ...         output_format='jpeg',
        ...         aspect_ratio='16:9',
        ...     )
        ... )
    """
    if len(images) == 0:
        raise pxt.Error('Must include at least 1 reference image')

    payload = {'prompt': prompt, 'reference_images': [to_base64(img) for img in images]}
    if version is not None:
        payload['version'] = version
    if aspect_ratio is not None:
        payload['aspect_ratio'] = aspect_ratio
    result = await _client()._post('/v1/image/remix', payload=payload, format=output_format)
    return result


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
