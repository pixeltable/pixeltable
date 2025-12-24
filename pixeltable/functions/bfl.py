"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap
[Black Forest Labs (BFL)](https://docs.bfl.ai/) FLUX image generation API. In order to use them,
the API key must be specified either with `BFL_API_KEY` environment variable, or as `api_key`
in the `bfl` section of the Pixeltable config file.

For more information on FLUX models, see the [BFL documentation](https://docs.bfl.ai/).
"""

import asyncio
import atexit
import logging
import re
from io import BytesIO
from typing import Literal

import aiohttp
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

_logger = logging.getLogger('pixeltable')


class BflRateLimitedError(Exception):
    pass


class BflContentModerationError(Exception):
    pass


class BflUnexpectedError(Exception):
    pass


class _BflClient:
    """
    Client for interacting with the BFL API. Maintains a long-lived HTTP session to the service.
    """

    api_key: str
    session: aiohttp.ClientSession

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = Env.get().event_loop.run_until_complete(self._start_session())
        atexit.register(lambda: asyncio.run(self.session.close()))

    async def _start_session(self) -> aiohttp.ClientSession:
        # Don't set base_url because polling_url and image URLs are absolute
        return aiohttp.ClientSession()

    async def _submit_task(self, endpoint: str, *, payload: dict) -> tuple[str, str]:
        """Submit a generation task and return (task_id, polling_url)."""
        request_headers = {'x-key': self.api_key, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        url = f'https://api.bfl.ai{endpoint}'

        async with self.session.post(url, json=payload, headers=request_headers) as resp:
            match resp.status:
                case 200:
                    data = await resp.json()
                    task_id = data.get('id')
                    polling_url = data.get('polling_url')
                    if not task_id or not polling_url:
                        raise BflUnexpectedError(f'BFL API: missing id or polling_url in response: {data}')
                    return task_id, polling_url
                case 402:
                    raise BflUnexpectedError('BFL API: insufficient credits. Please add credits to your account.')
                case 429:
                    # Try to honor the server-provided Retry-After value if present
                    retry_after_seconds = None
                    retry_after_header = resp.headers.get('Retry-After')
                    if retry_after_header is not None and re.fullmatch(r'\d{1,2}', retry_after_header):
                        retry_after_seconds = int(retry_after_header)
                    _logger.info(
                        f'BFL request failed due to rate limiting, retry after header value: {retry_after_header}'
                    )
                    # Error message formatted for RequestRateScheduler to extract the retry delay
                    raise BflRateLimitedError(
                        f'BFL request failed due to rate limiting (429). retry-after:{retry_after_seconds}'
                    )
                case _:
                    error_text = await resp.text()
                    _logger.info(f'BFL request failed with status code {resp.status}: {error_text}')
                    raise BflUnexpectedError(f'BFL API error (status {resp.status}): {error_text}')

    async def _poll_result(self, task_id: str, polling_url: str, max_wait: float = 300.0) -> PIL.Image.Image:
        """Poll for task completion and return the generated image."""
        request_headers = {'x-key': self.api_key, 'Accept': 'application/json'}

        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            async with self.session.get(polling_url, headers=request_headers) as resp:
                match resp.status:
                    case 200:
                        data = await resp.json()
                        status = data.get('status')

                        match status:
                            case 'Ready':
                                sample_url = data.get('result', {}).get('sample')
                                if not sample_url:
                                    raise BflUnexpectedError(f'BFL task {task_id}: missing sample URL in result')
                                return await self._download_image(task_id, sample_url)
                            case 'Request Moderated' | 'Content Moderated':
                                raise BflContentModerationError(
                                    f'BFL task {task_id} resulted in content moderation: {status}'
                                )
                            case 'Error' | 'Failed':
                                error_msg = data.get('error', 'Unknown error')
                                raise BflUnexpectedError(f'BFL task {task_id} failed: {error_msg}')
                            case 'Task not found':
                                raise BflUnexpectedError(f'BFL task {task_id} not found (may have expired)')
                            case 'Pending':
                                await asyncio.sleep(poll_interval)
                                elapsed += poll_interval
                            case _:
                                # Unknown status, wait and retry
                                await asyncio.sleep(poll_interval)
                                elapsed += poll_interval
                    case 429:
                        retry_after_seconds = None
                        retry_after_header = resp.headers.get('Retry-After')
                        if retry_after_header is not None and re.fullmatch(r'\d{1,2}', retry_after_header):
                            retry_after_seconds = int(retry_after_header)
                        _logger.info(f'BFL task {task_id} polling rate limited, retry after: {retry_after_header}')
                        raise BflRateLimitedError(
                            f'BFL task {task_id} polling rate limited (429). retry-after:{retry_after_seconds}'
                        )
                    case _:
                        error_text = await resp.text()
                        _logger.info(f'BFL task {task_id} polling failed with status {resp.status}: {error_text}')
                        raise BflUnexpectedError(f'BFL polling error (status {resp.status}): {error_text}')

        raise BflUnexpectedError(f'BFL task {task_id} timed out after {max_wait} seconds')

    async def _download_image(self, task_id: str, url: str) -> PIL.Image.Image:
        """Download image from the result URL."""
        async with self.session.get(url) as resp:
            if resp.status != 200:
                raise BflUnexpectedError(f'BFL task {task_id}: failed to download image, status {resp.status}')
            img_data = await resp.read()
            if len(img_data) == 0:
                raise BflUnexpectedError(f'BFL task {task_id} resulted in an empty image')
            img = PIL.Image.open(BytesIO(img_data))
            img.load()
            _logger.debug(
                f'BFL task {task_id} successful. Image bytes: {len(img_data)}, size: {img.size}'
                f', format: {img.format}, mode: {img.mode}'
            )
            return img

    async def generate(self, endpoint: str, payload: dict) -> PIL.Image.Image:
        """Submit a generation task and wait for the result."""
        task_id, polling_url = await self._submit_task(endpoint, payload=payload)
        _logger.debug(f'BFL task {task_id} submitted to {endpoint}')
        return await self._poll_result(task_id, polling_url)


@register_client('bfl')
def _(api_key: str) -> _BflClient:
    return _BflClient(api_key=api_key)


def _client() -> _BflClient:
    return Env.get().get_client('bfl')


# Model endpoint mapping
_MODEL_ENDPOINTS = {
    # FLUX.2 models
    'flux-2-pro': '/v1/flux-2-pro',
    'flux-2-flex': '/v1/flux-2-flex',
    'flux-2-max': '/v1/flux-2-max',
    # FLUX 1.1 models
    'flux-pro-1.1': '/v1/flux-pro-1.1',
    'flux-pro-1.1-ultra': '/v1/flux-pro-1.1-ultra',
    # FLUX.1 dev
    'flux-dev': '/v1/flux-dev',
    # Kontext models (for editing)
    'flux-kontext-pro': '/v1/flux-kontext-pro',
    'flux-kontext-max': '/v1/flux-kontext-max',
}


@pxt.udf(resource_pool='request-rate:bfl')
async def generate(
    prompt: str,
    *,
    model: Literal[
        'flux-2-pro', 'flux-2-flex', 'flux-2-max', 'flux-pro-1.1', 'flux-pro-1.1-ultra', 'flux-dev'
    ] = 'flux-2-pro',
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    safety_tolerance: int | None = None,
    output_format: Literal['jpeg', 'png'] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
) -> PIL.Image.Image:
    """
    Generate an image from a text prompt using FLUX models.

    This UDF wraps the BFL FLUX API endpoints. For more information, refer to the official
    [API documentation](https://docs.bfl.ai/flux_2/flux2_text_to_image).

    Args:
        prompt: Text description of the image to generate.
        model: FLUX model to use. Options:
            - 'flux-2-pro': Best balance of speed and quality (default)
            - 'flux-2-flex': Adjustable steps and guidance for fine control
            - 'flux-2-max': Highest quality with grounding search
            - 'flux-pro-1.1': Fast and reliable baseline
            - 'flux-pro-1.1-ultra': High resolution variant
            - 'flux-dev': Free development model (non-commercial)
        width: Output width in pixels (multiple of 16). Default 1024.
        height: Output height in pixels (multiple of 16). Default 1024.
        seed: Random seed for reproducible results.
        safety_tolerance: Moderation level from 0 (strict) to 5 (permissive). Default 2.
        output_format: Image format, 'jpeg' or 'png'. Default 'jpeg'.
        steps: Number of inference steps (flux-2-flex only, max 50).
        guidance: Guidance scale 1.5-10 (flux-2-flex only). Default 4.5.

    Returns:
        A generated PIL Image.

    Examples:
        Add a computed column to generate images from prompts:

        >>> t.add_computed_column(
        ...     image=bfl.generate(t.prompt, model='flux-2-pro', width=1920, height=1080)
        ... )

        Generate a square image with specific seed for reproducibility:

        >>> t.add_computed_column(
        ...     image=bfl.generate(t.prompt, seed=42, width=1024, height=1024)
        ... )
    """
    if model not in _MODEL_ENDPOINTS:
        raise pxt.Error(f'Unknown model: {model}. Available: {list(_MODEL_ENDPOINTS.keys())}')

    # Only text-to-image models
    if model in ('flux-kontext-pro', 'flux-kontext-max'):
        raise pxt.Error(f'Model {model} is for image editing. Use bfl.edit() instead.')

    endpoint = _MODEL_ENDPOINTS[model]
    payload: dict = {'prompt': prompt}

    if width is not None:
        payload['width'] = width
    if height is not None:
        payload['height'] = height
    if seed is not None:
        payload['seed'] = seed
    if safety_tolerance is not None:
        payload['safety_tolerance'] = safety_tolerance
    if output_format is not None:
        payload['output_format'] = output_format

    # flux-2-flex specific parameters
    if model == 'flux-2-flex':
        if steps is not None:
            payload['steps'] = steps
        if guidance is not None:
            payload['guidance'] = guidance
    elif steps is not None or guidance is not None:
        _logger.warning(f'Parameters steps/guidance are only supported for flux-2-flex, ignoring for {model}')

    return await _client().generate(endpoint, payload)


@pxt.udf(resource_pool='request-rate:bfl')
async def edit(
    prompt: str,
    image: PIL.Image.Image,
    *,
    model: Literal['flux-2-pro', 'flux-2-flex', 'flux-kontext-pro', 'flux-kontext-max'] = 'flux-2-pro',
    reference_images: list[PIL.Image.Image] | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    safety_tolerance: int | None = None,
    output_format: Literal['jpeg', 'png'] | None = None,
    steps: int | None = None,
    guidance: float | None = None,
) -> PIL.Image.Image:
    """
    Edit an image using FLUX models with text prompts and optional reference images.

    This UDF wraps the BFL FLUX image editing API. For more information, refer to the official
    [API documentation](https://docs.bfl.ai/flux_2/flux2_image_editing).

    Args:
        prompt: Text description of the edit to apply.
        image: The base image to edit.
        model: FLUX model to use for editing. Options:
            - 'flux-2-pro': Best balance of speed and quality (default)
            - 'flux-2-flex': Adjustable steps and guidance
            - 'flux-kontext-pro': Specialized for context-aware editing
            - 'flux-kontext-max': Highest quality context-aware editing
        reference_images: Additional reference images (up to 7) for multi-reference editing.
        width: Output width in pixels (multiple of 16). Matches input if not specified.
        height: Output height in pixels (multiple of 16). Matches input if not specified.
        seed: Random seed for reproducible results.
        safety_tolerance: Moderation level from 0 (strict) to 5 (permissive). Default 2.
        output_format: Image format, 'jpeg' or 'png'. Default 'jpeg'.
        steps: Number of inference steps (flux-2-flex only, max 50).
        guidance: Guidance scale 1.5-10 (flux-2-flex only). Default 4.5.

    Returns:
        An edited PIL Image.

    Examples:
        Edit an image to change its background:

        >>> t.add_computed_column(
        ...     edited=bfl.edit(
        ...         'Change the background to a sunset beach',
        ...         t.original_image
        ...     )
        ... )

        Multi-reference editing with additional images:

        >>> t.add_computed_column(
        ...     edited=bfl.edit(
        ...         'Combine the person from the first image with the background from the second',
        ...         t.person_image,
        ...         reference_images=[t.background_image]
        ...     )
        ... )
    """
    valid_edit_models = ('flux-2-pro', 'flux-2-flex', 'flux-kontext-pro', 'flux-kontext-max')
    if model not in valid_edit_models:
        raise pxt.Error(f'Model {model} not supported for editing. Use one of: {valid_edit_models}')

    endpoint = _MODEL_ENDPOINTS[model]
    payload: dict = {'prompt': prompt, 'input_image': to_base64(image)}

    # Add reference images if provided
    if reference_images:
        if len(reference_images) > 7:
            raise pxt.Error('Maximum 7 additional reference images allowed (8 total including input_image)')
        for i, ref_img in enumerate(reference_images, start=2):
            payload[f'input_image_{i}'] = to_base64(ref_img)

    if width is not None:
        payload['width'] = width
    if height is not None:
        payload['height'] = height
    if seed is not None:
        payload['seed'] = seed
    if safety_tolerance is not None:
        payload['safety_tolerance'] = safety_tolerance
    if output_format is not None:
        payload['output_format'] = output_format

    # flux-2-flex specific parameters
    if model == 'flux-2-flex':
        if steps is not None:
            payload['steps'] = steps
        if guidance is not None:
            payload['guidance'] = guidance
    elif steps is not None or guidance is not None:
        _logger.warning(f'Parameters steps/guidance are only supported for flux-2-flex, ignoring for {model}')

    return await _client().generate(endpoint, payload)


@pxt.udf(resource_pool='request-rate:bfl')
async def fill(
    prompt: str,
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    *,
    seed: int | None = None,
    safety_tolerance: int | None = None,
    output_format: Literal['jpeg', 'png'] | None = None,
) -> PIL.Image.Image:
    """
    Inpaint an image using FLUX.1 Fill [pro].

    Fill specified areas of an image based on a mask and text prompt. The mask can be
    a separate image or applied to the alpha channel of the input image.

    This UDF wraps the BFL FLUX Fill API. For more information, refer to the official
    [API documentation](https://docs.bfl.ai/flux_tools/flux_1_fill).

    Args:
        prompt: Text description of what to fill in the masked area.
        image: The base image to inpaint.
        mask: Mask image where white areas indicate regions to fill.
        seed: Random seed for reproducible results.
        safety_tolerance: Moderation level from 0 (strict) to 5 (permissive). Default 2.
        output_format: Image format, 'jpeg' or 'png'. Default 'jpeg'.

    Returns:
        An inpainted PIL Image.

    Examples:
        Fill a masked region with generated content:

        >>> t.add_computed_column(
        ...     filled=bfl.fill(
        ...         'A beautiful garden with flowers',
        ...         t.original_image,
        ...         t.mask_image
        ...     )
        ... )
    """
    payload: dict = {'prompt': prompt, 'image': to_base64(image), 'mask': to_base64(mask)}

    if seed is not None:
        payload['seed'] = seed
    if safety_tolerance is not None:
        payload['safety_tolerance'] = safety_tolerance
    if output_format is not None:
        payload['output_format'] = output_format

    return await _client().generate('/v1/flux-pro-1.0-fill', payload)


@pxt.udf(resource_pool='request-rate:bfl')
async def expand(
    prompt: str,
    image: PIL.Image.Image,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    seed: int | None = None,
    safety_tolerance: int | None = None,
    output_format: Literal['jpeg', 'png'] | None = None,
) -> PIL.Image.Image:
    """
    Expand an image by adding pixels on any side using FLUX.1 Expand [pro].

    Outpaint an image by specifying how many pixels to add to each edge.
    The expansion maintains context from the original image.

    This UDF wraps the BFL FLUX Expand API. For more information, refer to the official
    [API documentation](https://docs.bfl.ai/flux_tools/flux_1_expand).

    Args:
        prompt: Text description to guide the expansion.
        image: The base image to expand.
        top: Pixels to add to the top edge.
        bottom: Pixels to add to the bottom edge.
        left: Pixels to add to the left edge.
        right: Pixels to add to the right edge.
        seed: Random seed for reproducible results.
        safety_tolerance: Moderation level from 0 (strict) to 5 (permissive). Default 2.
        output_format: Image format, 'jpeg' or 'png'. Default 'jpeg'.

    Returns:
        An expanded PIL Image.

    Examples:
        Expand an image to create a wider landscape:

        >>> t.add_computed_column(
        ...     wide=bfl.expand(
        ...         'Continue the landscape scenery',
        ...         t.original_image,
        ...         left=256,
        ...         right=256
        ...     )
        ... )
    """
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        raise pxt.Error('At least one expansion direction (top, bottom, left, right) must be > 0')

    payload: dict = {
        'prompt': prompt,
        'image': to_base64(image),
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
    }

    if seed is not None:
        payload['seed'] = seed
    if safety_tolerance is not None:
        payload['safety_tolerance'] = safety_tolerance
    if output_format is not None:
        payload['output_format'] = output_format

    return await _client().generate('/v1/flux-pro-1.0-expand', payload)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
