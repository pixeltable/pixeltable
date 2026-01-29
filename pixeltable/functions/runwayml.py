"""
Pixeltable UDFs
that wrap various endpoints from the RunwayML API. In order to use them, you must
first `pip install runwayml` and configure your RunwayML credentials by setting the `RUNWAYML_API_SECRET` environment
variable.
"""

import datetime
from typing import TYPE_CHECKING, Any

import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    from runwayml import AsyncRunwayML


@register_client('runwayml')
def _(api_secret: str) -> 'AsyncRunwayML':
    from runwayml import AsyncRunwayML

    return AsyncRunwayML(api_key=api_secret)


def _runwayml_client() -> 'AsyncRunwayML':
    return Env.get().get_client('runwayml')


def _image_to_data_uri(image: PIL.Image.Image) -> str:
    """Convert a PIL Image to a data URI suitable for RunwayML API."""
    fmt = 'png' if image.has_transparency_data else 'webp'
    b64 = to_base64(image, format=fmt)
    return f'data:image/{fmt};base64,{b64}'


def _serialize_result(obj: Any) -> Any:
    """Convert RunwayML result to JSON-serializable format.

    Handles datetime objects and nested structures that may not be JSON serializable.
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_result(item) for item in obj]
    return obj


@pxt.udf(resource_pool='request-rate:runwayml')
async def text_to_image(
    prompt_text: str,
    reference_images: list[PIL.Image.Image],
    model: str,
    ratio: str,
    *,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate images from text prompts and reference images.

    For additional details, see: [Text/Image to Image](https://docs.dev.runwayml.com/api/#tag/Start-generating/paths/~1v1~1text_to_image/post)

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_text: Text description of the image to generate.
        reference_images: List of 1-3 reference images.
        model: The model to use.
        ratio: Aspect ratio of the generated image.
        seed: Seed for reproducibility.
        model_kwargs: Additional API parameters.

    Returns:
        A dictionary containing the response and metadata.

    Examples:
        Add a computed column that generates images from prompts:

        >>> tbl.add_computed_column(
        ...     response=text_to_image(tbl.prompt, [tbl.ref_image], model='gen4_image', ratio='16:9')
        ... )
        >>> tbl.add_computed_column(image=tbl.response['output'][0].astype(pxt.Image))
    """
    Env.get().require_package('runwayml')
    client = _runwayml_client()

    # Convert reference images to data URIs
    ref_images = [{'uri': _image_to_data_uri(img)} for img in reference_images]

    kwargs: dict[str, Any] = {
        'model': model,
        'prompt_text': prompt_text,
        'ratio': ratio,
        'reference_images': ref_images,
    }

    if seed is not None:
        kwargs['seed'] = seed
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.text_to_image.create(**kwargs)
    result = await task.wait_for_task_output()
    if result.status != 'SUCCEEDED':
        raise pxt.Error(f'RunwayML task failed with status: {result.status}')
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool='request-rate:runwayml')
async def text_to_video(
    prompt_text: str,
    model: str,
    ratio: str,
    *,
    duration: int | None = None,
    audio: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate videos from text prompts.

    For additional details, see: [Text to video](https://docs.dev.runwayml.com/api/#tag/Start-generating/paths/~1v1~1text_to_video/post)

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_text: Text description of the video to generate.
        model: The model to use.
        ratio: Aspect ratio of the generated video.
        duration: Duration in seconds.
        audio: Whether to generate audio.
        model_kwargs: Additional API parameters.

    Returns:
        A dictionary containing the response and metadata.

    Examples:
        Add a computed column that generates videos from prompts:

        >>> tbl.add_computed_column(response=text_to_video(tbl.prompt, model='veo3.1', ratio='16:9', duration=4))
        >>> tbl.add_computed_column(video=tbl.response['output'].astype(pxt.Video))
    """
    Env.get().require_package('runwayml')
    client = _runwayml_client()

    kwargs: dict[str, Any] = {'model': model, 'prompt_text': prompt_text, 'ratio': ratio}

    if duration is not None:
        kwargs['duration'] = duration
    if audio is not None:
        kwargs['audio'] = audio
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.text_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    if result.status != 'SUCCEEDED':
        raise pxt.Error(f'RunwayML task failed with status: {result.status}')
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool='request-rate:runwayml')
async def image_to_video(
    prompt_image: PIL.Image.Image,
    model: str,
    ratio: str,
    *,
    prompt_text: str | None = None,
    duration: int | None = None,
    seed: int | None = None,
    audio: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate videos from images.

    For additional details, see: [Image to video](https://docs.dev.runwayml.com/api/#tag/Start-generating/paths/~1v1~1image_to_video/post)

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_image: Input image to use as the first frame.
        model: The model to use.
        ratio: Aspect ratio of the generated video.
        prompt_text: Text description to guide generation.
        duration: Duration in seconds.
        seed: Seed for reproducibility.
        audio: Whether to generate audio.
        model_kwargs: Additional API parameters.

    Returns:
        A dictionary containing the response and metadata.

    Examples:
        Add a computed column that generates videos from images:

        >>> tbl.add_computed_column(
        ...     response=image_to_video(tbl.image, model='gen4', ratio='16:9', prompt_text='Slow motion', duration=5)
        ... )
        >>> tbl.add_computed_column(video=tbl.response['output'].astype(pxt.Video))
    """
    Env.get().require_package('runwayml')
    client = _runwayml_client()

    image_uri = _image_to_data_uri(prompt_image)
    kwargs: dict[str, Any] = {'model': model, 'prompt_image': image_uri, 'ratio': ratio}

    if prompt_text is not None:
        kwargs['prompt_text'] = prompt_text
    if duration is not None:
        kwargs['duration'] = duration
    if seed is not None:
        kwargs['seed'] = seed
    if audio is not None:
        kwargs['audio'] = audio
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.image_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    if result.status != 'SUCCEEDED':
        raise pxt.Error(f'RunwayML task failed with status: {result.status}')
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool='request-rate:runwayml')
async def video_to_video(
    video_uri: str,
    prompt_text: str,
    model: str,
    ratio: str,
    *,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Transform videos with text guidance.

    For additional details, see: [Video to video](https://docs.dev.runwayml.com/api/#tag/Start-generating/paths/~1v1~1video_to_video/post)

    __Requirements:__

    - `pip install runwayml`

    Args:
        video_uri: HTTPS URL to the input video.
        prompt_text: Text description of the transformation.
        model: The model to use.
        ratio: Aspect ratio of the output video.
        seed: Seed for reproducibility.
        model_kwargs: Additional API parameters.

    Returns:
        A dictionary containing the response and metadata.

    Examples:
        Add a computed column that transforms videos:

        >>> tbl.add_computed_column(
        ...     response=video_to_video(tbl.video_url, 'Anime style', model='gen4_aleph', ratio='16:9')
        ... )
        >>> tbl.add_computed_column(video=tbl.response['output'].astype(pxt.Video))
    """
    Env.get().require_package('runwayml')
    client = _runwayml_client()

    kwargs: dict[str, Any] = {'model': model, 'video_uri': video_uri, 'prompt_text': prompt_text, 'ratio': ratio}

    if seed is not None:
        kwargs['seed'] = seed
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.video_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    if result.status != 'SUCCEEDED':
        raise pxt.Error(f'RunwayML task failed with status: {result.status}')
    return _serialize_result(result.to_dict())


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
