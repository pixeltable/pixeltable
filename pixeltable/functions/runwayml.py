"""
Pixeltable UDFs that wrap various endpoints from the RunwayML API.

In order to use them, you must first `pip install runwayml` and configure your
RunwayML API key. You can set your API key either:

1. As an environment variable: `RUNWAYML_API_KEY`
2. In `~/.pixeltable/config.toml` under the `[runwayml]` section:
   ```toml
   [runwayml]
   api_key = "your-api-key"
   ```

For more information, see: https://docs.dev.runwayml.com/
"""

import datetime
from typing import TYPE_CHECKING, Any, Literal

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from runwayml import AsyncRunwayML  # type: ignore[import-untyped,import-not-found,unused-ignore]


@register_client("runwayml")
def _(api_key: str) -> "AsyncRunwayML":
    from runwayml import AsyncRunwayML  # type: ignore[import-untyped,import-not-found,unused-ignore]

    return AsyncRunwayML(api_key=api_key)


def _runwayml_client() -> "AsyncRunwayML":
    return Env.get().get_client("runwayml")


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


# Type aliases for supported models and ratios
TextToImageModel = Literal["gen4_image", "gen4_image_turbo", "gemini_2.5_flash"]
TextToVideoModel = Literal["veo3.1", "veo3.1_fast", "veo3"]
ImageToVideoModel = Literal[
    "gen4_turbo", "gen3a_turbo", "veo3.1", "veo3.1_fast", "veo3"
]
VideoToVideoModel = Literal["gen4_aleph"]

ImageRatio = Literal[
    "1024:1024",
    "1080:1080",
    "1168:880",
    "1360:768",
    "1440:1080",
    "1080:1440",
    "1808:768",
    "1920:1080",
    "1080:1920",
    "2112:912",
    "1280:720",
    "720:1280",
    "720:720",
    "960:720",
    "720:960",
    "1680:720",
]
VideoRatio = Literal["1280:720", "720:1280", "1080:1920", "1920:1080"]


@pxt.udf(resource_pool="request-rate:runwayml")
async def text_to_image(
    prompt_text: str,
    reference_images: list[str],
    *,
    model: str = "gen4_image",
    ratio: str = "1920:1080",
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate images from text prompts and reference images using RunwayML models.

    This function starts a text-to-image generation task and waits for it to complete.
    The task is queued and processed asynchronously by RunwayML.

    Note: The RunwayML text_to_image endpoint requires at least one reference image.
    To guide the generation, provide 1-3 reference images as HTTPS URLs.

    Request throttling:
    Applies the rate limit set in the config (section `runwayml`, key `rate_limit`).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_text: A text description of the image to generate (up to 1000 characters).
        reference_images: A list of 1-3 HTTPS URLs pointing to reference images.
            These images guide the style or content of the generated output.
        model: The RunwayML model to use. Options: 'gen4_image', 'gen4_image_turbo', 'gemini_2.5_flash'.
            Default: 'gen4_image'.
        ratio: The aspect ratio of the generated image. Common options include:
            '1920:1080', '1080:1920', '1024:1024', '1280:720', '720:1280', etc.
            Default: '1920:1080'.
        seed: Optional seed for reproducible results. If not specified, a random seed is used.
        model_kwargs: Additional model-specific parameters passed to the API.

    Returns:
        A JSON object containing the task result with generated image URLs and metadata.
        The output images are available at `result['output']` as a list of URLs.

    Examples:
        Add a computed column that generates images from prompts with a reference image:

        >>> tbl.add_computed_column(
        ...     response=text_to_image(
        ...         tbl.prompt,
        ...         [tbl.reference_image_url],
        ...         model='gen4_image',
        ...         ratio='1920:1080'
        ...     )
        ... )
        >>> tbl.add_computed_column(
        ...     image=tbl.response['output'][0].astype(pxt.Image)
        ... )
    """
    Env.get().require_package("runwayml")
    client = _runwayml_client()

    # Convert reference images to the format expected by the API
    ref_images = [{"uri": url} for url in reference_images]

    kwargs: dict[str, Any] = {
        "model": model,
        "prompt_text": prompt_text,
        "ratio": ratio,
        "reference_images": ref_images,
    }

    if seed is not None:
        kwargs["seed"] = seed
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.text_to_image.create(**kwargs)
    result = await task.wait_for_task_output()
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool="request-rate:runwayml")
async def text_to_video(
    prompt_text: str,
    *,
    model: str = "veo3.1",
    ratio: str = "1280:720",
    duration: int | None = None,
    audio: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate videos from text prompts using RunwayML models.

    This function starts a text-to-video generation task and waits for it to complete.
    The task is queued and processed asynchronously by RunwayML.

    Request throttling:
    Applies the rate limit set in the config (section `runwayml`, key `rate_limit`).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_text: A text description of the video to generate (up to 1000 characters).
        model: The RunwayML model to use. Options: 'veo3.1', 'veo3.1_fast', 'veo3'.
            Default: 'veo3.1'.
        ratio: The aspect ratio of the generated video. Options:
            '1280:720', '720:1280', '1080:1920', '1920:1080'.
            Default: '1280:720'.
        duration: Duration of the video in seconds. Valid values depend on the model:
            - veo3.1/veo3.1_fast: 4, 6, or 8 seconds
            - veo3: 8 seconds (fixed)
        audio: Whether to generate audio for the video. Audio inclusion affects pricing.
        model_kwargs: Additional model-specific parameters passed to the API.

    Returns:
        A JSON object containing the task result with the generated video URL and metadata.
        The output video URL is available at `result['output']`.

    Examples:
        Add a computed column that generates videos from prompts:

        >>> tbl.add_computed_column(
        ...     response=text_to_video(tbl.prompt, model='veo3.1', duration=4)
        ... )
        >>> tbl.add_computed_column(
        ...     video=tbl.response['output'].astype(pxt.Video)
        ... )
    """
    Env.get().require_package("runwayml")
    client = _runwayml_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "prompt_text": prompt_text,
        "ratio": ratio,
    }

    if duration is not None:
        kwargs["duration"] = duration
    if audio is not None:
        kwargs["audio"] = audio
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.text_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool="request-rate:runwayml")
async def image_to_video(
    prompt_image: str,
    *,
    model: str = "gen4_turbo",
    ratio: str = "1280:720",
    prompt_text: str | None = None,
    duration: int | None = None,
    seed: int | None = None,
    audio: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Generate videos from images using RunwayML models.

    This function starts an image-to-video generation task and waits for it to complete.
    The input image is used as the first frame, and the model generates the subsequent frames.

    Request throttling:
    Applies the rate limit set in the config (section `runwayml`, key `rate_limit`).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install runwayml`

    Args:
        prompt_image: A HTTPS URL to the input image to use as the first frame.
        model: The RunwayML model to use. Options:
            'gen4_turbo', 'gen3a_turbo', 'veo3.1', 'veo3.1_fast', 'veo3'.
            Default: 'gen4_turbo'.
        ratio: The aspect ratio of the generated video. Options depend on the model.
            Default: '1280:720'.
        prompt_text: Optional text description to guide the video generation
            (up to 1000 characters).
        duration: Duration of the video in seconds. Valid values depend on the model:
            - gen4_turbo: variable (2-10 seconds typically)
            - gen3a_turbo: 5 or 10 seconds
            - veo3.1/veo3.1_fast: 4, 6, or 8 seconds
            - veo3: 8 seconds (fixed)
        seed: Optional seed for reproducible results.
        audio: Whether to generate audio for the video (for veo models).
        model_kwargs: Additional model-specific parameters passed to the API.

    Returns:
        A JSON object containing the task result with the generated video URL and metadata.

    Examples:
        Add a computed column that generates videos from images:

        >>> tbl.add_computed_column(
        ...     response=image_to_video(
        ...         tbl.image_url,
        ...         prompt_text='A person walking slowly',
        ...         model='gen4_turbo',
        ...         duration=5
        ...     )
        ... )
        >>> tbl.add_computed_column(
        ...     video=tbl.response['output'].astype(pxt.Video)
        ... )
    """
    Env.get().require_package("runwayml")
    client = _runwayml_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "prompt_image": prompt_image,
        "ratio": ratio,
    }

    if prompt_text is not None:
        kwargs["prompt_text"] = prompt_text
    if duration is not None:
        kwargs["duration"] = duration
    if seed is not None:
        kwargs["seed"] = seed
    if audio is not None:
        kwargs["audio"] = audio
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.image_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    return _serialize_result(result.to_dict())


@pxt.udf(resource_pool="request-rate:runwayml")
async def video_to_video(
    video_uri: str,
    prompt_text: str,
    *,
    model: str = "gen4_aleph",
    ratio: str = "1280:720",
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Transform videos using RunwayML models.

    This function starts a video-to-video transformation task and waits for it to complete.
    The input video is transformed based on the text prompt.

    Request throttling:
    Applies the rate limit set in the config (section `runwayml`, key `rate_limit`).
    If no rate limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install runwayml`

    Args:
        video_uri: A HTTPS URL to the input video.
        prompt_text: A text description of how to transform the video (up to 1000 characters).
        model: The RunwayML model to use. Currently only 'gen4_aleph' is supported.
            Default: 'gen4_aleph'.
        ratio: The aspect ratio of the output video. Options:
            '1280:720', '720:1280', '1104:832', '960:960', '832:1104', '1584:672',
            '848:480', '640:480'.
            Default: '1280:720'.
        seed: Optional seed for reproducible results.
        model_kwargs: Additional model-specific parameters passed to the API.

    Returns:
        A JSON object containing the task result with the transformed video URL and metadata.

    Examples:
        Add a computed column that transforms videos:

        >>> tbl.add_computed_column(
        ...     response=video_to_video(
        ...         tbl.video_url,
        ...         prompt_text='Transform to anime style',
        ...         model='gen4_aleph'
        ...     )
        ... )
        >>> tbl.add_computed_column(
        ...     video=tbl.response['output'].astype(pxt.Video)
        ... )
    """
    Env.get().require_package("runwayml")
    client = _runwayml_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "video_uri": video_uri,
        "prompt_text": prompt_text,
        "ratio": ratio,
    }

    if seed is not None:
        kwargs["seed"] = seed
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    task = await client.video_to_video.create(**kwargs)
    result = await task.wait_for_task_output()
    return _serialize_result(result.to_dict())


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
