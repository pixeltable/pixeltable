"""
Pixeltable UDFs
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-genai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://docs.pixeltable.com/notebooks/integrations/working-with-gemini) tutorial.
"""

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING

import PIL.Image

import pixeltable as pxt
from pixeltable import env, exceptions as excs, exprs
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from google import genai


@env.register_client('gemini')
def _(api_key: str) -> 'genai.client.Client':
    from google import genai

    return genai.client.Client(api_key=api_key)


def _genai_client() -> 'genai.client.Client':
    return env.Env.get().get_client('gemini')


@pxt.udf(resource_pool='request-rate:gemini')
async def generate_content(
    contents: pxt.Json, *, model: str, config: dict | None = None, tools: list[dict] | None = None
) -> dict:
    """
    Generate content from the specified model. `contents` can be a prompt, or a list containing images and text
    prompts, as described in: <https://ai.google.dev/gemini-api/docs/text-generation>

    Request throttling:
    Applies the rate limit set in the config (section `gemini.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The input content to generate from.
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

        >>> tbl.add_computed_column(response=generate_content(tbl.prompt, model='gemini-2.5-flash'))

        Add a computed column that applies the model `gemini-2.5-flash` for image understanding
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

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

    response = await _genai_client().aio.models.generate_content(model=model, contents=contents, config=config_)
    return response.model_dump()


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
    return f'request-rate:gemini:{model}'


@pxt.udf(resource_pool='request-rate:imagen')
async def generate_images(prompt: str, *, model: str, config: dict | None = None) -> PIL.Image.Image:
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

        >>> tbl.add_computed_column(response=generate_images(tbl.prompt, model='imagen-4.0-generate-001'))
    """
    env.Env.get().require_package('google.genai')
    from google.genai.types import GenerateImagesConfig

    config_ = GenerateImagesConfig(**config) if config else None
    response = await _genai_client().aio.models.generate_images(model=model, prompt=prompt, config=config_)
    return response.generated_images[0].image._pil_image


@generate_images.resource_pool
def _(model: str) -> str:
    return f'request-rate:imagen:{model}'


@pxt.udf(resource_pool='request-rate:veo')
async def generate_videos(
    prompt: str | None = None,
    image: PIL.Image.Image | None = None,
    *,
    model: str,
    last_frame: PIL.Image.Image | None = None,
    reference_images: list | None = None,
    config: dict | None = None,
) -> pxt.Video:
    """
    Generates videos based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/video>

    At least one of `prompt`, `image`, or `video` must be provided.

    Request throttling:
    Applies the rate limit set in the config (section `veo.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the videos to generate.
        image: An image to use as the first frame of the video.
        video: A video to use as the starting point for generation.
        model: The model to use.
        last_frame: An optional image to use as the last frame of the video.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateVideosConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig>

    Returns:
        The generated video.

    Examples:
        Add a computed column that applies the model `veo-3.0-generate-001`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_videos(tbl.prompt, model='veo-3.0-generate-001'))
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    if prompt is None and image is None:
        raise excs.Error('At least one of `prompt`, `image`, or `video` must be provided.')
    if image is None and last_frame is not None:
        raise excs.Error('If `last_frame` is provided, `image` must also be provided.')

    image_: types.Image | None = None
    if image is not None:
        with io.BytesIO() as buffer:
            image.save(buffer, format='webp')
            image_ = types.Image(image_bytes=buffer.getvalue(), mime_type='image/webp')

    config_ = types.GenerateVideosConfig(**config) if config else None

    if last_frame is not None:
        with io.BytesIO() as buffer:
            last_frame.save(buffer, format='webp')
            last_frame_ = types.Image(image_bytes=buffer.getvalue(), mime_type='image/webp')
            if config_ is None:
                config_ = types.GenerateVideosConfig()
            config_.last_frame = last_frame_

    if reference_images is not None:
        reference_images_: list[types.VideoGenerationReferenceImage] = []
        for img_dict in reference_images:
            if (
                not isinstance(img_dict, dict)
                or len(img_dict) == 0
                or 'image' not in img_dict
                or not set(img_dict.keys()).issubset({'image', 'reference_type'})
            ):
                raise excs.Error(
                    "Each element of `reference_images` must be a dict with an 'image' key "
                    "and an optional 'reference_type' key."
                )
            img = img_dict['image']
            ref_type = img_dict.get('reference_type')
            with io.BytesIO() as buffer:
                img.save(buffer, format='webp')
                reference_images_.append(
                    types.VideoGenerationReferenceImage(
                        image=types.Image(image_bytes=buffer.getvalue(), mime_type='image/webp'),
                        reference_type=ref_type,
                    )
                )
        if config_ is None:
            config_ = types.GenerateVideosConfig()
        config_.reference_images = reference_images_

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
    return f'request-rate:veo:{model}'


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
