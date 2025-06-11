"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-genai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://pixeltable.readme.io/docs/working-with-gemini) tutorial.
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import PIL.Image

import pixeltable as pxt
from pixeltable import env, exceptions as excs, exprs

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
    contents: str, *, model: str, config: Optional[dict] = None, tools: Optional[list[dict]] = None
) -> dict:
    """
    Generate content from the specified model. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/text-generation>

    Request throttling:
    Applies the rate limit set in the config (section `gemini`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The input content to generate from.
        model: The name of the model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateContentConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#module-genai.types>
        tools: An optional list of Pixeltable tools to use. It is also possible to specify tools manually via the
            `config['tools']` parameter, but at most one of `config['tools']` or `tools` may be used.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gemini-2.0-flash`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_content(tbl.prompt, model='gemini-2.0-flash'))
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
def _gemini_response_to_pxt_tool_calls(response: dict) -> Optional[dict]:
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
async def generate_images(prompt: str, *, model: str, config: Optional[dict] = None) -> PIL.Image.Image:
    """
    Generates images based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/image-generation>

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the images to generate.
        model: The model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateImagesConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#module-genai.types>

    Returns:
        The generated image.

    Examples:
        Add a computed column that applies the model `imagen-3.0-generate-002`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_images(tbl.prompt, model='imagen-3.0-generate-002'))
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
    prompt: Optional[str] = None, image: Optional[PIL.Image.Image] = None, *, model: str, config: Optional[dict] = None
) -> pxt.Video:
    """
    Generates videos based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/video-generation>

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the videos to generate.
        image: An optional image to use as the first frame of the video. At least one of `prompt` or `image` must be
            provided. (It is ok to specify both.)
        model: The model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateVideosConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#module-genai.types>

    Returns:
        The generated video.

    Examples:
        Add a computed column that applies the model `veo-2.0-generate-001`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=generate_videos(tbl.prompt, model='veo-2.0-generate-001'))
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    if prompt is None and image is None:
        raise excs.Error('At least one of `prompt` or `image` must be provided.')

    image_: Optional[types.Image] = None
    if image is not None:
        with io.BytesIO() as buffer:
            image.save(buffer, format='jpeg')
            image_ = types.Image(image_bytes=buffer.getvalue(), mime_type='image/jpeg')

    config_ = types.GenerateVideosConfig(**config) if config else None
    operation = await _genai_client().aio.models.generate_videos(
        model=model, prompt=prompt, image=image_, config=config_
    )
    while not operation.done:
        await asyncio.sleep(3)
        operation = await _genai_client().aio.operations.get(operation)

    video = operation.response.generated_videos[0]

    video_bytes = await _genai_client().aio.files.download(file=video.video)  # type: ignore[arg-type]
    assert video_bytes is not None

    _, output_filename = tempfile.mkstemp(suffix='.mp4', dir=str(env.Env.get().tmp_dir))
    Path(output_filename).write_bytes(video_bytes)
    return output_filename


@generate_videos.resource_pool
def _(model: str) -> str:
    return f'request-rate:veo:{model}'
