"""
Pixeltable UDFs
that wrap various endpoints from the Google Gemini API. In order to use them, you must
first `pip install google-genai` and configure your Gemini credentials, as described in
the [Working with Gemini](https://docs.pixeltable.com/howto/providers/working-with-gemini) tutorial.

Supports two authentication methods:

- Google AI Studio: set `GOOGLE_API_KEY` or `GEMINI_API_KEY` (or put `api_key` in the `gemini` section of
  the Pixeltable config file).
- Vertex AI: set `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_PROJECT` (and optionally
  `GOOGLE_CLOUD_LOCATION`), then authenticate via Application Default Credentials
  (e.g. `gcloud auth application-default login`).
"""

import asyncio
import base64
import io
import logging
import mimetypes
import os
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Literal, Sequence, TypeVar

import numpy as np
import PIL.Image
from tenacity import RetryCallState, retry, retry_if_result, stop_after_delay, wait_exponential

import pixeltable as pxt
from pixeltable import env, exceptions as excs, exprs, type_system as ts
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import exponential_backoff, parse_duration_str
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from google import genai

_logger = logging.getLogger('pixeltable')

# Max raw file size (bytes) for inline data; larger files use the Files API.
GEMINI_INLINE_LIMIT_BYTES = 4 * 2**20

# Placeholder key used in first pass for large file uploads.
_UPLOAD_PLACEHOLDER_KEY = '__google_genai_upload_ref__'

# Used to generalize polling logic for long-running Gemini operations (e.g. video generation).
T = TypeVar('T', bound='genai.types.Operation')


@env.register_client('gemini')
def _(api_key: str | None = None) -> 'genai.client.Client':
    from google import genai

    try:
        if api_key is not None:
            return genai.client.Client(api_key=api_key)
        # Vertex AI fall-through: rely on genai.client.Client to read its own env vars
        # (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION)
        return genai.client.Client()
    except Exception as e:
        raise excs.AuthorizationError(
            excs.ErrorCode.MISSING_CREDENTIALS,
            'Gemini client not initialized. '
            'For the Gemini Developer API set GOOGLE_API_KEY or GEMINI_API_KEY, '
            'or set api_key in the [gemini] section of $PIXELTABLE_HOME/config.toml. '
            'For Vertex AI set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT, '
            'then authenticate via: gcloud auth application-default login',
        ) from e


def _genai_client() -> 'genai.client.Client':
    return get_runtime().get_client('gemini')


class GeminiRateLimitsInfo(env.RateLimitsInfo):
    # TODO(PXT-996): Improve resource tracking for Gemini UDFs
    def is_initialized(self) -> bool:
        return True

    def get_retry_delay(self, exc: Exception, attempt: int) -> float | None:
        if hasattr(exc, 'code') and exc.code == 429:
            try:
                for detail_dict in exc.details['error']['details']:  # type: ignore[attr-defined]
                    if detail_dict.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        delay = parse_duration_str(detail_dict['retryDelay'])
                        return delay
            except (AttributeError, KeyError, TypeError):
                return exponential_backoff(attempt)
        return None


@pxt.udf(is_deterministic=False)
async def generate_content(
    contents: pxt.Json, *, model: str, config: dict | None = None, tools: list[dict] | None = None
) -> dict:
    """
    Generate content from the specified model.

    Request throttling:
    Applies the rate limit set in the config (section `gemini.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The input content to generate from. Can be a prompt, or a list containing images and text
            prompts, as described in: <https://ai.google.dev/gemini-api/docs/text-generation> and
            <https://ai.google.dev/gemini-api/docs/image-generation> for image generation.
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

        >>> tbl.add_computed_column(
        ...     response=generate_content(tbl.prompt, model='gemini-2.5-flash')
        ... )

        Generate an image with a Nano Banana model (Gemini image-generation models such as
        `gemini-2.5-flash-image`) and extract the PIL image from the response using JSON
        subscripting. Image bytes in `inline_data.data` are decoded into PIL images
        automatically. Pass `response_modalities=['IMAGE']` so the response contains a
        single image part:

        >>> tbl.add_computed_column(
        ...     response=generate_content(
        ...         tbl.prompt,
        ...         model='gemini-2.5-flash-image',
        ...         config={'response_modalities': ['IMAGE']},
        ...     )
        ... )
        >>> tbl.add_computed_column(
        ...     image=tbl.response.candidates[0].content.parts[0].inline_data.data
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

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

    large_video_paths: list[str] = []
    client = _genai_client()

    contents = _process_media_contents(contents, large_video_paths)
    async with _gemini_file_uploads(large_video_paths) as uploaded:
        contents = _replace_upload_placeholders(contents, uploaded)
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config_)
        result = response.model_dump(mode='json')
        for candidate, result_candidate in zip(response.candidates or [], result.get('candidates', [])):
            if candidate.content is None:
                continue
            for part, result_part in zip(
                candidate.content.parts or [], result_candidate.get('content', {}).get('parts', [])
            ):
                blob = part.inline_data
                if blob is not None and blob.mime_type and blob.mime_type.startswith('image/') and blob.data:
                    result_part['inline_data']['data'] = PIL.Image.open(io.BytesIO(blob.data))
        return result


@asynccontextmanager
async def _gemini_file_uploads(files: list[str]) -> AsyncIterator[list['genai.types.File']]:
    """
    Context manager that makes uploaded files temporarily available to Gemini models, deleting them from the server
    after use.
    """
    client = _genai_client()
    uploaded: list['genai.types.File'] = []

    try:
        if len(files) > 0:
            tasks: list[Coroutine[Any, Any, 'genai.types.File']] = []
            for file in files:
                mime_type, _ = mimetypes.guess_type(file, strict=False)
                if mime_type is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT, f'Could not identify mime type of file: {file}'
                    )
                tasks.append(client.aio.files.upload(file=file, config={'mime_type': mime_type}))
            uploaded = await asyncio.gather(*tasks)
            # poll till server finished uploading files (state is ACTIVE)
            await _poll_until_active(async_client=client.aio, uploaded=uploaded, files=files)

        yield uploaded

    finally:
        if len(uploaded) > 0:
            await asyncio.gather(*[client.aio.files.delete(name=f.name) for f in uploaded], return_exceptions=True)


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
    return f'rate-limits:gemini:{model}'


@pxt.udf(is_deterministic=False)
async def generate_images(prompt: str, *, model: str, config: dict | None = None) -> PIL.Image.Image:
    """
    Generates images based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/imagen>

    Note: This function is for Imagen models only. For Gemini image-generation models (Nano Banana,
    e.g. `gemini-2.5-flash-image`), use [`generate_content`][pixeltable.functions.gemini.generate_content] instead.

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

        >>> tbl.add_computed_column(
        ...     response=generate_images(tbl.prompt, model='imagen-4.0-generate-001')
        ... )
    """
    if model.startswith('gemini-'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'Model {model!r} is a Gemini image-generation model (Nano Banana) and must be used via '
            f'`generate_content`, not `generate_images` (which is for Imagen models). See the '
            f'`generate_content` docstring for an example.',
        )

    env.Env.get().require_package('google.genai')
    from google.genai.types import GenerateImagesConfig

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_ = GenerateImagesConfig(**config) if config else None
    response = await _genai_client().aio.models.generate_images(model=model, prompt=prompt, config=config_)
    return response.generated_images[0].image._pil_image


@generate_images.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


def _pil_to_gemini_image(image: PIL.Image.Image) -> 'genai.types.Image':
    """Convert a PIL image to a Gemini API Image object."""
    from google.genai import types

    with io.BytesIO() as buffer:
        image.save(buffer, format='webp')
        return types.Image(image_bytes=buffer.getvalue(), mime_type='image/webp')


async def _poll_gemini_operation(operation: T) -> T:
    while not operation.done:
        await asyncio.sleep(3)
        operation = await _genai_client().aio.operations.get(operation)
    return operation


async def _generate_videos_impl(
    model: str, prompt: str | None, image: 'genai.types.Image | None', config: 'genai.types.GenerateVideosConfig | None'
) -> str:
    """Shared implementation for video generation: submit request, poll for completion, download result."""
    operation = await _genai_client().aio.models.generate_videos(model=model, prompt=prompt, image=image, config=config)

    try:
        operation = await asyncio.wait_for(_poll_gemini_operation(operation), timeout=300)
    except asyncio.TimeoutError as exc:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_TIMEOUT,
            f'Video generation timed out after 300 seconds for Gemini model {model!r}.',
            provider='gemini',
        ) from exc

    if operation.error:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, f'Video generation failed: {operation.error}', provider='gemini'
        )

    video = operation.response.generated_videos[0]

    video_bytes = await _genai_client().aio.files.download(file=video.video)  # type: ignore[arg-type]  # async overload missing Video type (sync version accepts it)
    assert video_bytes is not None

    output_path = TempStore.create_path(extension='.mp4')
    Path(output_path).write_bytes(video_bytes)
    return str(output_path)


@pxt.udf(is_deterministic=False)
async def generate_videos(
    prompt: str | None = None, image: PIL.Image.Image | None = None, *, model: str, config: dict | None = None
) -> pxt.Video:
    """
    Generates videos based on a text description and configuration. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/video>

    At least one of `prompt` or `image` must be provided. When `image` is a single image, it is used as the first
    frame of the generated video. When `image` is a list of images, they are used as reference images to guide the
    style or asset appearance throughout the video (Veo 3.1+). See the overloaded signature for details.

    Request throttling:
    Applies the rate limit set in the config (section `veo.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        prompt: A text description of the videos to generate.
        image: A single image to use as the first frame of the video, or as `images` a list of up to 3 reference images
            for Veo 3.1 (see overloaded signature).
        model: The model to use.
        config: Configuration for generation, corresponding to keyword arguments of
            `genai.types.GenerateVideosConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig>

    Returns:
        The generated video.

    Examples:
        Add a computed column that applies the model `veo-3.0-generate-001`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=generate_videos(tbl.prompt, model='veo-3.0-generate-001')
        ... )

        Use reference images with Veo 3.1 to guide video generation:

        >>> tbl.add_computed_column(
        ...     response=generate_videos(
        ...         tbl.prompt,
        ...         images=[tbl.ref_img1, tbl.ref_img2],
        ...         reference_types=['asset', 'asset'],
        ...         model='veo-3.1-generate-preview',
        ...     )
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    if prompt is None and image is None:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED, 'At least one of `prompt` or `image` must be provided.'
        )

    image_: types.Image | None = _pil_to_gemini_image(image) if image is not None else None
    config_ = types.GenerateVideosConfig(**config) if config else None

    return await _generate_videos_impl(model, prompt, image_, config_)


@generate_videos.overload
async def _(
    prompt: str | None = None,
    images: list[PIL.Image.Image] | None = None,
    *,
    model: str,
    config: dict | None = None,
    reference_types: list[Literal['style', 'asset']] | None = None,
) -> pxt.Video:
    """Overload that accepts a list of reference images for Veo 3.1+.

    Args:
        prompt: A text description of the videos to generate.
        images: A list of up to 3 reference images to guide style or asset appearance.
        model: The model to use.
        config: Configuration for generation.
        reference_types: A list of reference types corresponding to each image. Each must be one of
            `'style'` or `'asset'`. `'style'` is only supported on Vertex AI. If not provided, defaults to
            `'asset'` for all images.
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    if images is None:
        images = []

    if not images and prompt is None:
        raise excs.RequestError(
            excs.ErrorCode.MISSING_REQUIRED, 'At least one of `prompt` or `images` must be provided.'
        )
    if len(images) > 3:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'At most 3 reference images are allowed, but {len(images)} were provided.',
        )

    if reference_types is None:
        reference_types = ['asset'] * len(images)
    elif len(reference_types) != len(images):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'`reference_types` length ({len(reference_types)}) must match `images` length ({len(images)}).',
        )

    reference_images = [
        types.VideoGenerationReferenceImage(image=_pil_to_gemini_image(img), reference_type=ref_type)
        for img, ref_type in zip(images, reference_types)
    ]

    config_ = types.GenerateVideosConfig(**config) if config else types.GenerateVideosConfig()
    config_.reference_images = reference_images

    return await _generate_videos_impl(model, prompt, None, config_)


@generate_videos.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(is_deterministic=False)
async def generate_speech(text: str, *, model: str, voice: str, config: dict | None = None) -> pxt.Audio:
    """
    Generates speech audio from text using Gemini's text-to-speech capability. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/speech-generation>

    Request throttling:
    Applies the rate limit set in the config (section `gemini.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        text: The text to synthesize into speech.
        model: The model to use (e.g. `'gemini-2.5-flash-preview-tts'`).
        voice: The voice profile to use. Supported voices include `'Kore'`, `'Puck'`, `'Charon'`,
            `'Fenrir'`, `'Aoede'`, `'Leda'`, `'Orus'`, `'Zephyr'`, and others. See the
            [speech generation docs](https://ai.google.dev/gemini-api/docs/speech-generation) for the full list.
            Mutually exclusive with `voices`.
        voices: A mapping from speaker alias (as used in the text) to voice name. For example,
            `{'Alice': 'Kore', 'Bob': 'Puck'}`. Mutually exclusive with `voice`.
        config: Additional configuration, corresponding to keyword arguments of
            `genai.types.GenerateContentConfig`. Keys such as `response_modalities` and `speech_config`
            are set automatically and should not be included.

    Returns:
        An audio file (WAV, 24 kHz mono 16-bit) containing the synthesized speech.

    Examples:
        Add a computed column that generates speech from text:

        >>> tbl.add_computed_column(
        ...     audio=generate_speech(
        ...         tbl.text, model='gemini-2.5-flash-preview-tts', voice='Kore'
        ...     )
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_ = types.GenerateContentConfig(**(config or {}))
    config_.response_modalities = ['AUDIO']
    config_.speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
    )

    response = await _genai_client().aio.models.generate_content(model=model, contents=text, config=config_)
    try:
        data = response.candidates[0].content.parts[0].inline_data.data
    except (IndexError, AttributeError) as exc:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Gemini TTS returned unexpected response structure for model {model}.',
            provider='gemini',
        ) from exc
    if isinstance(data, str):
        data = base64.b64decode(data)

    output_path = str(TempStore.create_path(extension='.wav'))
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(data)
    return output_path


@generate_speech.overload
async def _(text: str, *, model: str, voices: dict[str, str], config: dict | None = None) -> pxt.Audio:
    """
    Multi-speaker variant: `voices` is a dict mapping speaker aliases to voice names.
    The input text should contain speaker labels matching the dict keys, e.g.
    `"Alice: Hello! Bob: Hi there!"`.

    Args:
        text: The text to synthesize, with speaker labels matching the keys of `voices`.
        model: The model to use (e.g. `'gemini-2.5-flash-preview-tts'`).
        voices: A mapping from speaker alias (as used in the text) to voice name. For example,
            `{'Alice': 'Kore', 'Bob': 'Puck'}`.
        config: Additional configuration, corresponding to keyword arguments of
            `genai.types.GenerateContentConfig`. Keys such as `response_modalities` and `speech_config`
            are set automatically and should not be included.

    Returns:
        An audio file (WAV, 24 kHz mono 16-bit) containing the synthesized multi-speaker speech.

    Examples:
        >>> tbl.add_computed_column(
        ...     audio=generate_speech(
        ...         tbl.dialogue,
        ...         model='gemini-2.5-flash-preview-tts',
        ...         voices={'Alice': 'Kore', 'Bob': 'Puck'},
        ...     )
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_ = types.GenerateContentConfig(**(config or {}))
    config_.response_modalities = ['AUDIO']
    config_.speech_config = types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker=alias,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    ),
                )
                for alias, voice_name in voices.items()
            ]
        )
    )

    response = await _genai_client().aio.models.generate_content(model=model, contents=text, config=config_)
    try:
        data = response.candidates[0].content.parts[0].inline_data.data
    except (IndexError, AttributeError) as exc:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Gemini multi-speaker TTS returned unexpected response structure for model {model}.',
            provider='gemini',
        ) from exc
    if isinstance(data, str):
        data = base64.b64decode(data)

    output_path = str(TempStore.create_path(extension='.wav'))
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(data)
    return output_path


@generate_speech.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(is_deterministic=False)
async def transcribe(audio: pxt.Audio, *, model: str, prompt: str, config: dict | None = None) -> str:
    """
    Transcribes audio to text using Gemini's audio understanding capability. For additional details, see:
    <https://ai.google.dev/gemini-api/docs/audio>

    Request throttling:
    Applies the rate limit set in the config (section `gemini.rate_limits`; use the model id as the key). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install google-genai`

    Args:
        audio: The audio file to transcribe.
        model: The model to use (e.g. `'gemini-2.5-flash'`).
        prompt: The instruction prompt sent alongside the audio. For example,
            `'Generate a transcript of the speech.'` or `'Summarize the audio content.'`.
        config: Additional configuration, corresponding to keyword arguments of
            `genai.types.GenerateContentConfig`.

    Returns:
        The transcribed text.

    Examples:
        Add a computed column that transcribes audio:

        >>> tbl.add_computed_column(
        ...     transcript=transcribe(tbl.audio, model='gemini-2.5-flash')
        ... )
    """
    env.Env.get().require_package('google.genai')
    from google.genai import types

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    config_ = types.GenerateContentConfig(**config) if config else None
    client = _genai_client()

    try:
        size_bytes = os.stat(audio).st_size
    except OSError as exc:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT, f"Error accessing audio file '{audio}': {exc.strerror or exc}"
        ) from exc
    if size_bytes > GEMINI_INLINE_LIMIT_BYTES:
        async with _gemini_file_uploads([audio]) as uploaded:
            audio_part = types.Part.from_uri(file_uri=uploaded[0].uri, mime_type=uploaded[0].mime_type)
            response = await client.aio.models.generate_content(
                model=model,
                contents=[audio_part, prompt],  # type: ignore[arg-type]
                config=config_,
            )
    else:
        mime_type, _ = mimetypes.guess_type(audio, strict=False)
        if mime_type is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_DATA_FORMAT, f'Could not identify mime type of file: {audio}'
            )
        try:
            audio_data = Path(audio).read_bytes()
        except (OSError, ValueError) as exc:
            raise excs.RequestError(excs.ErrorCode.INVALID_DATA_FORMAT, f'Failed to read audio file: {audio}') from exc
        audio_part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)
        response = await client.aio.models.generate_content(
            model=model,
            contents=[audio_part, prompt],  # type: ignore[arg-type]
            config=config_,
        )

    return response.text


@transcribe.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


@pxt.udf(batch_size=4)
async def embed_content(
    contents: Batch[str], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    """
    Generate embeddings for text, images, video, and other content. For more information on Gemini embeddings API, see:
    <https://ai.google.dev/gemini-api/docs/embeddings>

    __Requirements:__

    - `pip install google-genai`

    Args:
        contents: The string, image, audio, video, or document to embed.
        model: The Gemini model to use.
        config: Configuration for embedding generation, corresponding to keyword arguments of
            `genai.types.EmbedContentConfig`. For details on the parameters, see:
            <https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig>

    Returns:
        The corresponding embedding vector.

    Examples:
        Add a computed column with embeddings to an existing table with a `text` column:

        >>> t.add_computed_column(
        ...     embedding=embed_content(t.text, model='gemini-embedding-001')
        ... )

        Add an embedding index on `text` column:

        >>> t.add_embedding_index(
        ...     t.text, embedding=embed_content.using(model='gemini-embedding-001')
        ... )
    """
    return await _embed_content(contents, model, config)


@embed_content.overload
async def _(
    contents: Batch[PIL.Image.Image], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_content(contents, model, config)


@embed_content.overload
async def _(
    contents: Batch[pxt.Audio], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config)


@embed_content.overload
async def _(
    contents: Batch[pxt.Video], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config)


@embed_content.overload
async def _(
    contents: Batch[pxt.Document], *, model: str, config: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), np.float32]]:
    return await _embed_file_content(contents, model, config)


async def _embed_file_content(
    contents: list[str], model: str, config: dict[str, Any] | None
) -> Batch[pxt.Array[(None,), np.float32]]:
    env.Env.get().require_package('google.genai')
    from google.genai import types

    large_files: list[str] = []
    for item in contents:
        size_bytes = os.stat(item).st_size
        if size_bytes > GEMINI_INLINE_LIMIT_BYTES:
            large_files.append(item)

    async with _gemini_file_uploads(large_files) as uploaded:
        upload_map = dict(zip(large_files, uploaded))
        contents_: list[types.ContentUnion] = []
        for item in contents:
            if item in upload_map:
                contents_.append(upload_map[item])
            else:
                mime_type, _ = mimetypes.guess_type(item, strict=False)
                if mime_type is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT, f'Could not identify mime type of file: {item}'
                    )

                try:
                    # TODO: Do this on a background thread.
                    data = Path(item).read_bytes()
                except (OSError, ValueError) as exc:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_DATA_FORMAT, f'Error reading file for embedding: {item}'
                    ) from exc

                contents_.append(types.Part.from_bytes(data=data, mime_type=mime_type))

        return await _embed_content(contents_, model, config)


async def _embed_content(
    contents: Sequence['genai.types.ContentUnion'], model: str, config: dict[str, Any] | None
) -> Batch[pxt.Array[(None,), np.float32]]:
    env.Env.get().require_package('google.genai')

    resource_pool_id = f'rate-limits:gemini:{model}'
    env.Env.get().get_resource_pool_info(resource_pool_id, GeminiRateLimitsInfo)

    client = _genai_client()
    config_ = _embedding_config(config)

    result = await client.aio.models.embed_content(model=model, contents=list(contents), config=config_)
    if len(result.embeddings) != len(contents):
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'Unexpected response from Gemini server: number of embeddings returned ({len(result.embeddings)}) '
            f'does not match request batch size ({len(contents)}).',
            provider='gemini',
        )
    return [np.array(emb.values, dtype=np.float32) for emb in result.embeddings]

    # Batch API
    # batch_job = client.batches.create_embeddings(
    #     model=model,
    #     src=types.EmbeddingsBatchJobSource(
    #         inlined_requests=types.EmbedContentBatch(contents=contents, config=config_)
    #     ),
    # )

    # await asyncio.sleep(3)
    # i = 0
    # while True:
    #     batch_job = client.batches.get(name=batch_job.name)
    #     if batch_job.state in (
    #         types.JobState.JOB_STATE_SUCCEEDED,
    #         types.JobState.JOB_STATE_FAILED,
    #         types.JobState.JOB_STATE_CANCELLED,
    #         types.JobState.JOB_STATE_EXPIRED,
    #     ):
    #         break
    #     delay = min(10 + i * 2, 30)
    #     _logger.debug(
    #         f'Waiting for embedding batch job {batch_job.name} to complete. Latest state: {batch_job.state}. Sleeping'
    #         f' for {delay}s before the next attempt.'
    #     )
    #     await asyncio.sleep(delay)
    #     i += 1

    # if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
    #     raise excs.ExternalServiceError(
    #         excs.ErrorCode.PROVIDER_ERROR,
    #         f'Embedding batch job did not succeed: {batch_job.state}. Error: {batch_job.error}',
    #         provider='gemini',
    #     )

    # assert batch_job.error is None
    # results = []
    # for resp in batch_job.dest.inlined_embed_content_responses:
    #     assert resp.error is None
    #     results.append(np.array(resp.response.embedding.values, dtype=np.float32))
    # return results


_DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL: dict[str, int] = {
    'gemini-embedding-001': 3072,
    'gemini-embedding-2-preview': 3072,
}


@embed_content.conditional_return_type
def _(model: str, config: dict | None) -> ts.ArrayType:
    config_ = _embedding_config(config)
    dim = config_.output_dimensionality
    if dim is None and model in _DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL:
        dim = _DEFAULT_EMBEDDING_DIMENSIONALITY_BY_MODEL.get(model)
    return ts.ArrayType((dim,), dtype=np.dtype('float32'), nullable=False)


@embed_content.resource_pool
def _(model: str) -> str:
    return f'rate-limits:gemini:{model}'


def _embedding_config(config: dict | None) -> 'genai.types.EmbedContentConfig':
    from google.genai import types

    return types.EmbedContentConfig(**config) if config else types.EmbedContentConfig()


def _is_processing(remote_files: list['genai.types.File']) -> bool:
    from google.genai import types

    return any(file.state != types.FileState.ACTIVE for file in remote_files)


def _handle_polling_timeout(retry_state: RetryCallState) -> None:
    """Triggered when timeout is reached."""
    from google.genai import types

    remote_files: list[types.File] = retry_state.outcome.result()

    # Extract files from the keyword arguments
    files: list[str] = retry_state.kwargs.get('files', [])
    stuck_details = []
    for i, file in enumerate(remote_files):
        if file.state != types.FileState.ACTIVE:
            local_path = files[i] if i < len(files) else 'Unknown path'
            stuck_details.append(f'{local_path} (ID: {file.name}, State: {file.state.name})')

    detail_str = '\n- '.join(stuck_details)
    raise excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_TIMEOUT,
        f'Timeout: failed to upload {len(stuck_details)}/{len(remote_files)} file(s):\n- {detail_str}',
        provider='gemini',
    )


@retry(
    retry=retry_if_result(_is_processing),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_delay(600),
    retry_error_callback=_handle_polling_timeout,
)
async def _poll_until_active(
    async_client: 'genai.client.AsyncClient', uploaded: list['genai.types.File'], files: list[str]
) -> list['genai.types.File']:
    from google.genai import types

    # Collect statuses for all uploaded files
    remote_files = await asyncio.gather(*[async_client.files.get(name=f.name) for f in uploaded])
    for i, file in enumerate(remote_files):
        if file.state == types.FileState.FAILED:
            # Fail immediately
            raise excs.ExternalServiceError(
                excs.ErrorCode.PROVIDER_ERROR,
                f'Server processing failed for {files[i]} ({file.name})',
                provider='gemini',
            )
    return remote_files


def _process_media_contents(data: Any, large_video_paths: list[str]) -> Any:
    """
    Recursively traverse a nested content structure (dict/list/str) and process video file paths.

    - Strings that are not local video file paths are returned unchanged.
    - Small video files (<= GEMINI_INLINE_LIMIT_BYTES * 0.75) are base64-encoded inline.
    - Large video files are queued for async upload and replaced with a placeholder dict
      (keyed by _UPLOAD_PLACEHOLDER_KEY) to be resolved later by _replace_upload_placeholders.

    Returns the same nested structure with video path strings replaced by inline_data or placeholder dicts.
    """
    if isinstance(data, dict):
        return {k: _process_media_contents(v, large_video_paths) for k, v in data.items()}
    if isinstance(data, list):
        return [_process_media_contents(v, large_video_paths) for v in data]
    if isinstance(data, str):
        # Check if string is a file path containing video
        mime_type, _ = mimetypes.guess_type(data, strict=False)
        if mime_type is None or not mime_type.lower().startswith('video/'):
            return data
        local_path = Path(data).expanduser()
        try:
            if not local_path.exists():
                return data
        except (OSError, ValueError):
            return data
        size_bytes = local_path.stat().st_size
        if size_bytes <= GEMINI_INLINE_LIMIT_BYTES * 3 // 4:  # scale by 0.75 to account for base64 expansion
            # TODO: Do this on a background thread.
            data_b64 = base64.b64encode(local_path.read_bytes()).decode('utf-8')
            return {'inline_data': {'mime_type': mime_type, 'data': data_b64}}
        # Record the large file for upload and insert a placeholder to be resolved later
        large_video_paths.append(str(local_path))
        return {_UPLOAD_PLACEHOLDER_KEY: {'task_id': len(large_video_paths) - 1, 'mime_type': mime_type}}
    return data


def _replace_upload_placeholders(obj: Any, uploaded: list['genai.types.File']) -> Any:
    """
    Recursively traverse a nested content structure (dict/list/str) and resolve upload placeholders.

    Returns the same nested structure with all placeholders replaced by file_data dicts.
    """
    if isinstance(obj, dict) and _UPLOAD_PLACEHOLDER_KEY in obj:
        idx = obj[_UPLOAD_PLACEHOLDER_KEY]['task_id']
        mime_type = obj[_UPLOAD_PLACEHOLDER_KEY]['mime_type']
        f = uploaded[idx]
        return {'file_data': {'file_uri': f.uri, 'mime_type': mime_type}}
    if isinstance(obj, dict):
        return {k: _replace_upload_placeholders(v, uploaded) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_upload_placeholders(v, uploaded) for v in obj]
    return obj


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
