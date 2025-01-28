"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the OpenAI API. In order to use them, you must
first `pip install openai` and configure your OpenAI credentials, as described in
the [Working with OpenAI](https://pixeltable.readme.io/docs/working-with-openai) tutorial.
"""

import base64
import datetime
import io
import json
import logging
import pathlib
import re
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union, cast

import httpx
import numpy as np
import PIL.Image
import tenacity

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Batch, Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import openai

_logger = logging.getLogger('pixeltable')


@env.register_client('openai')
def _(api_key: str) -> tuple['openai.OpenAI', 'openai.AsyncOpenAI']:
    import openai

    return (
        openai.OpenAI(api_key=api_key),
        openai.AsyncOpenAI(
            api_key=api_key,
            # recommended to increase limits for async client to avoid connection errors
            http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
        ),
    )


def _openai_client() -> 'openai.OpenAI':
    return env.Env.get().get_client('openai')[0]


def _async_openai_client() -> 'openai.AsyncOpenAI':
    return env.Env.get().get_client('openai')[1]


# Exponential backoff decorator using tenacity.
# TODO(aaron-siegel): Right now this hardwires random exponential backoff with defaults suggested
# by OpenAI. Should we investigate making this more customizable in the future?
def _retry(fn: Callable) -> Callable:
    import openai

    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(openai.RateLimitError),
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(20),
    )(fn)


# models that share rate limits; see https://platform.openai.com/settings/organization/limits for details
_shared_rate_limits = {
    'gpt-4-turbo': [
        'gpt-4-turbo',
        'gpt-4-turbo-latest',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-turbo-preview',
        'gpt-4-0125-preview',
        'gpt-4-1106-preview',
    ],
    'gpt-4o': [
        'gpt-4o',
        'gpt-4o-latest',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
        'gpt-4o-2024-11-20',
        'gpt-4o-audio-preview',
        'gpt-4o-audio-preview-2024-10-01',
        'gpt-4o-audio-preview-2024-12-17',
    ],
    'gpt-4o-mini': [
        'gpt-4o-mini',
        'gpt-4o-mini-latest',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini-audio-preview',
        'gpt-4o-mini-audio-preview-2024-12-17',
    ],
    'gpt-4o-mini-realtime-preview': [
        'gpt-4o-mini-realtime-preview',
        'gpt-4o-mini-realtime-preview-latest',
        'gpt-4o-mini-realtime-preview-2024-12-17',
    ],
}


def _resource_pool(model: str) -> str:
    for model_family, models in _shared_rate_limits.items():
        if model in models:
            return f'rate-limits:openai:{model_family}'
    return f'rate-limits:openai:{model}'


class OpenAIRateLimitsInfo(env.RateLimitsInfo):
    retryable_errors: tuple[Type[Exception], ...]

    def __init__(self, get_request_resources: Callable[..., dict[str, int]]):
        super().__init__(get_request_resources)
        import openai

        self.retryable_errors = (
            # ConnectionError: we occasionally see this error when the AsyncConnectionPool is trying to close
            # expired connections
            # (AsyncConnectionPool._close_expired_connections() fails with ConnectionError when executing
            # 'await connection.aclose()', which is potentially a bug in AsyncConnectionPool)
            openai.APIConnectionError,
            # the following errors are retryable according to OpenAI's API documentation
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.UnprocessableEntityError,
            openai.InternalServerError,
        )

    def get_retry_delay(self, exc: Exception) -> Optional[float]:
        import openai

        if not isinstance(exc, self.retryable_errors):
            return None
        assert isinstance(exc, openai.APIError)
        return 1.0


# RE pattern for duration in '*-reset' headers;
# examples: 1d2h3ms, 4m5.6s; # fractional seconds can be reported as 0.5s or 500ms
_header_duration_pattern = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)ms)|(?:(\d+)m)?(?:([\d.]+)s)?')


def _parse_header_duration(duration_str):
    match = _header_duration_pattern.match(duration_str)
    if not match:
        raise ValueError('Invalid duration format')

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    milliseconds = int(match.group(3) or 0)
    minutes = int(match.group(4) or 0)
    seconds = float(match.group(5) or 0)

    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def _get_header_info(
    headers: httpx.Headers, *, requests: bool = True, tokens: bool = True
) -> tuple[Optional[tuple[int, int, datetime.datetime]], Optional[tuple[int, int, datetime.datetime]]]:
    assert requests or tokens
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    requests_info: Optional[tuple[int, int, datetime.datetime]] = None
    if requests:
        requests_limit_str = headers.get('x-ratelimit-limit-requests')
        requests_limit = int(requests_limit_str) if requests_limit_str is not None else None
        requests_remaining_str = headers.get('x-ratelimit-remaining-requests')
        requests_remaining = int(requests_remaining_str) if requests_remaining_str is not None else None
        requests_reset_str = headers.get('x-ratelimit-reset-requests')
        requests_reset_ts = now + _parse_header_duration(requests_reset_str)
        requests_info = (requests_limit, requests_remaining, requests_reset_ts)

    tokens_info: Optional[tuple[int, int, datetime.datetime]] = None
    if tokens:
        tokens_limit_str = headers.get('x-ratelimit-limit-tokens')
        tokens_limit = int(tokens_limit_str) if tokens_limit_str is not None else None
        tokens_remaining_str = headers.get('x-ratelimit-remaining-tokens')
        tokens_remaining = int(tokens_remaining_str) if tokens_remaining_str is not None else None
        tokens_reset_str = headers.get('x-ratelimit-reset-tokens')
        tokens_reset_ts = now + _parse_header_duration(tokens_reset_str)
        tokens_info = (tokens_limit, tokens_remaining, tokens_reset_ts)

    return requests_info, tokens_info


#####################################
# Audio Endpoints


@pxt.udf
def speech(
    input: str, *, model: str, voice: str, response_format: Optional[str] = None, speed: Optional[float] = None
) -> pxt.Audio:
    """
    Generates audio from the input text.

    Equivalent to the OpenAI `audio/speech` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/text-to-speech](https://platform.openai.com/docs/guides/text-to-speech)

    __Requirements:__

    - `pip install openai`

    Args:
        input: The text to synthesize into speech.
        model: The model to use for speech synthesis.
        voice: The voice profile to use for speech synthesis. Supported options include:
            `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/audio/createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

    Returns:
        An audio file containing the synthesized speech.

    Examples:
        Add a computed column that applies the model `tts-1` to an existing Pixeltable column `tbl.text`
        of the table `tbl`:

        >>> tbl['audio'] = speech(tbl.text, model='tts-1', voice='nova')
    """
    content = _retry(_openai_client().audio.speech.create)(
        input=input, model=model, voice=voice, response_format=_opt(response_format), speed=_opt(speed)
    )
    ext = response_format or 'mp3'
    output_filename = str(env.Env.get().tmp_dir / f'{uuid.uuid4()}.{ext}')
    content.write_to_file(output_filename)
    return output_filename


@pxt.udf
def transcriptions(
    audio: pxt.Audio,
    *,
    model: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Transcribes audio into the input language.

    Equivalent to the OpenAI `audio/transcriptions` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)

    __Requirements:__

    - `pip install openai`

    Args:
        audio: The audio to transcribe.
        model: The model to use for speech transcription.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/audio/createTranscription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

    Returns:
        A dictionary containing the transcription and other metadata.

    Examples:
        Add a computed column that applies the model `whisper-1` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl['transcription'] = transcriptions(tbl.audio, model='whisper-1', language='en')
    """
    file = pathlib.Path(audio)
    transcription = _retry(_openai_client().audio.transcriptions.create)(
        file=file, model=model, language=_opt(language), prompt=_opt(prompt), temperature=_opt(temperature)
    )
    return transcription.dict()


@pxt.udf
def translations(
    audio: pxt.Audio, *, model: str, prompt: Optional[str] = None, temperature: Optional[float] = None
) -> dict:
    """
    Translates audio into English.

    Equivalent to the OpenAI `audio/translations` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)

    __Requirements:__

    - `pip install openai`

    Args:
        audio: The audio to translate.
        model: The model to use for speech transcription and translation.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/audio/createTranslation](https://platform.openai.com/docs/api-reference/audio/createTranslation)

    Returns:
        A dictionary containing the translation and other metadata.

    Examples:
        Add a computed column that applies the model `whisper-1` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl['translation'] = translations(tbl.audio, model='whisper-1', language='en')
    """
    file = pathlib.Path(audio)
    translation = _retry(_openai_client().audio.translations.create)(
        file=file, model=model, prompt=_opt(prompt), temperature=_opt(temperature)
    )
    return translation.dict()


#####################################
# Chat Endpoints


def _chat_completions_get_request_resources(
    messages: list, max_tokens: Optional[int], n: Optional[int]
) -> dict[str, int]:
    completion_tokens = n * max_tokens

    num_tokens = 0.0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(value) / 4
            if key == 'name':  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return {'requests': 1, 'tokens': int(num_tokens) + completion_tokens}


@pxt.udf
async def chat_completions(
    messages: list,
    *,
    model: str,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = 1024,
    n: Optional[int] = 1,
    presence_penalty: Optional[float] = None,
    response_format: Optional[dict] = None,
    seed: Optional[int] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[dict] = None,
    user: Optional[str] = None,
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the OpenAI `chat/completions` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/chat-completions](https://platform.openai.com/docs/guides/chat-completions)

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the OpenAI API documentation.
        model: The model to use for chat completion.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat)

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gpt-4o-mini` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': tbl.prompt}
            ]
            tbl['response'] = chat_completions(messages, model='gpt-4o-mini')
    """
    if tools is not None:
        tools = [{'type': 'function', 'function': tool} for tool in tools]

    tool_choice_: Union[str, dict, None] = None
    if tool_choice is not None:
        if tool_choice['auto']:
            tool_choice_ = 'auto'
        elif tool_choice['required']:
            tool_choice_ = 'required'
        else:
            assert tool_choice['tool'] is not None
            tool_choice_ = {'type': 'function', 'function': {'name': tool_choice['tool']}}

    extra_body: Optional[dict[str, Any]] = None
    if tool_choice is not None and not tool_choice['parallel_tool_calls']:
        extra_body = {'parallel_tool_calls': False}

    # make sure the pool info exists prior to making the request
    resource_pool = _resource_pool(model)
    rate_limits_info = env.Env.get().get_resource_pool_info(
        resource_pool, lambda: OpenAIRateLimitsInfo(_chat_completions_get_request_resources)
    )

    # cast(Any, ...): avoid mypy errors
    result = await _async_openai_client().chat.completions.with_raw_response.create(
        messages=messages,
        model=model,
        frequency_penalty=_opt(frequency_penalty),
        logit_bias=_opt(logit_bias),
        logprobs=_opt(logprobs),
        top_logprobs=_opt(top_logprobs),
        max_tokens=_opt(max_tokens),
        n=_opt(n),
        presence_penalty=_opt(presence_penalty),
        response_format=_opt(cast(Any, response_format)),
        seed=_opt(seed),
        stop=_opt(stop),
        temperature=_opt(temperature),
        top_p=_opt(top_p),
        tools=_opt(cast(Any, tools)),
        tool_choice=_opt(cast(Any, tool_choice_)),
        user=_opt(user),
        timeout=10,
        extra_body=extra_body,
    )

    requests_info, tokens_info = _get_header_info(result.headers)
    rate_limits_info.record(requests=requests_info, tokens=tokens_info)

    return json.loads(result.text)


@pxt.udf
def vision(prompt: str, image: PIL.Image.Image, *, model: str) -> str:
    """
    Analyzes an image with the OpenAI vision capability. This is a convenience function that takes an image and
    prompt, and constructs a chat completion request that utilizes OpenAI vision.

    For additional details, see: [https://platform.openai.com/docs/guides/vision](https://platform.openai.com/docs/guides/vision)

    __Requirements:__

    - `pip install openai`

    Args:
        prompt: A prompt for the OpenAI vision request.
        image: The image to analyze.
        model: The model to use for OpenAI vision.

    Returns:
        The response from the OpenAI vision API.

    Examples:
        Add a computed column that applies the model `gpt-4o-mini` to an existing Pixeltable column `tbl.image`
        of the table `tbl`:

        >>> tbl['response'] = vision("What's in this image?", tbl.image, model='gpt-4o-mini')
    """
    # TODO(aaron-siegel): Decompose CPU/GPU ops into separate functions
    bytes_arr = io.BytesIO()
    image.save(bytes_arr, format='png')
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    b64_encoded_image = b64_bytes.decode('utf-8')
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64_encoded_image}'}},
            ],
        }
    ]
    result = _retry(_openai_client().chat.completions.create)(messages=messages, model=model)
    return result.choices[0].message.content


#####################################
# Embeddings Endpoints

_embedding_dimensions_cache: dict[str, int] = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}


def _embeddings_get_request_resources(input: list[str]) -> dict[str, int]:
    input_len = sum(len(s) for s in input)
    return {'requests': 1, 'tokens': int(input_len / 4)}


@pxt.udf(batch_size=32)
async def embeddings(
    input: Batch[str], *, model: str, dimensions: Optional[int] = None, user: Optional[str] = None
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Creates an embedding vector representing the input text.

    Equivalent to the OpenAI `embeddings` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)

    __Requirements:__

    - `pip install openai`

    Args:
        input: The text to embed.
        model: The model to use for the embedding.
        dimensions: The vector length of the embedding. If not specified, Pixeltable will use
            a default value based on the model.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/embeddings](https://platform.openai.com/docs/api-reference/embeddings)

    Returns:
        An array representing the application of the given embedding to `input`.

    Examples:
        Add a computed column that applies the model `text-embedding-3-small` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl['embed'] = embeddings(tbl.text, model='text-embedding-3-small')
    """
    _logger.debug(f'embeddings: batch_size={len(input)}')
    resource_pool = _resource_pool(model)
    rate_limits_info = env.Env.get().get_resource_pool_info(
        resource_pool, lambda: OpenAIRateLimitsInfo(_embeddings_get_request_resources)
    )
    result = await _async_openai_client().embeddings.with_raw_response.create(
        input=input, model=model, dimensions=_opt(dimensions), user=_opt(user), encoding_format='float'
    )
    requests_info, tokens_info = _get_header_info(result.headers)
    rate_limits_info.record(requests=requests_info, tokens=tokens_info)
    return [np.array(data['embedding'], dtype=np.float64) for data in json.loads(result.content)['data']]


@embeddings.conditional_return_type
def _(model: str, dimensions: Optional[int] = None) -> pxt.ArrayType:
    if dimensions is None:
        if model not in _embedding_dimensions_cache:
            # TODO: find some other way to retrieve a sample
            return pxt.ArrayType((None,), dtype=pxt.FloatType(), nullable=False)
        dimensions = _embedding_dimensions_cache.get(model, None)
    return pxt.ArrayType((dimensions,), dtype=pxt.FloatType(), nullable=False)


#####################################
# Images Endpoints


@pxt.udf
def image_generations(
    prompt: str,
    *,
    model: str = 'dall-e-2',
    quality: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
) -> PIL.Image.Image:
    """
    Creates an image given a prompt.

    Equivalent to the OpenAI `images/generations` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/images](https://platform.openai.com/docs/guides/images)

    __Requirements:__

    - `pip install openai`

    Args:
        prompt: Prompt for the image.
        model: The model to use for the generations.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/images/create](https://platform.openai.com/docs/api-reference/images/create)

    Returns:
        The generated image.

    Examples:
        Add a computed column that applies the model `dall-e-2` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl['gen_image'] = image_generations(tbl.text, model='dall-e-2')
    """
    # TODO(aaron-siegel): Decompose CPU/GPU ops into separate functions
    result = _retry(_openai_client().images.generate)(
        prompt=prompt,
        model=_opt(model),
        quality=_opt(quality),
        size=_opt(size),
        style=_opt(style),
        user=_opt(user),
        response_format='b64_json',
    )
    b64_str = result.data[0].b64_json
    b64_bytes = base64.b64decode(b64_str)
    img = PIL.Image.open(io.BytesIO(b64_bytes))
    img.load()
    return img


@image_generations.conditional_return_type
def _(size: Optional[str] = None) -> pxt.ImageType:
    if size is None:
        return pxt.ImageType(size=(1024, 1024))
    x_pos = size.find('x')
    if x_pos == -1:
        return pxt.ImageType()
    try:
        width, height = int(size[:x_pos]), int(size[x_pos + 1 :])
    except ValueError:
        return pxt.ImageType()
    return pxt.ImageType(size=(width, height))


#####################################
# Moderations Endpoints


@pxt.udf
def moderations(input: str, *, model: str = 'omni-moderation-latest') -> dict:
    """
    Classifies if text is potentially harmful.

    Equivalent to the OpenAI `moderations` API endpoint.
    For additional details, see: [https://platform.openai.com/docs/guides/moderation](https://platform.openai.com/docs/guides/moderation)

    __Requirements:__

    - `pip install openai`

    Args:
        input: Text to analyze with the moderations model.
        model: The model to use for moderations.

    For details on the other parameters, see: [https://platform.openai.com/docs/api-reference/moderations](https://platform.openai.com/docs/api-reference/moderations)

    Returns:
        Details of the moderations results.

    Examples:
        Add a computed column that applies the model `text-moderation-stable` to an existing
        Pixeltable column `tbl.input` of the table `tbl`:

        >>> tbl['moderations'] = moderations(tbl.text, model='text-moderation-stable')
    """
    result = _retry(_openai_client().moderations.create)(input=input, model=_opt(model))
    return result.dict()


# @speech.resource_pool
# @transcriptions.resource_pool
# @translations.resource_pool
@chat_completions.resource_pool
# @vision.resource_pool
@embeddings.resource_pool
# @image_generations.resource_pool
# @moderations.resource_pool
def _(model: str) -> str:
    return _resource_pool(model)


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenAI response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


@pxt.udf
def _openai_response_to_pxt_tool_calls(response: dict) -> Optional[dict]:
    if 'tool_calls' not in response['choices'][0]['message'] or response['choices'][0]['message']['tool_calls'] is None:
        return None
    openai_tool_calls = response['choices'][0]['message']['tool_calls']
    return {
        tool_call['function']['name']: {'args': json.loads(tool_call['function']['arguments'])}
        for tool_call in openai_tool_calls
    }


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, 'openai.NotGiven']:
    import openai

    return arg if arg is not None else openai.NOT_GIVEN


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
