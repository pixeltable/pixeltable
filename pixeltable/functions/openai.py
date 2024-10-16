"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the OpenAI API. In order to use them, you must
first `pip install openai` and configure your OpenAI credentials, as described in
the [Working with OpenAI](https://pixeltable.readme.io/docs/working-with-openai) tutorial.
"""

import base64
import io
import pathlib
import uuid
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import numpy as np
import PIL.Image
import tenacity

import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import openai


@env.register_client('openai')
def _(api_key: str) -> 'openai.OpenAI':
    import openai
    return openai.OpenAI(api_key=api_key)


def _openai_client() -> 'openai.OpenAI':
    return env.Env.get().get_client('openai')


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
    audio: pxt.Audio,
    *,
    model: str,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None
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


@pxt.udf
def chat_completions(
    messages: list,
    *,
    model: str,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
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
    result = _retry(_openai_client().chat.completions.create)(
        messages=messages,
        model=model,
        frequency_penalty=_opt(frequency_penalty),
        logit_bias=_opt(logit_bias),
        logprobs=_opt(logprobs),
        top_logprobs=_opt(top_logprobs),
        max_tokens=_opt(max_tokens),
        n=_opt(n),
        presence_penalty=_opt(presence_penalty),
        response_format=_opt(response_format),
        seed=_opt(seed),
        stop=_opt(stop),
        temperature=_opt(temperature),
        top_p=_opt(top_p),
        tools=_opt(tools),
        tool_choice=_opt(tool_choice),
        user=_opt(user),
    )
    return result.dict()


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


@pxt.udf(batch_size=32)
def embeddings(
    input: Batch[str], *, model: str, dimensions: Optional[int] = None, user: Optional[str] = None
) -> Batch[pxt.Array[(None,), float]]:
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
    result = _retry(_openai_client().embeddings.create)(
        input=input, model=model, dimensions=_opt(dimensions), user=_opt(user), encoding_format='float'
    )
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


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
    model: Optional[str] = None,
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
def moderations(input: str, *, model: Optional[str] = None) -> dict:
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


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, 'openai.NotGiven']:
    import openai
    return arg if arg is not None else openai.NOT_GIVEN


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
